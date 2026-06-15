#!/usr/bin/env python3
"""Game-playing SPSA tuner for search parameters.

Tunes UCI search options (NnueScale, RfpBase, FutilityMargin1, FutilityMargin2,
DeltaMargin) by playing engine-vs-engine matches via fastchess.

Each iteration perturbs the current parameter vector in a random direction,
plays a match between theta+delta and theta-delta, then uses the score
difference to estimate the gradient.

Usage:
    uv run python testing/tune_search.py \\
        --engine ./zig-out/bin/uci \\
        --eval-file path/to/net.nnue \\
        [--games 64] [--iterations 500] [--tc 5+0.05] \\
        [--threads 1] [--concurrency 4] [--book path.pgn] \\
        [--resume] [--checkpoint testing/spsa_checkpoint.json]
"""

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

PARAMS = [
    {"name": "NnueScale", "default": 2, "min": 1, "max": 10, "c_scale": 1},
    {"name": "RfpBase", "default": 80, "min": 20, "max": 200, "c_scale": 8},
    {"name": "FutilityMargin1", "default": 300, "min": 50, "max": 800, "c_scale": 30},
    {"name": "FutilityMargin2", "default": 600, "min": 100, "max": 1500, "c_scale": 60},
    {"name": "DeltaMargin", "default": 200, "min": 50, "max": 600, "c_scale": 20},
]


def find_fastchess():
    path = shutil.which("fastchess")
    if not path:
        print("Error: fastchess not found on PATH.", file=sys.stderr)
        print(
            "Install from: https://github.com/Disservin/fastchess", file=sys.stderr
        )
        sys.exit(1)
    return path


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def theta_to_dict(theta):
    return {p["name"]: int(round(theta[i])) for i, p in enumerate(PARAMS)}


def make_uci_options(values, eval_file=None, threads=1):
    opts = []
    for name, val in values.items():
        opts.append(f"option.{name}={val}")
    if eval_file:
        opts.append(f"option.EvalFile={eval_file}")
    opts.append(f"option.Threads={threads}")
    opts.append("option.OwnBook=false")
    return opts


def run_match(fastchess, engine, plus_opts, minus_opts, games, tc, concurrency, book):
    rounds = (games + 1) // 2

    cmd = [fastchess]

    cmd += ["-engine", f"cmd={engine}", "name=plus", "proto=uci"]
    for opt in plus_opts:
        cmd.append(opt)

    cmd += ["-engine", f"cmd={engine}", "name=minus", "proto=uci"]
    for opt in minus_opts:
        cmd.append(opt)

    cmd += ["-each", f"tc={tc}", "restart=on", "timemargin=300"]

    if book:
        book_path = Path(book)
        if book_path.suffix == ".pgn":
            cmd += ["-openings", f"file={book}", "format=pgn", "order=random"]
        else:
            cmd += ["-openings", f"file={book}", "format=epd", "order=random"]

    cmd += [
        "-rounds",
        str(rounds),
        "-repeat",
        "-recover",
        "-concurrency",
        str(concurrency),
        "-draw",
        "movenumber=40",
        "movecount=10",
        "score=5",
        "-resign",
        "movecount=5",
        "score=1000",
        "-output",
        "format=cutechess",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return parse_score(proc.stdout)


def parse_score(output):
    """Parse fastchess output and return score from plus's perspective.

    Score = (wins + draws/2) / games. Returns 0.5 on parse failure.
    """
    score_re = re.compile(
        r"Score of plus vs minus: (\d+) - (\d+) - (\d+)"
    )
    for line in output.splitlines():
        m = score_re.search(line)
        if m:
            wins = int(m.group(1))
            losses = int(m.group(2))
            draws = int(m.group(3))
            total = wins + losses + draws
            if total == 0:
                return 0.5
            return (wins + draws / 2) / total
    return 0.5


def spsa_step_sizes(k, a, c, alpha, gamma, big_a):
    a_k = a / (k + big_a) ** alpha
    c_k = c / k**gamma
    return a_k, c_k


def save_checkpoint(path, theta, iteration, history):
    data = {
        "theta": list(theta),
        "iteration": iteration,
        "params": [p["name"] for p in PARAMS],
        "history": history,
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_checkpoint(path):
    data = json.loads(Path(path).read_text())
    return data["theta"], data["iteration"], data.get("history", [])


def print_header():
    print(f"{'iter':>5}  ", end="")
    for p in PARAMS:
        print(f"{p['name']:>16}", end="")
    print(f"  {'score+':>7} {'score-':>7}")
    print("-" * (7 + 16 * len(PARAMS) + 16))


def print_row(k, theta, score_plus, score_minus):
    print(f"{k:5d}  ", end="")
    for i in range(len(PARAMS)):
        print(f"{theta[i]:16.1f}", end="")
    print(f"  {score_plus:7.3f} {score_minus:7.3f}")


def main():
    parser = argparse.ArgumentParser(description="SPSA tuner for search parameters")
    parser.add_argument("--engine", required=True, help="Path to UCI engine binary")
    parser.add_argument("--eval-file", default=None, help="Path to .nnue eval file")
    parser.add_argument("--games", type=int, default=64, help="Games per SPSA step")
    parser.add_argument(
        "--iterations", type=int, default=500, help="Number of SPSA iterations"
    )
    parser.add_argument("--tc", default="5+0.05", help="Time control")
    parser.add_argument("--threads", type=int, default=1, help="Engine threads")
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Parallel games in fastchess"
    )
    parser.add_argument("--book", default=None, help="Opening book path (PGN or EPD)")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        default="testing/spsa_checkpoint.json",
        help="Checkpoint file path",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    fastchess = find_fastchess()

    if args.seed is not None:
        random.seed(args.seed)

    # SPSA hyperparameters
    a = 1.0
    c = 1.0
    alpha = 0.602
    gamma = 0.101
    big_a = args.iterations * 0.1

    # Initialize theta
    if args.resume and Path(args.checkpoint).exists():
        theta, start_iter, history = load_checkpoint(args.checkpoint)
        start_iter += 1
        print(f"Resumed from iteration {start_iter - 1}")
    else:
        theta = [float(p["default"]) for p in PARAMS]
        start_iter = 1
        history = []

    print(f"\nSPSA Search Parameter Tuner")
    print(f"  Engine:     {args.engine}")
    print(f"  Eval file:  {args.eval_file or '(none)'}")
    print(f"  Games/step: {args.games}")
    print(f"  Iterations: {args.iterations}")
    print(f"  TC:         {args.tc}")
    print(f"  Threads:    {args.threads}")
    print(f"  Concurrency:{args.concurrency}")
    print()

    print("Starting values:")
    for i, p in enumerate(PARAMS):
        print(f"  {p['name']:>16} = {theta[i]:.0f}")
    print()

    print_header()

    for k in range(start_iter, args.iterations + 1):
        a_k, c_k = spsa_step_sizes(k, a, c, alpha, gamma, big_a)

        # Bernoulli perturbation
        delta = [random.choice([-1, 1]) for _ in PARAMS]

        # Perturbed parameter vectors
        theta_plus = []
        theta_minus = []
        for i, p in enumerate(PARAMS):
            pert = c_k * p["c_scale"] * delta[i]
            tp = clamp(round(theta[i] + pert), p["min"], p["max"])
            tm = clamp(round(theta[i] - pert), p["min"], p["max"])
            theta_plus.append(float(tp))
            theta_minus.append(float(tm))

        plus_vals = {p["name"]: int(theta_plus[i]) for i, p in enumerate(PARAMS)}
        minus_vals = {p["name"]: int(theta_minus[i]) for i, p in enumerate(PARAMS)}

        plus_opts = make_uci_options(plus_vals, args.eval_file, args.threads)
        minus_opts = make_uci_options(minus_vals, args.eval_file, args.threads)

        score_plus = run_match(
            fastchess,
            args.engine,
            plus_opts,
            minus_opts,
            args.games,
            args.tc,
            args.concurrency,
            args.book,
        )
        score_minus = 1.0 - score_plus

        # Gradient estimate and update
        for i, p in enumerate(PARAMS):
            pert = c_k * p["c_scale"] * delta[i]
            if abs(pert) < 1e-9:
                continue
            g_i = (score_plus - score_minus) / (2 * pert)
            a_scale = p["c_scale"]
            theta[i] -= a_k * a_scale * g_i
            theta[i] = clamp(theta[i], float(p["min"]), float(p["max"]))

        print_row(k, theta, score_plus, score_minus)

        history.append(
            {
                "iteration": k,
                "theta": list(theta),
                "score_plus": score_plus,
                "score_minus": score_minus,
            }
        )

        if k % 10 == 0:
            save_checkpoint(args.checkpoint, theta, k, history)

    # Final checkpoint
    save_checkpoint(args.checkpoint, theta, args.iterations, history)

    # Summary
    print()
    print("=" * 50)
    print("  SPSA Tuning Complete")
    print("=" * 50)
    print()
    print("Final values:")
    for i, p in enumerate(PARAMS):
        val = int(round(theta[i]))
        diff = val - p["default"]
        sign = "+" if diff >= 0 else ""
        print(f"  {p['name']:>16} = {val:>5}  (default {p['default']}, {sign}{diff})")

    print()
    print("UCI setoption commands:")
    for i, p in enumerate(PARAMS):
        val = int(round(theta[i]))
        print(f"  setoption name {p['name']} value {val}")
    print()


if __name__ == "__main__":
    main()
