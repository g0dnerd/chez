#!/usr/bin/env python3
"""Absolute Elo estimation via gauntlet against rated engines.

Run an engine against a set of opponents with known ratings (e.g. CCRL) using
fastchess, then compute a weighted-average absolute rating estimate.

Usage:
  python gauntlet.py                               # Run with gauntlet.json
  python gauntlet.py --config my_config.json       # Custom config
  python gauntlet.py --init-config                 # Generate template config
  python gauntlet.py --engine /path/to/uci         # Use pre-built binary
"""

import argparse
import atexit
import json
import math
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BUILDS_DIR = SCRIPT_DIR / ".builds"
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_CONFIG = SCRIPT_DIR / "gauntlet.json"

_active_worktrees: list[Path] = []


def cleanup_worktrees():
    for wt in _active_worktrees:
        if wt.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt)],
                cwd=PROJECT_ROOT,
                capture_output=True,
            )


atexit.register(cleanup_worktrees)

_orig_sigint = signal.getsignal(signal.SIGINT)
_orig_sigterm = signal.getsignal(signal.SIGTERM)


def _signal_handler(signum, frame):
    cleanup_worktrees()
    if signum == signal.SIGINT and callable(_orig_sigint):
        _orig_sigint(signum, frame)
    elif signum == signal.SIGTERM and callable(_orig_sigterm):
        _orig_sigterm(signum, frame)
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# -- Utilities ----------------------------------------------------------------


def run(cmd, **kwargs):
    kwargs.setdefault("check", True)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    return subprocess.run(cmd, **kwargs)


def git(*args, **kwargs):
    return run(["git", *args], cwd=PROJECT_ROOT, **kwargs)


def find_fastchess():
    path = shutil.which("fastchess")
    if not path:
        print("Error: fastchess not found on PATH.", file=sys.stderr)
        print(
            "Install it from: https://github.com/Disservin/fastchess", file=sys.stderr
        )
        sys.exit(1)
    return path


# -- Build --------------------------------------------------------------------


def build_current():
    build_dir = BUILDS_DIR / "gauntlet-current"
    binary = build_dir / "uci"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("  Building Chez from working tree...")
    try:
        run(["zig", "build"], cwd=PROJECT_ROOT, capture_output=False)
    except subprocess.CalledProcessError:
        print("Error: zig build failed", file=sys.stderr)
        sys.exit(1)

    built = PROJECT_ROOT / "zig-out" / "bin" / "uci"
    if not built.exists():
        print(f"Error: binary not found at {built}", file=sys.stderr)
        sys.exit(1)
    shutil.copy2(built, binary)
    print(f"  Built: {binary}")
    return binary


def build_from_commit(ref):
    try:
        result = git("rev-parse", "--verify", ref)
        commit = result.stdout.strip()
    except subprocess.CalledProcessError:
        print(f"Error: '{ref}' is not a valid git ref.", file=sys.stderr)
        sys.exit(1)

    short = commit[:7]
    build_dir = BUILDS_DIR / f"gauntlet-{short}"
    binary = build_dir / "uci"
    if binary.exists():
        print(f"  Using cached build: {short}")
        return binary

    build_dir.mkdir(parents=True, exist_ok=True)
    wt_dir = BUILDS_DIR / f".worktree-gauntlet-{short}"
    if wt_dir.exists():
        git("worktree", "remove", "--force", str(wt_dir))

    print(f"  Checking out {short} into worktree...")
    git("worktree", "add", "--detach", str(wt_dir), commit)
    _active_worktrees.append(wt_dir)

    try:
        run(
            ["git", "submodule", "update", "--init"],
            cwd=wt_dir,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        pass

    print(f"  Building {short}...")
    try:
        run(["zig", "build"], cwd=wt_dir, capture_output=False)
    except subprocess.CalledProcessError:
        print(f"Error: zig build failed for {short}", file=sys.stderr)
        sys.exit(1)

    built = wt_dir / "zig-out" / "bin" / "uci"
    if not built.exists():
        print(f"Error: binary not found at {built}", file=sys.stderr)
        sys.exit(1)
    shutil.copy2(built, binary)

    git("worktree", "remove", "--force", str(wt_dir))
    _active_worktrees.remove(wt_dir)

    print(f"  Built: {binary}")
    return binary


# -- Elo math -----------------------------------------------------------------


def score_to_elo(score):
    """Convert score fraction (0-1) to Elo difference."""
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_error_95(wins, draws, losses):
    """95% confidence interval for Elo difference via delta method."""
    n = wins + draws + losses
    if n == 0:
        return float("inf")

    ws = wins / n
    ds = draws / n
    ls = losses / n
    score = ws + ds / 2

    if score <= 0.0 or score >= 1.0:
        return float("inf")

    # Trinomial variance of the score
    var_score = (ws * (1 - score) ** 2 + ds * (0.5 - score) ** 2 + ls * score**2) / n
    if var_score <= 0:
        return float("inf")

    # Propagate through Elo formula: dElo/dScore = 400 / (ln10 * s * (1-s))
    dEdS = 400.0 / (math.log(10) * score * (1 - score))
    return 1.96 * abs(dEdS) * math.sqrt(var_score)


def compute_estimates(match_results):
    """Compute per-opponent estimates and inverse-variance weighted average.

    match_results: list of dicts with keys: name, rating, wins, draws, losses, games.
    Returns (weighted_avg, ci_95, per_opponent_details).
    """
    estimates = []

    for m in match_results:
        n = m["games"]
        if n == 0:
            estimates.append(
                {
                    **m,
                    "score_pct": 0,
                    "elo_diff": 0,
                    "ci_95": float("inf"),
                    "estimate": m["rating"],
                }
            )
            continue

        ws, ds, ls = m["wins"], m["draws"], m["losses"]
        score = (ws + ds / 2) / n
        elo_diff = score_to_elo(score)
        ci = elo_error_95(ws, ds, ls)
        est = m["rating"] + elo_diff

        estimates.append(
            {
                **m,
                "score_pct": score * 100,
                "elo_diff": elo_diff,
                "ci_95": ci,
                "estimate": est,
            }
        )

    # Inverse-variance weighted average
    valid = [e for e in estimates if 0 < e["ci_95"] < float("inf")]

    if not valid:
        if estimates:
            avg = sum(e["estimate"] for e in estimates) / len(estimates)
            return avg, float("inf"), estimates
        return 0, float("inf"), estimates

    total_inv_var = 0.0
    weighted_sum = 0.0
    for e in valid:
        sigma = e["ci_95"] / 1.96
        inv_var = 1.0 / (sigma**2)
        total_inv_var += inv_var
        weighted_sum += e["estimate"] * inv_var

    weighted_avg = weighted_sum / total_inv_var
    combined_ci = 1.96 / math.sqrt(total_inv_var)

    return weighted_avg, combined_ci, estimates


def run_match(
    fastchess,
    chez_binary,
    opponent,
    tc,
    rounds,
    threads,
    concurrency,
    openings=None,
    pgn_out=None,
    eval_file="/home/paul/projects/chez/data/net.nnue",
    chez_options=None,
):
    """Run a match against one opponent. Returns W/D/L dict."""
    opp_name = opponent["name"]
    opp_cmd = opponent["cmd"]
    opp_proto = opponent.get("proto", "uci")

    # "none"/"hce" sentinel: omit EvalFile so the engine falls back to HCE (no NNUE).
    hce = eval_file is None or str(eval_file).lower() in ("none", "hce", "")
    chez_args = [
        "-engine",
        "name=Chez",
        f"cmd={chez_binary}",
        f"option.Threads={threads}",
    ]
    if not hce:
        chez_args.append(f"option.EvalFile={eval_file}")
    for opt in (chez_options or []):
        chez_args.append(f"option.{opt}")
    chez_args.append("proto=uci")

    cmd = [
        fastchess,
        *chez_args,
        "-engine",
        f"name={opp_name}",
        f"cmd={opp_cmd}",
        f"proto={opp_proto}",
    ]

    for key, val in opponent.get("options", {}).items():
        cmd.append(f"option.{key}={val}")

    cmd += [
        "-each",
        f"tc={tc}",
        "restart=on",
        "timemargin=300",
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
        "-ratinginterval",
        "10",
        "-output",
        "format=cutechess",
    ]

    if openings:
        book = Path(openings)
        if book.exists():
            fmt = "pgn" if book.suffix == ".pgn" else "epd"
            cmd += ["-openings", f"file={book}", f"format={fmt}", "order=random"]

    if pgn_out:
        cmd += ["-pgnout", f"file={pgn_out}"]

    print(f"Running command {cmd}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("Could not open subprocess stdout.")

    output_lines = []

    for line in proc.stdout:
        output_lines.append(line)
        if line.startswith("Score of"):
            print(f"    {line.strip()}")
    proc.wait()
    output = "".join(output_lines)

    score_re = re.compile(
        r"Score of .+? vs .+?: (\d+) - (\d+) - (\d+)\s+\[([0-9.]+)\]\s+(\d+)"
    )
    wins = losses = draws = games = 0
    for line in output.splitlines():
        m = score_re.search(line)
        if m:
            wins = int(m.group(1))
            losses = int(m.group(2))
            draws = int(m.group(3))
            games = int(m.group(5))

    return {"wins": wins, "losses": losses, "draws": draws, "games": games}


def print_results(estimates, weighted_avg, combined_ci, config):
    tc = config.get("time_control", "?")
    threads = config.get("threads", "?")
    rating_list = config.get("rating_list", "CCRL")

    print()
    print("=" * 72)
    print("  Chez Gauntlet Results")
    print("=" * 72)
    print(f"  TC: {tc}  |  Threads: {threads}  |  Ratings: {rating_list}")
    print("-" * 72)
    print(
        f"  {'Opponent':<22} {'Rating':>6} {'Games':>6} {'Score':>7}"
        f" {'Elo Diff':>10} {'Est.':>7}"
    )
    print(f"  {'─' * 22} {'─' * 6} {'─' * 6} {'─' * 7} {'─' * 10} {'─' * 7}")

    for e in estimates:
        name = e["name"][:22]
        rating = e["rating"]
        games = e["games"]
        if games > 0:
            score_str = f"{e['score_pct']:.1f}%"
            if e["ci_95"] < float("inf"):
                diff_str = f"{e['elo_diff']:+.0f} \u00b1{e['ci_95']:.0f}"
            else:
                diff_str = f"{e['elo_diff']:+.0f}"
            est_str = f"{e['estimate']:.0f}"
        else:
            score_str = "\u2014"
            diff_str = "\u2014"
            est_str = "\u2014"
        print(
            f"  {name:<22} {rating:>6} {games:>6} {score_str:>7}"
            f" {diff_str:>10} {est_str:>7}"
        )

    print("-" * 72)
    if combined_ci < float("inf"):
        print(f"  Estimated rating: {weighted_avg:.0f} \u00b1 {combined_ci:.0f}")
    else:
        print(f"  Estimated rating: {weighted_avg:.0f} (insufficient data for CI)")
    print("=" * 72)


def save_results(estimates, weighted_avg, combined_ci, config, pgn_path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"gauntlet_{timestamp}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "config": {k: v for k, v in config.items() if k != "opponents"},
        "per_opponent": [
            {
                "name": e["name"],
                "rating": e["rating"],
                "games": e["games"],
                "wins": e["wins"],
                "draws": e["draws"],
                "losses": e["losses"],
                "score_pct": round(e.get("score_pct", 0), 1),
                "elo_diff": round(e.get("elo_diff", 0), 1),
                "ci_95": round(e["ci_95"], 1) if e["ci_95"] < float("inf") else None,
                "estimate": round(e.get("estimate", 0)),
            }
            for e in estimates
        ],
        "estimated_rating": round(weighted_avg),
        "ci_95": round(combined_ci, 1) if combined_ci < float("inf") else None,
    }

    if pgn_path and pgn_path.exists():
        data["pgn_file"] = str(pgn_path)

    json_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\n  Results saved: {json_path}")
    return json_path


TEMPLATE_CONFIG = {
    "_comment": "Opponent ratings: https://www.computerchess.org.uk/ccrl/",
    "time_control": "2+1",
    "rounds_per_opponent": 100,
    "concurrency": 4,
    "threads": 4,
    "rating_list": "CCRL Blitz",
    "opponents": [
        {
            "name": "Engine Name",
            "cmd": "/path/to/engine/binary",
            "rating": 2000,
            "proto": "uci",
            "options": {},
        },
    ],
}


def init_config(path):
    if path.exists():
        print(f"Config already exists: {path}", file=sys.stderr)
        print("Delete it first or use --config to specify a different path.")
        sys.exit(1)
    path.write_text(json.dumps(TEMPLATE_CONFIG, indent=2) + "\n")
    print(f"Template config written to: {path}")
    print()
    print("Edit it to add your opponent engines with their known ratings.")
    print(
        "Each opponent needs: name, cmd (path to binary), rating, proto (uci/xboard)."
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Estimate absolute Elo via gauntlet against rated engines"
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Config file path (default: gauntlet.json)",
    )
    p.add_argument(
        "--init-config",
        action="store_true",
        help="Generate template config file",
    )
    p.add_argument("--engine", default=None, help="Path to pre-built Chez binary")
    p.add_argument("--eval", default=None, help="EvalFile (.nnue) for the tested Chez engine")
    p.add_argument(
        "--chez-option",
        action="append",
        default=None,
        metavar="KEY=VAL",
        help="Extra UCI option for the tested Chez engine (repeatable), e.g. --chez-option NnueScale=2",
    )
    p.add_argument("--commit", default=None, help="Git ref to build Chez from")
    p.add_argument("--tc", default=None, help="Override time control")
    p.add_argument(
        "--rounds", type=int, default=None, help="Override rounds per opponent"
    )
    p.add_argument("--threads", type=int, default=None, help="Override engine threads")
    p.add_argument(
        "--concurrency", type=int, default=None, help="Override parallel games"
    )
    p.add_argument("--no-save", action="store_true", help="Don't save result JSON")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    if args.init_config:
        init_config(config_path)
        return

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        print("Run with --init-config to generate a template.", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    opponents = config.get("opponents", [])

    if not opponents:
        print("No opponents configured. Edit your config file.", file=sys.stderr)
        sys.exit(1)

    # Apply CLI overrides
    tc = args.tc or config.get("time_control", "2+1")
    rounds = args.rounds or config.get("rounds_per_opponent", 100)
    threads = args.threads or config.get("threads", 4)
    concurrency = args.concurrency or config.get("concurrency", 4)
    openings = config.get("opening_book")

    # Resolve relative opening book path
    if openings and not Path(openings).is_absolute():
        openings = str(SCRIPT_DIR / openings)

    fastchess = find_fastchess()

    print()
    print("Chez Gauntlet")
    print("=" * 72)

    # Build/locate Chez
    if args.engine:
        chez_binary = Path(args.engine).resolve()
        if not chez_binary.exists():
            print(f"Error: binary not found: {chez_binary}", file=sys.stderr)
            sys.exit(1)
        print(f"  Using engine: {chez_binary}")
    elif args.commit:
        chez_binary = build_from_commit(args.commit)
    else:
        chez_binary = build_current()

    # Summary
    n_opp = len(opponents)
    games_per_opp = rounds * 2
    total_games = games_per_opp * n_opp
    print(
        f"\n  {n_opp} opponents, {rounds} round pairs each"
        f" ({games_per_opp} games/opponent, {total_games} total)"
    )
    print(f"  TC: {tc}  |  Threads: {threads}  |  Concurrency: {concurrency}")
    if openings:
        print(f"  Openings: {openings}")

    # PGN output
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pgn_out = RESULTS_DIR / f"gauntlet_{timestamp}.pgn"

    # EvalFile for the tested engine: --eval overrides config "eval_file", else the run_match default
    eval_file = args.eval or config.get("eval_file") or "/home/paul/projects/chez/data/net.nnue"
    _hce = str(eval_file).lower() in ("none", "hce", "")
    print(f"  EvalFile: {'HCE (no NNUE)' if _hce else eval_file}")

    # Run matches sequentially against each opponent
    match_results = []
    for i, opp in enumerate(opponents, 1):
        print(f"\n  [{i}/{n_opp}] Chez vs {opp['name']} (rated {opp['rating']})...")
        result = run_match(
            fastchess,
            chez_binary,
            opp,
            tc,
            rounds,
            threads,
            concurrency,
            openings,
            pgn_out,
            eval_file=eval_file,
            chez_options=args.chez_option,
        )
        match_results.append({"name": opp["name"], "rating": opp["rating"], **result})

        ws, ls, ds = result["wins"], result["losses"], result["draws"]
        print(f"    Final: +{ws} -{ls} ={ds}")

    # Compute estimates
    weighted_avg, combined_ci, estimates = compute_estimates(match_results)

    # Display
    display_config = {
        "time_control": tc,
        "threads": threads,
        "rating_list": config.get("rating_list", "CCRL"),
    }
    print_results(estimates, weighted_avg, combined_ci, display_config)

    # Save
    if not args.no_save:
        save_config = {
            "time_control": tc,
            "rounds_per_opponent": rounds,
            "threads": threads,
            "concurrency": concurrency,
            "rating_list": config.get("rating_list", "CCRL"),
        }
        save_results(estimates, weighted_avg, combined_ci, save_config, pgn_out)
        if pgn_out.exists():
            print(f"  PGN saved: {pgn_out}")


if __name__ == "__main__":
    main()
