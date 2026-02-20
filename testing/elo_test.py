#!/usr/bin/env python3
"""Strength testing framework for Chez chess engine.

Uses cutechess-cli to run engine-vs-engine matches with two modes:
  quick  - fast regression check (100 games, 1+0.01)
  full   - SPRT-based Elo measurement (up to 10000 games, 5+0.05)
  custom - all defaults from quick, but no preset overrides

Supports self-play (current vs baseline commit) and play against external engines.
"""

import argparse
import atexit
import json
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
DEFAULT_BOOK = "/home/paul/projects/chez/testing/books/komodo.bin"
BOOK_URL = "https://www.sp-cc.de/files/8moves_v3.pgn"

# Worktrees created during this run, cleaned up on exit
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

# Also clean up on SIGINT/SIGTERM
_original_sigint = signal.getsignal(signal.SIGINT)
_original_sigterm = signal.getsignal(signal.SIGTERM)


def _signal_handler(signum, frame):
    cleanup_worktrees()
    if signum == signal.SIGINT and callable(_original_sigint):
        _original_sigint(signum, frame)
    elif signum == signal.SIGTERM and callable(_original_sigterm):
        _original_sigterm(signum, frame)
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# -- Utilities ----------------------------------------------------------------


def run(cmd, **kwargs):
    """Run a command, returning CompletedProcess. Raises on failure by default."""
    kwargs.setdefault("check", True)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    return subprocess.run(cmd, **kwargs)


def git(*args, **kwargs):
    """Run a git command in the project root."""
    return run(["git", *args], cwd=PROJECT_ROOT, **kwargs)


def resolve_commit(ref):
    """Resolve a git ref to a full commit hash, or None if it's not a valid ref."""
    try:
        result = git("rev-parse", "--verify", ref)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def short_hash(commit):
    """Return first 7 chars of a commit hash."""
    return commit[:7] if commit else None


def is_dirty():
    """Check if the working tree has uncommitted changes."""
    result = git("status", "--porcelain")
    return bool(result.stdout.strip())


def find_cutechess():
    """Find cutechess-cli on PATH."""
    path = shutil.which("cutechess-cli")
    if not path:
        print("Error: cutechess-cli not found on PATH.", file=sys.stderr)
        print(
            "Install it from: https://github.com/cutechess/cutechess", file=sys.stderr
        )
        sys.exit(1)
    return path


# -- Build management ---------------------------------------------------------


def build_from_worktree(commit, name):
    """Build the engine from a git commit using a temporary worktree.

    Returns the path to the built binary.
    """
    build_dir = BUILDS_DIR / name
    binary = build_dir / "uci"

    if binary.exists():
        print(f"  Using cached build: {binary}")
        return binary

    build_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary worktree
    wt_dir = BUILDS_DIR / f".worktree-{name}"
    if wt_dir.exists():
        git("worktree", "remove", "--force", str(wt_dir))

    print(f"  Checking out {short_hash(commit)} into worktree...")
    git("worktree", "add", "--detach", str(wt_dir), commit)
    _active_worktrees.append(wt_dir)

    # Update submodules in the worktree
    try:
        run(
            ["git", "submodule", "update", "--init"],
            cwd=wt_dir,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        pass  # may not have submodules

    # Build
    print(f"  Building {name}...")
    try:
        run(["zig", "build"], cwd=wt_dir, capture_output=False)
    except subprocess.CalledProcessError:
        print(
            f"\nError: zig build failed for {name} ({short_hash(commit)})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Copy binary
    built = wt_dir / "zig-out" / "bin" / "uci"
    if not built.exists():
        print(f"\nError: binary not found at {built}", file=sys.stderr)
        sys.exit(1)
    shutil.copy2(built, binary)

    # Clean up worktree
    git("worktree", "remove", "--force", str(wt_dir))
    _active_worktrees.remove(wt_dir)

    print(f"  Built: {binary}")
    return binary


def build_current(name="current"):
    """Build the engine from the current working tree.

    Returns the path to the built binary.
    """
    build_dir = BUILDS_DIR / name
    binary = build_dir / "uci"
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Building {name} from working tree...")
    try:
        run(["zig", "build"], cwd=PROJECT_ROOT, capture_output=False)
    except subprocess.CalledProcessError:
        print("\nError: zig build failed for current working tree", file=sys.stderr)
        sys.exit(1)

    built = PROJECT_ROOT / "zig-out" / "bin" / "uci"
    if not built.exists():
        print(f"\nError: binary not found at {built}", file=sys.stderr)
        sys.exit(1)
    shutil.copy2(built, binary)

    print(f"  Built: {binary}")
    return binary


def prepare_engine(spec, name, is_current=False):
    """Prepare an engine binary from a commit ref or file path.

    Returns (binary_path, display_name, commit_hash_or_None).
    """
    # Check if spec is a file path to an existing binary
    spec_path = Path(spec)
    if spec_path.is_file():
        print(f"  Using external engine: {spec_path}")
        return spec_path.resolve(), name, None

    if is_current and spec == "worktree":
        dirty = is_dirty()
        head = resolve_commit("HEAD")
        display = f"{name} (HEAD{'*' if dirty else ''})"
        binary = build_current(name)
        return binary, display, head

    # It's a git ref
    commit = resolve_commit(spec)
    if not commit:
        print(f"Error: '{spec}' is not a valid git ref or file path.", file=sys.stderr)
        sys.exit(1)

    display = f"{name} ({short_hash(commit)})"
    binary = build_from_worktree(commit, f"{name}-{short_hash(commit)}")
    return binary, display, commit


# -- cutechess-cli invocation -------------------------------------------------


def build_cutechess_cmd(args, engine_current, engine_baseline, pgn_out):
    """Build the cutechess-cli command line."""
    cutechess = find_cutechess()
    tc = args.tc
    rounds = args.rounds

    cmd = [
        cutechess,
        "-engine",
        f"name={args.current_name}",
        f"cmd={engine_current}",
        f"option.Threads={args.threads}",
        "proto=uci",
    ]

    book0 = Path(args.book0)
    if book0.exists():
        cmd += [f"option.BookFile={book0}"]

    cmd += [
        "-engine",
        f"name={args.baseline_name}",
        f"cmd={engine_baseline}",
        f"option.Threads={args.threads}",
        "proto=uci",
    ]

    book1 = Path(args.book1)
    if book1.exists():
        cmd += [f"option.BookFile={book1}"]

    # Time control or fixed depth
    if args.depth:
        cmd += [
            "-each",
            "tc=inf",
            f"depth={args.depth}",
            "restart=on",
            "timemargin=300",
        ]
    else:
        cmd += ["-each", f"tc={tc}", "restart=on", "timemargin=300"]

    cmd += [
        "-rounds",
        str(rounds),
        "-repeat",
        "2",
        "-recover",
        "-concurrency",
        str(args.concurrency),
        "-draw",
        "movenumber=40",
        "movecount=10",
        "score=5",
        "-resign",
        "movecount=5",
        "score=1000",
        "-ratinginterval",
        "10",
    ]

    # PGN output
    if pgn_out:
        cmd += ["-pgnout", str(pgn_out)]

    # SPRT
    if args.mode == "full" or (args.elo0 is not None and args.elo1 is not None):
        elo0 = args.elo0 if args.elo0 is not None else 0
        elo1 = args.elo1 if args.elo1 is not None else 5
        cmd += ["-sprt", f"elo0={elo0}", f"elo1={elo1}", "alpha=0.05", "beta=0.05"]

    print(f"cutechess-cli Command:\n {cmd}")
    return cmd


# -- Result parsing -----------------------------------------------------------


def parse_results(output):
    """Parse cutechess-cli output for results."""
    results = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "games": 0,
        "score_pct": None,
        "elo": None,
        "elo_error": None,
        "los": None,
        "sprt_result": None,
    }

    # Score line: "Score of dev vs baseline: W - L - D  [pct] N"
    score_re = re.compile(
        r"Score of .+? vs .+?: (\d+) - (\d+) - (\d+)\s+\[([0-9.]+)\]\s+(\d+)"
    )

    # Elo line: "Elo difference: +X.X +/- Y.Y"
    # or: "Elo difference: X.X +/- Y.Y, LOS: Z.Z %"
    elo_re = re.compile(r"Elo difference:\s+([+-]?[0-9.]+)\s*\+/-\s*([0-9.]+)")
    los_re = re.compile(r"LOS:\s+([0-9.]+)\s*%")

    # SPRT: "SPRT: llr X.XX (Y.YY%), lbound -Z.ZZ, ubound Z.ZZ - H1 was accepted"
    sprt_re = re.compile(r"SPRT:.*?(H[01] was (?:accepted|rejected))")

    for line in output.splitlines():
        m = score_re.search(line)
        if m:
            results["wins"] = int(m.group(1))
            results["losses"] = int(m.group(2))
            results["draws"] = int(m.group(3))
            results["score_pct"] = float(m.group(4)) * 100
            results["games"] = int(m.group(5))

        m = elo_re.search(line)
        if m:
            results["elo"] = float(m.group(1))
            results["elo_error"] = float(m.group(2))

        m = los_re.search(line)
        if m:
            results["los"] = float(m.group(1))

        m = sprt_re.search(line)
        if m:
            results["sprt_result"] = m.group(1)

    return results


# -- Output -------------------------------------------------------------------


def print_results(results, args, current_display, baseline_display):
    """Pretty-print the results summary."""
    wins, losses, draws = results["wins"], results["losses"], results["draws"]
    games = results["games"]
    pct = results["score_pct"]

    tc_str = f"depth {args.depth}" if args.depth else args.tc

    print()
    print("=" * 42)
    print("  Chez Strength Test Results")
    print("=" * 42)
    print(f"  Mode:      {args.mode} ({args.rounds} game pairs)")
    print(f"  Current:   {current_display}")
    print(f"  Baseline:  {baseline_display}")
    print(f"  TC:        {tc_str}  |  Threads: {args.threads}")
    print("-" * 42)
    print(f"  Games:     {games}")

    if games > 0:
        print(f"  Score:     +{wins} -{losses} ={draws} ({pct:.1f}%)")

        if results["elo"] is not None:
            print(f"  Elo:       {results['elo']:+.1f} +/- {results['elo_error']:.1f}")

        if results["los"] is not None:
            print(f"  LOS:       {results['los']:.1f}%")

    print("-" * 42)

    if results["sprt_result"]:
        print(f"  SPRT:      {results['sprt_result']}")
    elif games > 0 and results["elo"] is not None:
        if results["elo_error"] and abs(results["elo"]) < results["elo_error"]:
            print("  Verdict:   No significant difference")
        elif results["elo"] > 0:
            print("  Verdict:   Likely improvement")
        else:
            print("  Verdict:   Likely regression")
    else:
        print("  Verdict:   Inconclusive")

    print("=" * 42)


def save_results(
    results,
    args,
    current_display,
    baseline_display,
    current_commit,
    baseline_commit,
    pgn_path,
):
    """Save results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"{timestamp}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "current": {"name": current_display, "commit": current_commit},
        "baseline": {"name": baseline_display, "commit": baseline_commit},
        "config": {
            "tc": args.tc,
            "depth": args.depth,
            "threads": args.threads,
            "rounds": args.rounds,
            "concurrency": args.concurrency,
        },
        "results": results,
    }

    if pgn_path:
        data["pgn_file"] = str(pgn_path)

    json_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\n  Results saved: {json_path}")
    return json_path


# -- Main ---------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Chez engine strength testing via cutechess-cli"
    )
    p.add_argument(
        "--mode",
        choices=["quick", "full", "custom"],
        default="quick",
        help="Test mode (default: quick)",
    )
    p.add_argument(
        "--baseline", default=None, help="Git ref or path to baseline engine"
    )
    p.add_argument(
        "--baseline-name", default="baseline", help="Display name for baseline"
    )
    p.add_argument("--current", default=None, help="Git ref or path to current engine")
    p.add_argument(
        "--current-name", default="dev", help="Display name for current engine"
    )
    p.add_argument("--rounds", type=int, default=None, help="Number of game pairs")
    p.add_argument("--tc", default=None, help="Time control (e.g. 1+0.01)")
    p.add_argument("--threads", type=int, default=4, help="Engine threads (default: 4)")
    p.add_argument(
        "--concurrency", type=int, default=4, help="Parallel games (default: 4)"
    )
    p.add_argument(
        "--depth", type=int, default=None, help="Fixed search depth (instead of TC)"
    )
    p.add_argument("--book0", default=str(DEFAULT_BOOK), help="Opening book path")
    p.add_argument("--book1", default=str(DEFAULT_BOOK), help="Opening book path")
    p.add_argument("--elo0", type=float, default=None, help="SPRT lower bound")
    p.add_argument("--elo1", type=float, default=None, help="SPRT upper bound")
    p.add_argument("--pgn-out", default=None, help="PGN output path")
    p.add_argument("--no-save", action="store_true", help="Don't save result JSON")
    p.add_argument("--verbose", action="store_true", help="Stream cutechess output")
    return p.parse_args()


def apply_mode_defaults(args):
    """Apply mode-specific defaults for unset parameters."""
    if args.mode == "quick":
        if args.rounds is None:
            args.rounds = 50
        if args.tc is None:
            args.tc = "1+0.01"
    elif args.mode == "full":
        if args.rounds is None:
            args.rounds = 5000
        if args.tc is None:
            args.tc = "5+0.05"
        if args.elo0 is None:
            args.elo0 = 0
        if args.elo1 is None:
            args.elo1 = 5
    elif args.mode == "custom":
        if args.rounds is None:
            args.rounds = 50
        if args.tc is None:
            args.tc = "1+0.01"


def resolve_engine_specs(args):
    """Determine what to build for current and baseline."""
    dirty = is_dirty()

    # Current engine
    if args.current is None:
        args.current = "worktree"
    # Baseline engine
    if args.baseline is None:
        if dirty:
            args.baseline = "HEAD"
        else:
            args.baseline = "HEAD~1"


def main():
    args = parse_args()
    apply_mode_defaults(args)
    resolve_engine_specs(args)

    print()
    print("Chez Strength Test")
    print("=" * 42)

    # Check for cutechess-cli
    find_cutechess()

    # Build engines
    print("\nPreparing engines...")

    current_binary, current_display, current_commit = prepare_engine(
        args.current, args.current_name, is_current=True
    )
    baseline_binary, baseline_display, baseline_commit = prepare_engine(
        args.baseline, args.baseline_name
    )

    # PGN output path
    pgn_out = None
    if args.pgn_out:
        pgn_out = Path(args.pgn_out)
    elif not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pgn_out = RESULTS_DIR / f"{timestamp}.pgn"

    # Build and run cutechess command
    cmd = build_cutechess_cmd(args, current_binary, baseline_binary, pgn_out)

    tc_str = f"depth {args.depth}" if args.depth else args.tc
    print(f"\nRunning {args.rounds} game pairs ({args.rounds * 2} games)")
    print(
        f"  TC: {tc_str}  |  Threads: {args.threads}  |  Concurrency: {args.concurrency}"
    )

    if args.verbose:
        print(f"\n  Command: {' '.join(str(c) for c in cmd)}\n")

    try:
        if args.verbose:
            # Stream output in real time and capture it
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output_lines = []
            for line in proc.stdout:
                print(line, end="")
                output_lines.append(line)
            proc.wait()
            output = "".join(output_lines)
            if proc.returncode != 0:
                print(
                    f"\ncutechess-cli exited with code {proc.returncode}",
                    file=sys.stderr,
                )
        else:
            # Show periodic progress updates
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output_lines = []
            for line in proc.stdout:
                output_lines.append(line)
                # Print rating interval updates
                if line.startswith("Score of"):
                    print(f"  {line.strip()}")
                elif "SPRT" in line:
                    print(f"  {line.strip()}")
            proc.wait()
            output = "".join(output_lines)
    except FileNotFoundError:
        print("Error: cutechess-cli not found", file=sys.stderr)
        sys.exit(1)

    # Parse and display results
    results = parse_results(output)
    print_results(results, args, current_display, baseline_display)

    # Save results
    if not args.no_save:
        save_results(
            results,
            args,
            current_display,
            baseline_display,
            current_commit,
            baseline_commit,
            pgn_out,
        )

    if pgn_out and pgn_out.exists():
        print(f"  PGN saved:     {pgn_out}")


if __name__ == "__main__":
    main()
