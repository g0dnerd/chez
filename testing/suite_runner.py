#!/usr/bin/env python3
"""EPD test suite runner for chess engine diagnostics.

Runs EPD test suites against a UCI engine and produces categorized reports.
Supports binary scoring (WAC, ERET, Bratko-Kopec) and weighted scoring (STS).

Usage:
    # Single EPD file at fixed depth
    uv run python testing/suite_runner.py testing/suites/wac.epd --depth 12

    # Directory of EPD files (e.g., STS themes)
    uv run python testing/suite_runner.py testing/suites/sts/ --depth 10

    # Fixed time mode (milliseconds)
    uv run python testing/suite_runner.py testing/suites/eret.epd --movetime 5000

    # Save JSON report
    uv run python testing/suite_runner.py testing/suites/wac.epd --depth 12 \
        --json testing/results/wac_baseline.json
"""

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import chess

DEFAULT_ENGINE = "zig-out/bin/uci"
DEFAULT_DEPTH = 16
DEFAULT_THREADS = 1


class UCIEngine:
    """Manages a UCI chess engine subprocess."""

    def __init__(self, path: str, threads: int = 1, evalfile: str | None = None):
        self.path = str(Path(path).resolve())
        self.threads = threads
        self.evalfile = str(Path(evalfile).resolve()) if evalfile else None
        self.name = Path(path).name

    def start(self):
        p = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        if p is None:
            raise RuntimeError("Failed to spawn subprocess")

        self.process = p

        self._send("uci")
        for line in self._read_until("uciok"):
            if line.startswith("id name "):
                self.name = line[len("id name ") :]
        self._send(f"setoption name Threads value {self.threads}")
        self._send("setoption name OwnBook value false")
        if self.evalfile:
            self._send(f"setoption name EvalFile value {self.evalfile}")
        self._send("isready")
        lines = self._read_until("readyok")
        if self.evalfile and any("failed to load EvalFile" in l for l in lines):
            raise RuntimeError(
                f"Engine failed to load EvalFile '{self.evalfile}' "
                "(it fell back to HCE)"
            )

    def _send(self, cmd: str):
        assert self.process is not None
        assert self.process.stdin is not None

        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def _readline(self) -> str:
        assert self.process is not None
        assert self.process.stdout is not None

        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Engine process terminated unexpectedly")
        return line.strip()

    def _read_until(self, token: str) -> list[str]:
        lines = []
        while True:
            line = self._readline()
            lines.append(line)
            if line.startswith(token):
                return lines

    def search(
        self, fen: str, *, depth: int | None = None, movetime: int | None = None
    ) -> tuple[str, int, dict]:
        """Search a position. Returns (bestmove_uci, score_cp, info)."""
        self._send(f"position fen {fen}")
        self._send("isready")
        self._read_until("readyok")

        if depth is not None:
            self._send(f"go depth {depth}")
        elif movetime is not None:
            self._send(f"go movetime {movetime}")
        else:
            raise ValueError("Must specify depth or movetime")

        score_cp = 0
        info = {}
        while True:
            line = self._readline()
            if line.startswith("info") and "score" in line:
                parts = line.split()
                try:
                    if "cp" in parts:
                        score_cp = int(parts[parts.index("cp") + 1])
                    elif "mate" in parts:
                        mate_in = int(parts[parts.index("mate") + 1])
                        score_cp = 100000 * (1 if mate_in > 0 else -1)
                except (ValueError, IndexError):
                    pass
                try:
                    if "depth" in parts:
                        info["depth"] = int(parts[parts.index("depth") + 1])
                    if "nodes" in parts:
                        info["nodes"] = int(parts[parts.index("nodes") + 1])
                    if "time" in parts:
                        info["time_ms"] = int(parts[parts.index("time") + 1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("bestmove"):
                bestmove = line.split()[1]
                info["score_cp"] = score_cp
                return bestmove, score_cp, info

    def quit(self):
        if self.process and self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
                self.process.wait()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.quit()


@dataclass
class EPDPosition:
    fen: str  # Full 6-field FEN
    best_moves: list[str]  # UCI notation
    avoid_moves: list[str]  # UCI notation
    position_id: str
    category: str
    move_scores: dict[str, int]  # UCI move -> weighted score (STS-style)
    max_score: int  # Max achievable score for this position
    raw_line: str  # Original EPD line


def extract_category(position_id: str, filename: str = "") -> str:
    """Extract category/theme from EPD id or filename."""
    if position_id:
        # STS format: "STS(v14.0) Theme Name.NNN"
        m = re.match(r"STS\([^)]*\)\s+(.*?)\.\d+$", position_id)
        if m:
            return m.group(1).strip()
        # "WAC.001" or "BK.01" -> "WAC" or "BK"
        m = re.match(r"([A-Za-z_-]+)[\s.]\d+", position_id)
        if m:
            return m.group(1).strip()
        # "ERET 001 - Description" -> "ERET"
        m = re.match(r"([A-Za-z_-]+)\s+\d+", position_id)
        if m:
            return m.group(1).strip()
    if filename:
        return Path(filename).stem
    return "Unknown"


def parse_sts_scores(board: chess.Board, c0: str) -> dict[str, int]:
    """Parse STS c0 scoring string like 'd4=10, Nc3=5, a3=3'."""
    scores = {}
    for pair in c0.split(","):
        pair = pair.strip().strip('"')
        if "=" not in pair:
            continue
        # Use rsplit to handle promotion notation (e.g. "e8=Q=10")
        san, pts = pair.rsplit("=", 1)
        san = san.strip()
        pts = pts.strip().rstrip(";").strip('"')
        try:
            move = board.parse_san(san)
            scores[move.uci()] = int(pts)
        except (
            chess.IllegalMoveError,
            chess.InvalidMoveError,
            chess.AmbiguousMoveError,
            ValueError,
        ):
            pass
    return scores


def parse_epd_file(
    path: Path, category_override: str | None = None
) -> list[EPDPosition]:
    """Parse an EPD file into a list of EPDPosition objects."""
    positions = []
    filename = path.name

    with open(path) as f:
        for line_num, raw_line in enumerate(f, 1):
            raw_line = raw_line.strip()
            if not raw_line or raw_line.startswith("#"):
                continue

            try:
                board, ops = chess.Board.from_epd(raw_line)
            except Exception as e:
                print(
                    f"  Warning: skipping {path.name}:{line_num}: {e}",
                    file=sys.stderr,
                )
                continue

            fen = board.fen()
            position_id = ops.get("id", f"{filename}:{line_num}")

            best_moves = [m.uci() for m in ops.get("bm", [])]
            avoid_moves = [m.uci() for m in ops.get("am", [])]

            c0 = ops.get("c0", "")
            move_scores = parse_sts_scores(board, c0) if c0 else {}
            max_score = (
                max(move_scores.values())
                if move_scores
                else (1 if best_moves or avoid_moves else 0)
            )

            category = category_override or extract_category(position_id, filename)

            positions.append(
                EPDPosition(
                    fen=fen,
                    best_moves=best_moves,
                    avoid_moves=avoid_moves,
                    position_id=position_id,
                    category=category,
                    move_scores=move_scores,
                    max_score=max_score,
                    raw_line=raw_line,
                )
            )

    return positions


def load_positions(
    path: Path, category_override: str | None = None
) -> list[EPDPosition]:
    """Load positions from a file or directory of EPD files."""
    if path.is_dir():
        positions = []
        for epd_file in sorted(path.glob("*.epd")):
            positions.extend(parse_epd_file(epd_file, category_override))
        if not positions:
            print(f"Error: no .epd files found in {path}", file=sys.stderr)
            sys.exit(1)
        return positions
    elif path.is_file():
        return parse_epd_file(path, category_override)
    else:
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)


# -- Suite Runner --------------------------------------------------------------


@dataclass
class PositionResult:
    position_id: str
    fen: str
    category: str
    expected: list[str]  # Best moves (UCI)
    engine_move: str  # Engine's choice (UCI)
    correct: bool
    points: int
    max_points: int
    score_cp: int
    search_info: dict


def evaluate_result(
    pos: EPDPosition, engine_move: str, score_cp: int, info: dict
) -> PositionResult:
    """Score the engine's move against the EPD solution."""
    if pos.move_scores:
        points = pos.move_scores.get(engine_move, 0)
        correct = engine_move in pos.best_moves if pos.best_moves else points > 0
    elif pos.best_moves:
        correct = engine_move in pos.best_moves
        points = 1 if correct else 0
    elif pos.avoid_moves:
        correct = engine_move not in pos.avoid_moves
        points = 1 if correct else 0
    else:
        correct = False
        points = 0

    return PositionResult(
        position_id=pos.position_id,
        fen=pos.fen,
        category=pos.category,
        expected=pos.best_moves or [f"not {m}" for m in pos.avoid_moves],
        engine_move=engine_move,
        correct=correct,
        points=points,
        max_points=pos.max_score,
        score_cp=score_cp,
        search_info=info,
    )


def run_suite(
    engine: UCIEngine,
    positions: list[EPDPosition],
    *,
    depth: int | None = None,
    movetime: int | None = None,
    verbose: bool = False,
) -> list[PositionResult]:
    """Run all positions through the engine and collect results."""
    results = []
    total = len(positions)
    start_time = time.time()

    for i, pos in enumerate(positions):
        if not verbose:
            pct = (i + 1) / total * 100
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"\r  [{i + 1}/{total}] {pct:.0f}% "
                f"({rate:.1f} pos/s, ETA {eta:.0f}s)  ",
                end="",
                flush=True,
            )

        bestmove, score_cp, info = engine.search(
            pos.fen, depth=depth, movetime=movetime
        )

        result = evaluate_result(pos, bestmove, score_cp, info)
        results.append(result)

        if verbose:
            status = "OK" if result.correct else "FAIL"
            pts = (
                f" [{result.points}/{result.max_points}]"
                if result.max_points > 1
                else ""
            )
            exp = ", ".join(result.expected[:3])
            print(
                f"  {status:4s} {pos.position_id}: "
                f"got {bestmove}, expected {exp} "
                f"({score_cp}cp){pts}"
            )

    if not verbose:
        print()

    return results


# -- Reporting -----------------------------------------------------------------


def generate_report(
    results: list[PositionResult],
    engine_name: str,
    suite_path: str,
    mode: str,
    budget: int,
    threads: int,
    elapsed_s: float,
) -> dict:
    """Generate a structured report from suite results."""
    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {
                "total": 0,
                "correct": 0,
                "points": 0,
                "max_points": 0,
            }
        categories[cat]["total"] += 1
        categories[cat]["correct"] += int(r.correct)
        categories[cat]["points"] += r.points
        categories[cat]["max_points"] += r.max_points

    for cat_data in categories.values():
        cat_data["pass_rate"] = round(
            cat_data["correct"] / cat_data["total"] if cat_data["total"] else 0,
            4,
        )
        cat_data["score_pct"] = round(
            cat_data["points"] / cat_data["max_points"]
            if cat_data["max_points"]
            else 0,
            4,
        )

    total_positions = len(results)
    total_correct = sum(1 for r in results if r.correct)
    total_points = sum(r.points for r in results)
    total_max = sum(r.max_points for r in results)

    failures = [
        {
            "id": r.position_id,
            "fen": r.fen,
            "category": r.category,
            "expected": r.expected,
            "got": r.engine_move,
            "score_cp": r.score_cp,
            "points": r.points,
            "max_points": r.max_points,
        }
        for r in results
        if not r.correct
    ]

    return {
        "engine": engine_name,
        "suite": suite_path,
        "mode": mode,
        "budget": budget,
        "threads": threads,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_s, 1),
        "summary": {
            "total": total_positions,
            "correct": total_correct,
            "pass_rate": round(
                total_correct / total_positions if total_positions else 0, 4
            ),
            "points": total_points,
            "max_points": total_max,
            "score_pct": round(total_points / total_max if total_max else 0, 4),
        },
        # Sort categories by score_pct ascending (worst first)
        "categories": dict(sorted(categories.items(), key=lambda x: x[1]["score_pct"])),
        "failures": failures,
    }


def print_summary(report: dict):
    """Print a formatted summary table to stdout."""
    s = report["summary"]
    cats = report["categories"]
    has_weighted = s["max_points"] != s["total"]

    print()
    print(
        f"Engine: {report['engine']}  |  "
        f"Mode: {report['mode']} {report['budget']}  |  "
        f"Threads: {report['threads']}  |  "
        f"Time: {report['elapsed_seconds']}s"
    )
    print(f"Suite: {report['suite']}")
    print()

    if has_weighted:
        hdr = f"{'Category':<40} {'Score':>6} {'Max':>6} {'%':>7}  {'Correct':>12}"
        print(hdr)
        print("-" * len(hdr))
        for cat, d in cats.items():
            pct = d["score_pct"] * 100
            print(
                f"{cat:<40} {d['points']:>6} {d['max_points']:>6} "
                f"{pct:>6.1f}%  {d['correct']:>5}/{d['total']:<5}"
            )
        print("-" * len(hdr))
        pct = s["score_pct"] * 100
        print(
            f"{'TOTAL':<40} {s['points']:>6} {s['max_points']:>6} "
            f"{pct:>6.1f}%  {s['correct']:>5}/{s['total']:<5}"
        )
    else:
        hdr = f"{'Category':<40} {'Correct':>8} {'Total':>6} {'%':>7}"
        print(hdr)
        print("-" * len(hdr))
        for cat, d in cats.items():
            pct = d["pass_rate"] * 100
            print(f"{cat:<40} {d['correct']:>8} {d['total']:>6} {pct:>6.1f}%")
        print("-" * len(hdr))
        pct = s["pass_rate"] * 100
        print(f"{'TOTAL':<40} {s['correct']:>8} {s['total']:>6} {pct:>6.1f}%")

    print()

    failures = report["failures"]
    if failures and len(failures) <= 50:
        print(f"Failures ({len(failures)}):")
        for f in failures[:20]:
            exp = ", ".join(f["expected"][:3])
            print(f"  {f['id']}: got {f['got']}, expected {exp} ({f['score_cp']}cp)")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        print()
    elif failures:
        print(
            f"Failures: {len(failures)}/{s['total']} (use --json to save full details)"
        )
        print()


# -- CLI -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run EPD test suites against a UCI chess engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s testing/suites/wac.epd --depth 12
  %(prog)s testing/suites/sts/ --depth 10 --json results.json
  %(prog)s testing/suites/eret.epd --movetime 5000 --verbose
        """,
    )
    parser.add_argument(
        "suite",
        type=Path,
        help="Path to EPD file or directory of EPD files",
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        help=f"Path to UCI engine (default: {DEFAULT_ENGINE})",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help=f"Search depth (default: {DEFAULT_DEPTH})",
    )
    parser.add_argument(
        "--movetime",
        type=int,
        default=None,
        help="Search time in milliseconds (alternative to --depth)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Engine threads (default: {DEFAULT_THREADS})",
    )
    parser.add_argument(
        "--evalfile",
        default=None,
        help="NNUE net to load via UCI EvalFile (default: engine HCE)",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Override category name for all positions",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        dest="json_out",
        help="Save JSON report to file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-position results",
    )

    args = parser.parse_args()

    if args.depth is None and args.movetime is None:
        args.depth = DEFAULT_DEPTH
    if args.depth and args.movetime:
        parser.error("Cannot specify both --depth and --movetime")

    mode = "depth" if args.depth else "movetime"
    budget = args.depth or args.movetime

    # Verify engine exists
    engine_path = Path(args.engine)
    if not engine_path.exists():
        print(f"Error: engine not found at {engine_path}", file=sys.stderr)
        print("  Run 'zig build' first to build the UCI engine.", file=sys.stderr)
        sys.exit(1)

    # Load positions
    print(f"Loading positions from {args.suite}...")
    positions = load_positions(args.suite, args.category)
    categories = set(p.category for p in positions)
    print(
        f"  {len(positions)} positions loaded "
        f"({len(categories)} categor{'y' if len(categories) == 1 else 'ies'})"
    )

    # Run suite
    print(f"Running at {mode} {budget}, {args.threads} thread(s)...")
    start_time = time.time()

    with UCIEngine(
        args.engine, threads=args.threads, evalfile=args.evalfile
    ) as engine:
        engine_name = engine.name
        results = run_suite(
            engine,
            positions,
            depth=args.depth,
            movetime=args.movetime,
            verbose=args.verbose,
        )

    elapsed = time.time() - start_time

    # Generate and display report
    report = generate_report(
        results,
        engine_name=engine_name,
        suite_path=str(args.suite),
        mode=mode,
        budget=budget,
        threads=args.threads,
        elapsed_s=elapsed,
    )

    print_summary(report)

    # Save JSON report
    if args.json_out is not None:
        # args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.json_out}")


if __name__ == "__main__":
    main()
