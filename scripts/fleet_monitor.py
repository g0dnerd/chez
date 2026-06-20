#!/usr/bin/env python3
"""Live aggregate dashboard for fleet selfplay.

Invoked by `fleet_selfplay.sh monitor`. Each host is passed as
`--node "host|key|cost|port"` (key, cost, and port all optional).

Rates are CUMULATIVE AVERAGES over the whole time this monitor has been alive --
each host's rate is (positions_now - positions_at_first_sight) / elapsed. This is
stable, unlike a per-interval delta: the remote progress counter only updates
every 100 games, which is coarser than the poll interval, so naive deltas read 0
on most polls. The average just converges to the true steady-state rate.

If any host carries a cost (interpreted as $/HOUR), the dashboard sums it and adds
a $/hour total, a $/1M-positions figure, and a projected cost on each ETA.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# POSIX/bash payload, fed to `bash -s` over ssh so it runs regardless of the
# remote login shell (some boxes default to fish, which can't parse `VAR=$(...)`).
REMOTE = r"""
f=$(ls -t /tmp/selfplay_shards.*/shard_0.log 2>/dev/null | head -1)
pos=0; games=0
if [ -n "$f" ]; then
  pos=$(tr '\r' '\n' < "$f" | grep -oE 'Positions: [0-9]+' | tail -1 | grep -oE '[0-9]+$')
  games=$(tr '\r' '\n' < "$f" | grep -oE 'Games: [0-9]+' | tail -1 | grep -oE '[0-9]+$')
fi
run=$(pgrep -c -f '[s]elfplay --num_games' 2>/dev/null || echo 0)
echo "${pos:-0} ${games:-0} ${run:-0}"
"""


class Node:
    def __init__(self, host, key, cost, port=""):
        self.host = host
        self.key = os.path.expanduser(key) if key else ""
        self.cost = cost  # float $/hr, or None
        self.port = port  # ssh port, "" => default 22
        # A loopback host is this machine: sample it directly, no ssh.
        self.local = host.split("@")[-1] in ("localhost", "127.0.0.1", "::1")
        self.pos0 = None  # positions at first sighting (rate baseline)
        self.t0 = None
        self.pos = 0
        self.games = 0
        self.run = -1  # -1 unreachable, 0 down, >=1 up


def sample(node):
    if node.local:
        cmd = ["bash", "-s"]
    else:
        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
        if node.key:
            cmd += ["-i", node.key]
        if node.port:
            cmd += ["-p", node.port]
        cmd += [node.host, "bash", "-s"]
    try:
        r = subprocess.run(
            cmd, input=REMOTE, capture_output=True, text=True, timeout=25
        )
        parts = r.stdout.split()
        if len(parts) < 3:
            node.run = -1
            return
        node.pos, node.games, node.run = int(parts[0]), int(parts[1]), int(parts[2])
        if node.run >= 1:
            now = time.time()
            # Establish baseline on first sight; reset if the counter went
            # backwards (a box that was restarted).
            if node.pos0 is None or node.pos < node.pos0:
                node.pos0, node.t0 = node.pos, now
    except Exception:
        node.run = -1


def rate_of(node, now):
    if node.run >= 1 and node.t0 is not None:
        dt = now - node.t0
        if dt >= 1:
            return (node.pos - node.pos0) / dt
    return None


def fmt_secs(s):
    s = int(s)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", action="append", default=[], help="host|key|cost|port")
    ap.add_argument("--interval", type=float, default=10)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    nodes = []
    for spec in args.node:
        host, key, cost, port = (spec.split("|") + ["", "", ""])[:4]
        nodes.append(
            Node(host, key, float(cost) if cost.strip() else None, port.strip())
        )
    if not nodes:
        print("no hosts given", file=sys.stderr)
        sys.exit(1)

    have_cost = any(n.cost is not None for n in nodes)
    start = time.time()
    hide_cursor = not args.once
    if hide_cursor:
        sys.stdout.write("\033[?25l")

    try:
        while True:
            with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
                list(ex.map(sample, nodes))
            now = time.time()
            elapsed = now - start

            out = []
            if not args.once:
                out.append("\033[H\033[J")  # cursor home + clear
            tail = ", Ctrl-C to quit)" if not args.once else ")"
            out.append(
                f"Fleet selfplay — {time.strftime('%H:%M:%S')}   "
                f"({len(nodes)} hosts, refresh {args.interval:g}s, "
                f"elapsed {fmt_secs(elapsed)}{tail}"
            )

            header = (
                f"{'HOST':<26} {'STATUS':<8} {'POSITIONS':>14} "
                f"{'POS/S':>8} {'GAMES':>12}"
            )
            if have_cost:
                header += f" {'$/HR':>7} {'$/1M':>8}"
            out.append(header)
            out.append("-" * len(header))

            total_pos = total_rate = total_cost = 0.0
            total_games = 0
            alive = 0
            rated = False
            for n in nodes:
                r = rate_of(n, now)
                status = "UNREACH" if n.run < 0 else ("up" if n.run >= 1 else "DOWN")
                rate_disp = f"{r:.1f}" if r is not None else "-"
                row = (
                    f"{n.host:<26} {status:<8} {n.pos:>14,} "
                    f"{rate_disp:>8} {n.games:>12,}"
                )
                if have_cost:
                    cost_s = f"{n.cost:.3f}" if n.cost is not None else "-"
                    if n.run >= 1 and n.cost and r is not None and r > 0:
                        per_m_s = f"{n.cost / (r * 3600 / 1e6):.2f}"
                    else:
                        per_m_s = "-"
                    row += f" {cost_s:>7} {per_m_s:>8}"
                out.append(row)
                total_pos += n.pos
                total_games += n.games
                if n.run >= 1:
                    alive += 1
                    if n.cost:
                        total_cost += n.cost
                    if r is not None:
                        total_rate += r
                        rated = True
            out.append("-" * len(header))

            status_total = f"{alive}/{len(nodes)} up"
            rate_s = f"{total_rate:.1f}" if rated else "…"
            total_row = (
                f"{'TOTAL':<26} {status_total:<8} {int(total_pos):>14,} "
                f"{rate_s:>8} {total_games:>12,}"
            )
            if have_cost:
                cost_s = f"{total_cost:.3f}"
                if rated and total_rate > 0 and total_cost > 0:
                    per_m_s = f"{total_cost / (total_rate * 3600 / 1e6):.2f}"
                else:
                    per_m_s = "-"
                total_row += f" {cost_s:>7} {per_m_s:>8}"
            out.append(total_row)
            out.append("")

            if rated and total_rate > 0:

                def eta(target):
                    remaining = target - total_pos
                    if remaining <= 0:
                        return "✓"
                    secs = remaining / total_rate
                    label = fmt_secs(secs)
                    if total_cost > 0:
                        label += f" (${total_cost * secs / 3600:.2f})"
                    return label

                out.append(
                    f"ETA     1M: {eta(1e6)}   2M: {eta(2e6)}   "
                    f"5M: {eta(5e6)}   10M: {eta(1e7)}"
                )

            sys.stdout.write("\n".join(out) + "\n")
            sys.stdout.flush()

            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if hide_cursor:
            sys.stdout.write("\033[?25h\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
