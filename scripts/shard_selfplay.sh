#!/usr/bin/env bash
#
# shard_selfplay.sh -- fan out multiple selfplay processes for high-throughput
# NNUE data generation on many-core boxes.
#
# Each shard is an independent selfplay process (its own TT and RNG seed --
# selfplay auto-seeds from getrandom, so shards diverge) writing to its own
# output file; the shards are concatenated into one dataset at the end.
# Sharding gives NUMA locality, crash isolation, and avoids contention on the
# single in-process write mutex.
#
# Example:
#   scripts/shard_selfplay.sh --games 2000000 --depth 10 --nodes 360000 \
#       --eval data/net_v7_lambda075.nnue --out data/selfplay_v8.bin

set -euo pipefail

bin=./zig-out/bin/selfplay
record_size=35          # 32 pos + 2 score + 1 wdl (must match selfplay.zig)
shards=0                # 0 => auto (NUMA nodes if numactl present, else 1)
threads=0               # 0 => auto (cores / shards)
games=100000            # total games across all shards
depth=10                # search-depth ceiling (node cap is the real limiter)
# Generous soft node cap: bounds worst-case per-move time (predictable fleet
# throughput), labels stay near depth-10 except on tactical explosions.
# 0 = uncapped (slower, unbounded on pathological positions).
nodes=200000
eval_file=""
openings=""             # balanced opening book (FEN per line); empty => startpos
random_plies=""         # random plies on top of the book root; empty => binary default
# NB: data-quality filters (keep decisive positions, adjudicate only clearly-won
# games) are binary defaults in selfplay.zig now (score_filter 10000,
# adjudication 2500cp) -- no need to set them here.
out=selfplay_data.bin
numa=auto               # auto | on | off
keep_shards=0

die() { echo "error: $*" >&2; exit 1; }

usage() {
  cat >&2 <<EOF
Usage: $0 [options]
  --games N           total games across all shards (default $games)
  --shards N          number of processes (default: NUMA node count, else 1)
  --threads N         worker threads per shard (default: cores / shards)
  --depth D           search depth (default $depth)
  --nodes N           soft per-move node cap, 0 = none (default $nodes)
  --eval PATH         .nnue network to play with
  --openings PATH     balanced opening book, FEN per line (default: startpos)
  --random_plies N    random plies on top of each book root (default: binary default)
  --out PATH          concatenated output file (default $out)
  --bin PATH          selfplay binary (default $bin)
  --numa on|off|auto  NUMA pinning, one shard per node (default $numa)
  --keep-shards       keep per-shard files (and logs) after merge
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --games) games=$2; shift 2;;
    --shards) shards=$2; shift 2;;
    --threads) threads=$2; shift 2;;
    --depth) depth=$2; shift 2;;
    --nodes) nodes=$2; shift 2;;
    --eval) eval_file=$2; shift 2;;
    --openings) openings=$2; shift 2;;
    --random_plies) random_plies=$2; shift 2;;
    --out) out=$2; shift 2;;
    --bin) bin=$2; shift 2;;
    --numa) numa=$2; shift 2;;
    --keep-shards) keep_shards=1; shift;;
    -h|--help) usage;;
    *) die "unknown arg: $1 (see --help)";;
  esac
done

[[ -x "$bin" ]] || die "selfplay binary not found/executable: $bin (run 'zig build')"
[[ -z "$eval_file" || -f "$eval_file" ]] || die "eval file not found: $eval_file"
[[ -z "$openings" || -f "$openings" ]] || die "openings file not found: $openings"

cores=$(nproc)

# NUMA node count (1 if numactl unavailable).
have_numactl=0
node_count=1
if command -v numactl >/dev/null 2>&1; then
  have_numactl=1
  node_count=$(numactl --hardware | awk '/^available:/ {print $2; exit}')
  [[ -n "$node_count" ]] || node_count=1
fi

# Derive shards/threads from hardware when not given.
if [[ "$shards" -eq 0 ]]; then
  if [[ "$numa" != off && "$have_numactl" -eq 1 ]]; then
    shards=$node_count
  else
    shards=1
  fi
fi
[[ "$shards" -ge 1 ]] || die "shards must be >= 1"

if [[ "$threads" -eq 0 ]]; then
  threads=$(( cores / shards ))
  [[ "$threads" -ge 1 ]] || threads=1
fi

# Pin one shard per NUMA node only when that mapping is exact.
pin=0
if [[ "$numa" == on ]]; then
  pin=1
elif [[ "$numa" == auto && "$have_numactl" -eq 1 && "$shards" -eq "$node_count" && "$node_count" -gt 1 ]]; then
  pin=1
fi

# Spread games across shards (remainder goes to the first shards).
base=$(( games / shards ))
rem=$(( games % shards ))
[[ "$base" -gt 0 || "$rem" -gt 0 ]] || die "games ($games) too small for $shards shards"

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/selfplay_shards.XXXXXX")
shard_files=()
pids=()

cleanup() {
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup INT TERM

echo "Sharding: $shards process(es), $threads thread(s) each, depth $depth, $games games" >&2
[[ "$nodes" -gt 0 ]] && echo "  node cap: $nodes" >&2
[[ -n "$eval_file" ]] && echo "  eval: $eval_file" >&2
[[ -n "$openings" ]] && echo "  openings: $openings" >&2
[[ "$pin" -eq 1 ]] && echo "  NUMA pinning: one shard per node ($node_count nodes)" >&2

start=$(date +%s)
for (( i=0; i<shards; i++ )); do
  g=$base
  [[ "$i" -lt "$rem" ]] && g=$(( g + 1 ))
  [[ "$g" -gt 0 ]] || continue

  f="$tmpdir/shard_$i.bin"
  shard_files+=("$f")

  args=(--num_games "$g" --num_threads "$threads" --depth "$depth")
  [[ "$nodes" -gt 0 ]] && args+=(--nodes "$nodes")
  [[ -n "$eval_file" ]] && args+=(--eval "$eval_file")
  [[ -n "$openings" ]] && args+=(--openings "$openings")
  [[ -n "$random_plies" ]] && args+=(--random_plies "$random_plies")

  if [[ "$pin" -eq 1 ]]; then
    numactl --cpunodebind="$i" --membind="$i" "$bin" "${args[@]}" >"$f" 2>"$tmpdir/shard_$i.log" &
  else
    "$bin" "${args[@]}" >"$f" 2>"$tmpdir/shard_$i.log" &
  fi
  pids+=("$!")
  echo "  shard $i: pid $! -> $f ($g games)" >&2
done

[[ "${#shard_files[@]}" -gt 0 ]] || die "no shards launched"

# Wait for all shards; fail loudly if any crashed (logs kept for inspection).
fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
trap - INT TERM
[[ "$fail" -eq 0 ]] || { keep_shards=1; die "one or more shards failed; logs in $tmpdir"; }

# Fixed-size records => plain concatenation yields a valid dataset.
cat "${shard_files[@]}" > "$out"

bytes=$(stat -c %s "$out")
positions=$(( bytes / record_size ))
secs=$(( $(date +%s) - start ))
echo "Done: $positions positions ($bytes bytes) -> $out in ${secs}s" >&2

if [[ "$keep_shards" -eq 1 ]]; then
  echo "Shard files/logs kept in $tmpdir" >&2
else
  rm -rf "$tmpdir"
fi
