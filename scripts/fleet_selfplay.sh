#!/usr/bin/env bash
#
# fleet_selfplay.sh -- run and manage NNUE selfplay data generation across any
# number of rented boxes at once. Each box generates an independent stream (its
# own RNG seed, so they diverge); the combined dataset is the merge of all of
# them.
#
# Subcommands:
#   start     build locally once, then upload + launch selfplay (detached) on
#             every host (delegates per host to cloud_selfplay.sh --no-build).
#   monitor   live aggregate dashboard: per-box + total positions and pos/s,
#             refreshing until Ctrl-C (--once for a single snapshot).
#   collect   stop selfplay on every host, download the current run's shards,
#             merge them into one validated dataset (via merge_datasets.fish),
#             and prune stale shard dirs left by past runs (keeping the one it
#             just pulled).
#
# Hosts come from a file (--hosts) and/or repeated --host flags. A hosts file
# has one entry per line:
#   "user@host [identity_path] [cost] [port=N] [repo=PATH] [build=FLAGS]".
# Fields after the host are matched by shape, in any order: a number is the
# per-host cost in $/HOUR, a key=value token is an option (port=N for a
# non-default ssh port, repo=/build= below), anything else the identity key.
# Blank lines and lines starting with # are ignored; hosts
# without their own key use --identity. The cost column is optional and only
# used by `monitor`, which sums it into a $/hour total, a $/1M-positions
# figure, and a projected cost on each ETA.
#
# A repo= token switches that host to build-on-remote: instead of uploading a
# binary, the box builds selfplay itself from the checkout at PATH (use build=
# for flags it needs, e.g. -Dgb10=true on the DGX Spark). Everything else --
# monitor, collect -- is identical to upload hosts.
#
# A host of localhost/127.0.0.1 is this machine: every step runs directly via
# bash/cp instead of ssh/scp, so no sshd or loopback key is needed. It joins the
# fleet like any other host (identity/port ignored).
#   # fleet.txt
#   debian@64.34.81.41   ~/.ssh/cloud   1.52
#   debian@64.34.81.42   ~/.ssh/cloud
#   paul@192.168.178.22  ~/.ssh/cloud   0.0  repo=~/misc/chez  build=-Dgb10=true
#   localhost            0.0
#
# Examples:
#   # Recommended defaults are built in (NNUE-labelled, depth 10, ~200k node cap,
#   # decisive positions kept), so a bare start is sound:
#   scripts/fleet_selfplay.sh start  --hosts fleet.txt --identity ~/.ssh/cloud
#   # Override per run after `--` (e.g. more games / different net):
#   scripts/fleet_selfplay.sh start  --hosts fleet.txt --identity ~/.ssh/cloud \
#       --eval data/net_v7_lambda075.nnue -- --games 4000000 --depth 10 --nodes 200000
#   scripts/fleet_selfplay.sh monitor --hosts fleet.txt --identity ~/.ssh/cloud
#   scripts/fleet_selfplay.sh collect --hosts fleet.txt --identity ~/.ssh/cloud \
#       --out data/selfplay_fleet.bin

set -uo pipefail   # not -e: per-host failures are handled, not fatal

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cloud="$script_dir/cloud_selfplay.sh"
merge="$script_dir/merge_datasets.fish"
record_size=35

die() { echo "error: $*" >&2; exit 1; }

# ---- argument parsing --------------------------------------------------------

[[ $# -ge 1 ]] || { echo "usage: $0 {start|monitor|collect} [options] [-- selfplay opts]" >&2; exit 1; }
cmd="$1"; shift

hosts_file=""
identity=""
remote_dir='~/chez-selfplay'
# Label self-play with the current best NNUE net (uploaded to each host).
# Bump this to the latest net each generation; pass --eval to override.
eval_file="data/net_v7_lambda075.nnue"
openings_file=""
do_build=1
interval=10
once=0
out=""
no_stop=0
force=0
declare -a cli_hosts=()
declare -a passthrough=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hosts) hosts_file=$2; shift 2;;
    --host) cli_hosts+=("$2"); shift 2;;
    --identity) identity=$2; shift 2;;
    --remote-dir) remote_dir=$2; shift 2;;
    --eval) eval_file=$2; shift 2;;
    --openings) openings_file=$2; shift 2;;
    --no-build) do_build=0; shift;;
    --interval) interval=$2; shift 2;;
    --once) once=1; shift;;
    --out) out=$2; shift 2;;
    --no-stop) no_stop=1; shift;;
    --force) force=1; shift;;
    -h|--help) echo "see header of $0 for usage" >&2; exit 0;;
    --) shift; passthrough=("$@"); break;;
    *) die "unknown arg: $1";;
  esac
done

# ---- build host/key tables ---------------------------------------------------

declare -a HOSTS=()
declare -a KEYS=()
declare -a COSTS=()
declare -a PORTS=()       # ssh port, empty => default 22
declare -a REPOS=()       # remote checkout path; set => build on that host
declare -a BUILDARGS=()   # extra 'zig build' flags for remote-build hosts

if [[ -n "$hosts_file" ]]; then
  [[ -f "$hosts_file" ]] || die "hosts file not found: $hosts_file"
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"                       # strip comments
    line="$(echo "$line" | xargs)"           # trim whitespace
    [[ -z "$line" ]] && continue
    # Fields after the host: a numeric token is the cost ($/hour), a key=value
    # token is an option (port=N, repo=PATH, build=FLAGS), anything else the
    # identity. A repo= token switches the host to build-on-remote (no binary
    # upload), e.g. the DGX Spark which needs build=-Dgb10=true. Order is free.
    # shellcheck disable=SC2206
    fields=($line)
    _host=${fields[0]}; _key=""; _cost=""; _port=""; _repo=""; _build=""
    for f in "${fields[@]:1}"; do
      if [[ "$f" =~ ^\$?[0-9]+(\.[0-9]+)?$ ]]; then _cost=${f#\$}
      elif [[ "$f" == port=* ]]; then _port=${f#port=}
      elif [[ "$f" == repo=* ]]; then _repo=${f#repo=}
      elif [[ "$f" == build=* ]]; then _build=${f#build=}
      else _key=$f; fi
    done
    HOSTS+=("$_host")
    KEYS+=("${_key:-$identity}")
    COSTS+=("$_cost")
    PORTS+=("$_port")
    REPOS+=("$_repo")
    BUILDARGS+=("$_build")
  done < "$hosts_file"
fi

for h in "${cli_hosts[@]:-}"; do
  [[ -z "$h" ]] && continue
  HOSTS+=("$h")
  KEYS+=("$identity")
  COSTS+=("")
  PORTS+=("")
  REPOS+=("")
  BUILDARGS+=("")
done

[[ ${#HOSTS[@]} -gt 0 ]] || die "no hosts given (use --hosts FILE and/or --host USER@HOST)"

# A loopback host means this machine: ssh_to/scp_from run locally (bash/cp), and
# `start` passes --local to cloud_selfplay.sh. Lets this box join the fleet with
# no sshd or loopback key.
declare -a LOCAL=()
for i in "${!HOSTS[@]}"; do
  case "${HOSTS[$i]##*@}" in
    localhost|127.0.0.1|::1) LOCAL+=(1);;
    *) LOCAL+=(0);;
  esac
done

# ---- ssh/scp helpers ---------------------------------------------------------

# ssh_to <index> <remote command...>
# The command is fed to a remote `bash -s` over stdin rather than handed to the
# host's login shell. Some boxes log in with fish (e.g. the DGX Spark), which
# can't parse the bash payloads below (pkill chains, $(...), globs); piping into
# bash -s runs them the same everywhere -- the same trick fleet_monitor.py uses.
ssh_to() {
  local i="$1"; shift
  local cmd="$*"
  [[ "${LOCAL[$i]}" -eq 1 ]] && { bash -c "$cmd"; return; }
  local opts=(-o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
  [[ -n "${KEYS[$i]}" ]] && opts+=(-i "${KEYS[$i]}")
  [[ -n "${PORTS[$i]}" ]] && opts+=(-p "${PORTS[$i]}")
  command ssh "${opts[@]}" "${HOSTS[$i]}" bash -s <<<"$cmd"
}

# scp_from <index> <remote path> <local path>
scp_from() {
  local i="$1" remote="$2" local="$3"
  [[ "${LOCAL[$i]}" -eq 1 ]] && { cp -f "$remote" "$local"; return; }
  local opts=(-o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
  [[ -n "${KEYS[$i]}" ]] && opts+=(-i "${KEYS[$i]}")
  [[ -n "${PORTS[$i]}" ]] && opts+=(-P "${PORTS[$i]}")
  command scp "${opts[@]}" "${HOSTS[$i]}:$remote" "$local"
}

# ---- start -------------------------------------------------------------------

cmd_start() {
  [[ -x "$cloud" ]] || die "missing $cloud"
  # Build locally only if some host needs an uploaded binary; remote-build hosts
  # (repo= set) build for themselves, so a fleet of only those skips it.
  local need_local_build=0 i
  for i in "${!HOSTS[@]}"; do
    [[ -z "${REPOS[$i]}" ]] && need_local_build=1
  done
  if [[ "$do_build" -eq 1 && "$need_local_build" -eq 1 ]]; then
    echo ">> building selfplay once locally (zig build)" >&2
    zig build || die "local build failed"
  fi

  # Preflight: reachability + skip hosts already running selfplay. This makes
  # `start` idempotent -- append new boxes to the hosts file and re-run to grow a
  # live fleet; running boxes are left untouched (override with --force).
  echo ">> preflight: ${#HOSTS[@]} host(s)" >&2
  local i
  local -a to_launch=()
  for i in "${!HOSTS[@]}"; do
    if ! ssh_to "$i" 'true' 2>/dev/null; then
      echo "   FAIL  ${HOSTS[$i]} (unreachable / auth)" >&2
      continue
    fi
    local running
    running=$(ssh_to "$i" 'pgrep -c -f "[s]elfplay --num_games" 2>/dev/null || true')
    if [[ "${running:-0}" -ge 1 && "$force" -ne 1 ]]; then
      echo "   SKIP  ${HOSTS[$i]} (already running; --force to relaunch)" >&2
    else
      echo "   ok    ${HOSTS[$i]}" >&2
      to_launch+=("$i")
    fi
  done

  if [[ ${#to_launch[@]} -eq 0 ]]; then
    echo ">> nothing to launch (all reachable hosts already running)" >&2
    return 0
  fi

  echo ">> launching on ${#to_launch[@]} host(s) (detached)" >&2
  local logdir; logdir=$(mktemp -d)
  declare -A pid_of=()
  for i in "${to_launch[@]}"; do
    (
      args=(--host "${HOSTS[$i]}" --no-build --remote-dir "$remote_dir")
      [[ "${LOCAL[$i]}" -eq 1 ]] && args+=(--local)
      [[ -n "${KEYS[$i]}" ]] && args+=(--identity "${KEYS[$i]}")
      [[ -n "${PORTS[$i]}" ]] && args+=(--port "${PORTS[$i]}")
      if [[ -n "${REPOS[$i]}" ]]; then
        args+=(--remote-build --repo "${REPOS[$i]}")
        [[ -n "${BUILDARGS[$i]}" ]] && args+=(--build-args "${BUILDARGS[$i]}")
      fi
      [[ -n "$eval_file" ]] && args+=(--eval "$eval_file")
      [[ -n "$openings_file" ]] && args+=(--openings "$openings_file")
      args+=(--)
      args+=("${passthrough[@]}")
      "$cloud" "${args[@]}"
    ) >"$logdir/$i.log" 2>&1 &
    pid_of[$i]=$!
  done

  local fail=0
  for i in "${to_launch[@]}"; do
    if wait "${pid_of[$i]}"; then
      echo "   launched ${HOSTS[$i]}" >&2
    else
      fail=$((fail+1))
      echo "   FAILED   ${HOSTS[$i]} -- last lines:" >&2
      tail -4 "$logdir/$i.log" | sed 's/^/      /' >&2
    fi
  done
  rm -rf "$logdir"

  echo "" >&2
  echo "Launched $(( ${#to_launch[@]} - fail ))/${#to_launch[@]} new host(s). Monitor with:" >&2
  echo "  $0 monitor ${hosts_file:+--hosts $hosts_file} ${identity:+--identity $identity}" >&2
  [[ "$fail" -eq 0 ]] || exit 1
}

# ---- monitor -----------------------------------------------------------------

cmd_monitor() {
  # The dashboard lives in fleet_monitor.py: it needs stable cumulative-average
  # rates and float cost math, both awkward in bash. We hand it each host as
  # "host|key|cost" and exec into it (so Ctrl-C goes straight to Python).
  local mon="$script_dir/fleet_monitor.py"
  [[ -f "$mon" ]] || die "missing $mon"
  command -v python3 >/dev/null || die "python3 not found (needed for monitor)"

  local nodes=() i
  for i in "${!HOSTS[@]}"; do
    nodes+=(--node "${HOSTS[$i]}|${KEYS[$i]}|${COSTS[$i]:-}|${PORTS[$i]:-}")
  done
  local extra=()
  [[ "$once" -eq 1 ]] && extra+=(--once)
  exec python3 "$mon" --interval "$interval" "${extra[@]}" "${nodes[@]}"
}

# ---- collect -----------------------------------------------------------------

cmd_collect() {
  [[ -f "$merge" ]] || die "missing $merge"
  command -v fish >/dev/null || die "fish not found (needed for merge_datasets.fish)"

  if [[ -z "$out" ]]; then
    out="data/selfplay_fleet_$(date +%Y%m%d_%H%M%S).bin"
  fi
  [[ -e "$out" ]] && die "output already exists: $out"
  mkdir -p "$(dirname "$out")"

  if [[ "$no_stop" -eq 0 ]]; then
    echo ">> stopping selfplay on ${#HOSTS[@]} host(s)" >&2
    local i
    for i in "${!HOSTS[@]}"; do
      ssh_to "$i" 'pkill -f "[s]elfplay --num_games" 2>/dev/null; pkill -f "tail -f /tmp/selfplay_shards" 2>/dev/null; true' \
        && echo "   stopped ${HOSTS[$i]}" >&2 || echo "   (no run / unreachable) ${HOSTS[$i]}" >&2
    done
    # Give the shard script a moment to hit its keep-shards fail path.
    sleep 2
  else
    echo ">> --no-stop: downloading a live snapshot (selfplay keeps running)" >&2
  fi

  local dldir; dldir=$(mktemp -d)
  echo ">> downloading shards" >&2
  local i
  for i in "${!HOSTS[@]}"; do
    (
      # Only the NEWEST selfplay_shards.* dir -- the current run. shard_selfplay.sh
      # leaves its tmpdir behind whenever it's killed (it only cleans up on clean
      # completion, and long data-gen runs are always pkill'd), so a box accrues
      # one stale dir per past run. Globbing all of them downloaded every run's
      # shard_0.bin onto the same local name (host{i}_shard_0.bin), clobbering to a
      # single random stale shard. Picking the newest dir matches what `monitor`
      # counts (it too reads the newest log), so the two agree.
      newest=$(ssh_to "$i" 'ls -td /tmp/selfplay_shards.*/ 2>/dev/null | head -1')
      [[ -z "$newest" ]] && { echo "   no shard on ${HOSTS[$i]}" >&2; exit 0; }
      files=$(ssh_to "$i" "ls ${newest}shard_*.bin 2>/dev/null")
      [[ -z "$files" ]] && { echo "   no shard on ${HOSTS[$i]}" >&2; exit 0; }
      local ok=1
      while IFS= read -r rf; do
        [[ -z "$rf" ]] && continue
        local base="host${i}_${rf##*/}"
        if scp_from "$i" "$rf" "$dldir/$base" 2>/dev/null; then
          echo "   ${HOSTS[$i]}:${rf##*/} -> $base ($(stat -c %s "$dldir/$base") bytes)" >&2
        else
          echo "   FAILED download ${HOSTS[$i]}:$rf" >&2
          ok=0
        fi
      done <<<"$files"
      # Prune every stale shard dir (all but the one we just pulled), so /tmp
      # doesn't grow one dir per past run. Only when all of this host's downloads
      # succeeded -- never delete remote data we failed to retrieve. The dir we
      # collected is kept as a remote backup until the next run replaces it.
      if [[ "$ok" -eq 1 ]]; then
        local pruned
        pruned=$(ssh_to "$i" "n=0; for d in /tmp/selfplay_shards.*/; do [ -d \"\$d\" ] || continue; [ \"\$d\" = '$newest' ] && continue; rm -rf \"\$d\" && n=\$((n+1)); done; echo \$n")
        [[ "${pruned:-0}" -gt 0 ]] && echo "   pruned $pruned stale shard dir(s) on ${HOSTS[$i]}" >&2
      fi
    ) &
  done
  wait

  # Floor-truncate each part to a whole number of records (a process killed
  # mid-write can leave a trailing partial record).
  local parts=("$dldir"/*.bin)
  [[ -e "${parts[0]}" ]] || { rm -rf "$dldir"; die "no shards downloaded from any host"; }
  local p sz aligned
  for p in "${parts[@]}"; do
    sz=$(stat -c %s "$p")
    aligned=$(( sz / record_size * record_size ))
    if (( aligned != sz )); then
      truncate -s "$aligned" "$p"
      echo "   trimmed $(basename "$p") to $aligned bytes (dropped partial record)" >&2
    fi
  done

  echo ">> merging $(ls "$dldir"/*.bin | wc -l) shard(s) -> $out" >&2
  if fish "$merge" "$out" "${parts[@]}"; then
    rm -rf "$dldir"
    echo "" >&2
    echo "Collected -> $out" >&2
  else
    echo "" >&2
    echo "merge failed; downloaded parts kept in $dldir" >&2
    exit 1
  fi
}

# ---- dispatch ----------------------------------------------------------------

case "$cmd" in
  start)   cmd_start;;
  monitor) cmd_monitor;;
  collect) cmd_collect;;
  *) die "unknown command: $cmd (expected start|monitor|collect)";;
esac
