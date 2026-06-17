#!/usr/bin/env bash
#
# cloud_selfplay.sh -- build selfplay locally, push it to a rented cloud box,
# and launch a high-throughput data-generation run there. Does NOT pull results
# back; the dataset stays on the remote (fetch it yourself when ready).
#
# selfplay is pure CPU and embarrassingly parallel, so the play is: rent a big
# many-core box by the hour, run this, tear the box down afterwards.
#
# The native binary links only libc (the -lOpenCL flag is dropped by
# --as-needed since selfplay never touches the GPU), so the remote needs no Zig,
# no OpenCL loader, and no source -- only a glibc at least as new as the build
# host's. The script probes for that before launching and aborts with guidance
# if the remote image is too old (pick Debian 13 / Ubuntu 24.10+ to be safe).
#
# Example:
#   scripts/cloud_selfplay.sh --host root@203.0.113.7 \
#       --eval data/net_v7.nnue -- --games 4000000 --depth 10 --nodes 360000
#
# Everything after `--` is forwarded verbatim to scripts/shard_selfplay.sh on
# the remote (see that script's --help for --games/--depth/--threads/--out/...).

set -euo pipefail

host=""
identity=""
port=""
remote_dir='~/chez-selfplay'
eval_file=""
bin=./zig-out/bin/selfplay
do_build=1
detach=1

die() { echo "error: $*" >&2; exit 1; }

usage() {
  cat >&2 <<EOF
Usage: $0 --host USER@HOST [options] [-- shard_selfplay.sh options]
  --host USER@HOST    ssh destination (required)
  --identity PATH     ssh private key (-i)
  --port N            ssh port
  --remote-dir DIR    remote working directory (default $remote_dir)
  --eval PATH         local .nnue to upload and play with
  --bin PATH          local selfplay binary (default $bin)
  --no-build          skip the local 'zig build' step
  --foreground        stream the run and block until it finishes
                      (default: launch detached under nohup and return)
  -h, --help          this help

  Options after '--' are passed through to scripts/shard_selfplay.sh on the
  remote, e.g.:  -- --games 2000000 --depth 10 --threads 96 --out data.bin
EOF
  exit 1
}

# Split argv at the first bare '--': ours before, passthrough after.
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) host=$2; shift 2;;
    --identity) identity=$2; shift 2;;
    --port) port=$2; shift 2;;
    --remote-dir) remote_dir=$2; shift 2;;
    --eval) eval_file=$2; shift 2;;
    --bin) bin=$2; shift 2;;
    --no-build) do_build=0; shift;;
    --foreground) detach=0; shift;;
    -h|--help) usage;;
    --) shift; passthrough=("$@"); break;;
    *) die "unknown arg: $1 (see --help)";;
  esac
done

[[ -n "$host" ]] || { echo "error: --host is required" >&2; usage; }
[[ -z "$eval_file" || -f "$eval_file" ]] || die "eval file not found: $eval_file"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
shard_script="$script_dir/shard_selfplay.sh"
[[ -f "$shard_script" ]] || die "missing $shard_script"

# User-facing ssh/scp flags (key + port). These appear in the monitor/stop
# commands we print at the end, so keep them clean of internal plumbing.
base_ssh=()
base_scp=()
[[ -n "$identity" ]] && { base_ssh+=(-i "$identity"); base_scp+=(-i "$identity"); }
[[ -n "$port" ]] && { base_ssh+=(-p "$port"); base_scp+=(-P "$port"); }

# Connection multiplexing: the first connection opens a master socket that every
# later ssh/scp reuses, so authentication -- key, agent, or a one-time passphrase
# prompt for an encrypted key -- happens exactly once instead of per command.
ctl_dir=$(mktemp -d "${TMPDIR:-/tmp}/cloud_selfplay.XXXXXX")
mux=(-o ControlMaster=auto -o "ControlPath=$ctl_dir/ctl-%r@%h:%p" -o ControlPersist=120)
ssh_opts=("${mux[@]}" "${base_ssh[@]}")
scp_opts=("${mux[@]}" "${base_scp[@]}")
ssh() { command ssh "${ssh_opts[@]}" "$@"; }
scp() { command scp "${scp_opts[@]}" "$@"; }

cleanup() {
  # Close the shared master connection (the detached remote run keeps going --
  # it's under nohup, independent of this ssh session).
  [[ -n "${host:-}" ]] && command ssh "${ssh_opts[@]}" -O exit "$host" 2>/dev/null || true
  rm -rf "$ctl_dir"
}
trap cleanup EXIT

if [[ "$do_build" -eq 1 ]]; then
  echo ">> building selfplay locally (zig build)" >&2
  zig build
fi
[[ -x "$bin" ]] || die "selfplay binary not found/executable: $bin (drop --no-build, or pass --bin)"

# Record the glibc version this binary was built against, so the remote probe
# can give a precise error instead of a raw loader failure.
need_glibc=$(strings -a "$bin" 2>/dev/null \
  | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sort -V | tail -1 || true)
echo ">> binary requires ${need_glibc:-glibc (unknown version)}" >&2

# Resolve a leading ~ to the remote's absolute $HOME. A literal ~ survives fine
# in unquoted ssh strings, but gets escaped (not expanded) once embedded in a
# quoted remote command, so keep every remote path tilde-free. This first ssh
# also opens the shared master connection (the one-time auth prompt).
case "$remote_dir" in
  "~" | "~/"*)
    remote_home=$(ssh "$host" 'printf %s "$HOME"')
    [[ -n "$remote_home" ]] || die "could not resolve remote \$HOME"
    remote_dir="$remote_home${remote_dir#\~}"
    ;;
esac

echo ">> preparing remote: $host:$remote_dir" >&2
ssh "$host" "mkdir -p $remote_dir"

echo ">> uploading binary + shard runner" >&2
scp "$bin" "$host:$remote_dir/selfplay"
scp "$shard_script" "$host:$remote_dir/shard_selfplay.sh"
ssh "$host" "chmod +x $remote_dir/selfplay $remote_dir/shard_selfplay.sh"

remote_eval=""
if [[ -n "$eval_file" ]]; then
  echo ">> uploading eval network ($eval_file)" >&2
  scp "$eval_file" "$host:$remote_dir/$(basename "$eval_file")"
  remote_eval="$remote_dir/$(basename "$eval_file")"
fi

# Probe: run zero games. This loads the binary (catching a glibc mismatch up
# front) and prints the core count we'll be working with.
echo ">> probing remote (glibc + core count)" >&2
if ! ssh "$host" "cd $remote_dir && ./selfplay --num_games 0 --num_threads 1 >/dev/null 2>probe.err"; then
  echo "---- remote probe failed ----" >&2
  ssh "$host" "cat $remote_dir/probe.err" >&2 || true
  die "selfplay did not run on the remote. If you see a 'GLIBC_x.y not found' error \
above, the remote image is too old -- it needs glibc >= ${need_glibc#GLIBC_}. \
Use a newer image (Debian 13 / Ubuntu 24.10+) or rebuild on the box."
fi
remote_cores=$(ssh "$host" "nproc")
echo ">> remote has $remote_cores core(s)" >&2

# Build the remote command line for shard_selfplay.sh.
run_args=(--bin ./selfplay)
[[ -n "$remote_eval" ]] && run_args+=(--eval "$remote_eval")
run_args+=("${passthrough[@]}")

# Quote each arg so the remote shell sees them intact.
remote_cmd="cd $remote_dir && ./shard_selfplay.sh"
for a in "${run_args[@]}"; do
  remote_cmd+=" $(printf '%q' "$a")"
done

if [[ "$detach" -eq 1 ]]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  log="run_${stamp}.log"
  echo ">> launching detached; logging to $remote_dir/$log" >&2
  # setsid + all fds redirected off the ssh channel fully detaches the run, so
  # the client returns immediately instead of hanging until the channel drains.
  # -n keeps the client from holding its own stdin open.
  ssh -n "$host" "cd $remote_dir && setsid bash -c $(printf '%q' "$remote_cmd") >$log 2>&1 </dev/null & echo launched"
  echo "" >&2
  echo "Started. Monitor with:" >&2
  echo "  ssh ${base_ssh[*]} $host 'tail -f $remote_dir/$log'" >&2
  echo "Stop with:" >&2
  echo "  ssh ${base_ssh[*]} $host \"pkill -f '[s]elfplay --num_games'\"" >&2
else
  echo ">> running in foreground (Ctrl-C stops the local stream, not the remote run)" >&2
  ssh -t "$host" "$remote_cmd"
fi
