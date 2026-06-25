#!/usr/bin/env bash
#
# cloud_selfplay.sh -- build selfplay locally, push it to a rented cloud box,
# and launch a high-throughput data-generation run there. Does NOT pull results
# back; the dataset stays on the remote (fetch it yourself when ready).
#
# selfplay is pure CPU and embarrassingly parallel, so the play is: rent a big
# many-core box by the hour, run this, tear the box down afterwards.
#
# Two provisioning modes:
#   upload (default)  build selfplay locally and scp the binary up. The native
#                     binary links only libc (the -lOpenCL flag is dropped by
#                     --as-needed since selfplay never touches the GPU), so the
#                     remote needs no Zig, no OpenCL loader, and no source --
#                     only a glibc at least as new as the build host's. The
#                     script probes for that before launching and aborts with
#                     guidance if the remote image is too old (pick Debian 13 /
#                     Ubuntu 24.10+ to be safe).
#   --remote-build    the host already has a checkout (--repo) and needs a build
#                     that can't be cross-supplied (e.g. the DGX Spark, which
#                     needs -Dgb10=true). ssh in, `zig build <args>` there, and
#                     run the binary the box just built. No binary upload, no
#                     glibc probe -- it's native to the box.
#
# Example (upload):
#   scripts/cloud_selfplay.sh --host root@203.0.113.7 \
#       --eval data/net_v7.nnue -- --games 4000000 --depth 10 --nodes 360000
#
# Example (remote build, e.g. DGX Spark):
#   scripts/cloud_selfplay.sh --host paul@192.168.178.22 \
#       --remote-build --repo ~/misc/chez --build-args -Dgb10=true \
#       --eval data/net_v7.nnue -- --games 4000000 --depth 10 --nodes 360000
#
# A host of localhost/127.0.0.1 (or --local) runs everything directly via
# bash/cp instead of ssh/scp -- no sshd, key, or loopback needed -- so this box
# can join the fleet alongside rented remotes.
#
# Everything after `--` is forwarded verbatim to scripts/shard_selfplay.sh on
# the remote (see that script's --help for --games/--depth/--threads/--out/...).

set -euo pipefail

host=""
identity=""
port=""
remote_dir='~/chez-selfplay'
eval_file=""
openings_file=""
bin=./zig-out/bin/selfplay
do_build=1
detach=1
remote_build=0
repo=""
build_args=""
is_local=0

die() { echo "error: $*" >&2; exit 1; }

usage() {
  cat >&2 <<EOF
Usage: $0 --host USER@HOST [options] [-- shard_selfplay.sh options]
  --host USER@HOST    ssh destination (required)
  --identity PATH     ssh private key (-i)
  --port N            ssh port
  --remote-dir DIR    remote working directory (default $remote_dir)
  --eval PATH         local .nnue to upload and play with
  --openings PATH     local opening book (FEN/line) to upload and play from
  --bin PATH          local selfplay binary (default $bin)
  --no-build          skip the local 'zig build' step
  --remote-build      build on the remote from an existing checkout instead of
                      uploading a binary (requires --repo)
  --repo PATH         remote path to the checkout for --remote-build
  --build-args ARGS   extra flags for the remote 'zig build' (e.g. -Dgb10=true)
  --local             treat the host as this machine: run via bash/cp instead
                      of ssh/scp (auto-detected for localhost/127.0.0.1)
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
    --openings) openings_file=$2; shift 2;;
    --bin) bin=$2; shift 2;;
    --no-build) do_build=0; shift;;
    --remote-build) remote_build=1; shift;;
    --repo) repo=$2; shift 2;;
    --build-args) build_args=$2; shift 2;;
    --local) is_local=1; shift;;
    --foreground) detach=0; shift;;
    -h|--help) usage;;
    --) shift; passthrough=("$@"); break;;
    *) die "unknown arg: $1 (see --help)";;
  esac
done

[[ -n "$host" ]] || { echo "error: --host is required" >&2; usage; }
[[ -z "$eval_file" || -f "$eval_file" ]] || die "eval file not found: $eval_file"
[[ -z "$openings_file" || -f "$openings_file" ]] || die "openings file not found: $openings_file"
[[ "$remote_build" -eq 0 || -n "$repo" ]] || die "--remote-build requires --repo PATH"

# A loopback host means this machine: run directly, no ssh/scp.
case "${host##*@}" in localhost|127.0.0.1|::1) is_local=1;; esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
shard_script="$script_dir/shard_selfplay.sh"
[[ -f "$shard_script" ]] || die "missing $shard_script"

# User-facing ssh/scp flags (key + port). These appear in the monitor/stop
# commands we print at the end, so keep them clean of internal plumbing.
base_ssh=()
base_scp=()
[[ -n "$identity" ]] && { base_ssh+=(-i "$identity"); base_scp+=(-i "$identity"); }
[[ -n "$port" ]] && { base_ssh+=(-p "$port"); base_scp+=(-P "$port"); }

if [[ "$is_local" -eq 1 ]]; then
  # Local transport: ignore ssh flags/host and run the trailing command string;
  # for scp, strip the "host:" prefix and copy. Covers exactly the call shapes
  # this script uses (one command as the last arg; single-source uploads).
  ctl_dir=""
  ssh() {
    [[ "$1" == "-O" ]] && return 0          # control-socket ops: no-op locally
    local cmd="${*: -1}"
    [[ -z "$cmd" || "$cmd" == "$host" ]] && return 0
    bash -c "$cmd"
  }
  scp() {
    local src dst
    if [[ "$2" == *:* ]]; then src="$1"; dst="${2#*:}"; else src="${1#*:}"; dst="$2"; fi
    cp -f "$src" "$dst"
  }
else
  # Connection multiplexing: the first connection opens a master socket that
  # every later ssh/scp reuses, so authentication -- key, agent, or a one-time
  # passphrase prompt for an encrypted key -- happens once, not per command.
  ctl_dir=$(mktemp -d "${TMPDIR:-/tmp}/cloud_selfplay.XXXXXX")
  mux=(-o ControlMaster=auto -o "ControlPath=$ctl_dir/ctl-%r@%h:%p" -o ControlPersist=120)
  ssh_opts=("${mux[@]}" "${base_ssh[@]}")
  scp_opts=("${mux[@]}" "${base_scp[@]}")
  ssh() { command ssh "${ssh_opts[@]}" "$@"; }
  scp() { command scp "${scp_opts[@]}" "$@"; }
fi

cleanup() {
  # Close the shared master connection (the detached remote run keeps going --
  # it's under nohup, independent of this ssh session). No socket when local.
  [[ "$is_local" -eq 0 && -n "${host:-}" ]] && command ssh "${ssh_opts[@]}" -O exit "$host" 2>/dev/null || true
  [[ -n "$ctl_dir" ]] && rm -rf "$ctl_dir"
  return 0   # never let the trap's last command taint the script's exit status
}
trap cleanup EXIT

need_glibc=""
if [[ "$remote_build" -eq 0 ]]; then
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
fi

# Resolve a leading ~ to the remote's absolute $HOME. A literal ~ survives fine
# in unquoted ssh strings, but gets escaped (not expanded) once embedded in a
# quoted remote command, so keep every remote path tilde-free. This first ssh
# also opens the shared master connection (the one-time auth prompt).
remote_home=""
resolve_remote() {  # echo $1 with a leading ~ swapped for the remote $HOME
  case "$1" in
    "~" | "~/"*)
      [[ -n "$remote_home" ]] || remote_home=$(ssh "$host" 'printf %s "$HOME"')
      [[ -n "$remote_home" ]] || die "could not resolve remote \$HOME"
      printf %s "$remote_home${1#\~}";;
    *) printf %s "$1";;
  esac
}
remote_dir=$(resolve_remote "$remote_dir")

echo ">> preparing remote: $host:$remote_dir" >&2
ssh "$host" "mkdir -p $remote_dir"

# Determine the binary to run on the remote: either one we upload, or one the
# box builds for itself from an existing checkout.
if [[ "$remote_build" -eq 1 ]]; then
  repo=$(resolve_remote "$repo")
  ssh "$host" "test -d $repo" || die "remote repo not found: $host:$repo"
  # Build only the selfplay binary. The v14 net-isolation branch reverts
  # nnue.zig to format-v4/HKP2 (to label with the v11 net), which drops the
  # output-bucket API that the trainer/inspector targets use -- so a full
  # `zig build` fails on those. The `selfplay-only` step builds just selfplay.
  echo ">> building on remote: $host:$repo (zig build selfplay-only $build_args)" >&2
  if ! ssh "$host" "cd $repo && zig build selfplay-only $build_args"; then
    die "remote build failed in $repo. Is zig on the remote PATH, and are the \
build args ($build_args) correct?"
  fi
  ssh "$host" "cd $repo && git log -1 --oneline 2>/dev/null | sed 's/^/>> remote HEAD: /'" >&2 || true
  remote_bin="$repo/zig-out/bin/selfplay"

  echo ">> uploading shard runner" >&2
  scp "$shard_script" "$host:$remote_dir/shard_selfplay.sh"
  ssh "$host" "chmod +x $remote_dir/shard_selfplay.sh"
else
  remote_bin="$remote_dir/selfplay"
  echo ">> uploading binary + shard runner" >&2
  scp "$bin" "$host:$remote_dir/selfplay"
  scp "$shard_script" "$host:$remote_dir/shard_selfplay.sh"
  ssh "$host" "chmod +x $remote_dir/selfplay $remote_dir/shard_selfplay.sh"
fi

remote_eval=""
if [[ -n "$eval_file" ]]; then
  echo ">> uploading eval network ($eval_file)" >&2
  scp "$eval_file" "$host:$remote_dir/$(basename "$eval_file")"
  remote_eval="$remote_dir/$(basename "$eval_file")"
fi

remote_openings=""
if [[ -n "$openings_file" ]]; then
  echo ">> uploading opening book ($openings_file)" >&2
  scp "$openings_file" "$host:$remote_dir/$(basename "$openings_file")"
  remote_openings="$remote_dir/$(basename "$openings_file")"
fi

# Probe: run zero games. This loads the binary (catching a glibc mismatch on
# uploaded binaries up front) and prints the core count we'll be working with.
echo ">> probing remote (binary + core count)" >&2
if ! ssh "$host" "cd $remote_dir && $remote_bin --num_games 0 --num_threads 1 >/dev/null 2>probe.err"; then
  echo "---- remote probe failed ----" >&2
  ssh "$host" "cat $remote_dir/probe.err" >&2 || true
  if [[ "$remote_build" -eq 1 ]]; then
    die "the binary built on $host did not run (see error above)."
  fi
  die "selfplay did not run on the remote. If you see a 'GLIBC_x.y not found' error \
above, the remote image is too old -- it needs glibc >= ${need_glibc#GLIBC_}. \
Use a newer image (Debian 13 / Ubuntu 24.10+) or rebuild on the box (--remote-build)."
fi
remote_cores=$(ssh "$host" "nproc")
echo ">> remote has $remote_cores core(s)" >&2

# Build the remote command line for shard_selfplay.sh.
run_args=(--bin "$remote_bin")
[[ -n "$remote_eval" ]] && run_args+=(--eval "$remote_eval")
[[ -n "$remote_openings" ]] && run_args+=(--openings "$remote_openings")
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
  if [[ "$is_local" -eq 1 ]]; then
    echo "Started. Monitor with:" >&2
    echo "  tail -f $remote_dir/$log" >&2
    echo "Stop with:" >&2
    echo "  pkill -f '[s]elfplay --num_games'" >&2
  else
    echo "Started. Monitor with:" >&2
    echo "  ssh ${base_ssh[*]} $host 'tail -f $remote_dir/$log'" >&2
    echo "Stop with:" >&2
    echo "  ssh ${base_ssh[*]} $host \"pkill -f '[s]elfplay --num_games'\"" >&2
  fi
else
  echo ">> running in foreground (Ctrl-C stops the local stream, not the remote run)" >&2
  ssh -t "$host" "$remote_cmd"
fi
