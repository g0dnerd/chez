#!/usr/bin/env bash
# Provision a fresh remote host and run a capped-Stockfish gauntlet for one
# prebuilt Chez UCI binary, in the background (survives disconnect via nohup).
# Copies prebuilt stockfish + fastchess (no apt/zig/build needed on the remote),
# so every host uses the IDENTICAL Stockfish for comparable results.
#
# Subcommands:
#   provision HOST LOCAL_UCI LABEL [ROUNDS] [TC] [THREADS] [ELOS]
#       Copy binaries+book+runner, launch the gauntlet under nohup, return.
#   status    HOST LABEL
#       Print the remote summary + per-opponent live tallies.
#   fetch     HOST LABEL DEST_DIR
#       Copy the remote results/<LABEL>/ back to DEST_DIR.
#
# Env: SSH_KEY (default ~/.ssh/cloud), REMOTE_ROOT (default /root/cgaunt),
#      BOOK (default testing/books/8moves_v3.pgn).
set -u

SSH_KEY="${SSH_KEY:-$HOME/.ssh/cloud}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/cgaunt}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BOOK_LOCAL="${BOOK:-$REPO/testing/books/8moves_v3.pgn}"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
SCP="scp -i $SSH_KEY -o StrictHostKeyChecking=accept-new -q"

die() { echo "ERROR: $*" >&2; exit 1; }

cmd_provision() {
  local host="$1" local_uci="$2" label="$3"
  local rounds="${4:-100}" tc="${5:-2+1}" threads="${6:-4}"
  local elos="${7:-2600 2800 3000 3190}"
  [ -x "$local_uci" ] || die "uci binary not found/executable: $local_uci"
  [ -f "$BOOK_LOCAL" ] || die "book not found: $BOOK_LOCAL"
  local sf; sf="$(readlink -f /usr/local/bin/stockfish)"
  [ -x "$sf" ] || die "local stockfish not found"
  local uci_name; uci_name="$(basename "$local_uci")"
  local book_name; book_name="$(basename "$BOOK_LOCAL")"

  echo "[$host] provisioning ($label)…"
  $SSH "root@$host" "mkdir -p $REMOTE_ROOT/bin $REMOTE_ROOT/books $REMOTE_ROOT/results" || die "ssh mkdir failed"
  # binaries (skip re-copy if same size already there)
  $SCP "$sf"                       "root@$host:$REMOTE_ROOT/bin/stockfish"
  $SCP /usr/local/bin/fastchess    "root@$host:$REMOTE_ROOT/bin/fastchess"
  $SCP "$local_uci"                "root@$host:$REMOTE_ROOT/bin/$uci_name"
  $SCP "$BOOK_LOCAL"               "root@$host:$REMOTE_ROOT/books/$book_name"
  $SCP "$REPO/scripts/remote_gauntlet_run.sh" "root@$host:$REMOTE_ROOT/remote_gauntlet_run.sh"
  $SSH "root@$host" "chmod +x $REMOTE_ROOT/bin/* $REMOTE_ROOT/remote_gauntlet_run.sh"

  # auto concurrency: floor((cores-1)/(threads+1)), >=1
  local conc
  conc="$($SSH "root@$host" "echo \$(( ( \$(nproc) - 1 ) / ( $threads + 1 ) ))")"
  [ "${conc:-0}" -ge 1 ] 2>/dev/null || conc=1

  echo "[$host] launching: $uci_name  TC=$tc threads=$threads rounds=$rounds conc=$conc  elos='$elos'"
  # </dev/null detaches stdin so the ssh channel closes immediately (otherwise
  # the backgrounded job keeps the channel open and ssh hangs).
  $SSH "root@$host" "cd $REMOTE_ROOT && setsid nohup ./remote_gauntlet_run.sh '$REMOTE_ROOT' '$uci_name' '$label' '$tc' '$threads' '$rounds' '$conc' '$book_name' '$elos' </dev/null > results/$label.nohup 2>&1 & echo \"launched pid \$!\""
  echo "[$host] running in background. Poll:  $0 status $host $label"
}

cmd_status() {
  local host="$1" label="$2"
  echo "=== [$host] $label summary ==="
  $SSH "root@$host" "cat $REMOTE_ROOT/results/$label/summary.txt 2>/dev/null || echo '(no summary yet)'"
  echo "=== live tally from PGNs ==="
  $SSH "root@$host" "
    for p in $REMOTE_ROOT/results/$label/Chez_vs_SF_*.pgn; do
      [ -e \"\$p\" ] || continue
      elo=\$(basename \"\$p\" .pgn | sed 's/.*SF_//')
      w=\$(grep -c '\\[Result \"1-0\"\\]' \"\$p\" 2>/dev/null)   # not Chez-relative; rough
      n=\$(grep -c '\\[Result ' \"\$p\" 2>/dev/null)
      echo \"  SF_\$elo: \$n games played\"
    done
    ! pgrep -f remote_gauntlet_run.sh >/dev/null && echo '  [runner finished]' || echo '  [runner still active]'
  "
}

cmd_fetch() {
  local host="$1" label="$2" dest="$3"
  mkdir -p "$dest"
  $SCP -r "root@$host:$REMOTE_ROOT/results/$label" "$dest/" && echo "fetched -> $dest/$label"
}

sub="${1:-}"; shift || true
case "$sub" in
  provision) cmd_provision "$@" ;;
  status)    cmd_status "$@" ;;
  fetch)     cmd_fetch "$@" ;;
  *) echo "usage: $0 {provision|status|fetch} …  (see header)"; exit 1 ;;
esac
