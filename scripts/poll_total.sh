#!/usr/bin/env bash
# Poll TOTAL recorded positions across the whole fleet (every active host in
# fleet.txt, honoring per-host identity/port), and exit when it reaches $TARGET.
# Counts the newest /tmp/selfplay_shards.* dir on each host (the current run).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TARGET="${1:-20000000}"
INTERVAL="${2:-300}"
DEFAULT_KEY="${3:-$HOME/.ssh/cloud}"

count_host() {  # key port host -> bytes on stdout
  local key="$1" port="$2" host="$3"
  # Pipe the command into a remote `bash -s` so it runs the same whether the
  # host's login shell is bash or fish (DGX/local box use fish). The heredoc
  # supplies ssh's stdin, so this is safe inside the while-read fleet.txt loop
  # (ssh never consumes the loop's stdin).
  ssh -i "$key" ${port:+-p "$port"} -o BatchMode=yes -o ConnectTimeout=12 \
    -o StrictHostKeyChecking=accept-new "$host" bash -s 2>/dev/null <<'RCMD'
d=$(ls -td /tmp/selfplay_shards.*/ 2>/dev/null | head -1); cat "$d"shard_*.bin 2>/dev/null | wc -c
RCMD
}

while true; do
  tot=0; up=0; down=0
  while IFS= read -r line; do
    line="${line%%#*}"; line="$(echo "$line" | xargs)"; [ -z "$line" ] && continue
    # shellcheck disable=SC2206
    f=($line); host="${f[0]}"; key=""; port=""
    case "${host##*@}" in localhost|127.0.0.1|::1) continue;; esac   # local: not an ssh shard host here
    for t in "${f[@]:1}"; do
      case "$t" in
        port=*) port="${t#port=}" ;;
        repo=*|build=*) : ;;
        *[!0-9.]*) key="$t" ;;        # non-numeric, non-key=val => identity path
        *) : ;;                        # bare number => cost
      esac
    done
    key="${key:-$DEFAULT_KEY}"; key="${key/#\~/$HOME}"
    b=$(count_host "$key" "$port" "$host")
    if [ -n "$b" ] && [ "$b" -gt 0 ] 2>/dev/null; then tot=$((tot+b)); up=$((up+1)); else down=$((down+1)); fi
  done < fleet.txt
  pos=$((tot/35))
  echo "$(date +%H:%M:%S) up=$up down=$down positions=$pos / $TARGET"
  [ "$pos" -ge "$TARGET" ] && { echo "THRESHOLD REACHED: $pos"; break; }
  sleep "$INTERVAL"
done
