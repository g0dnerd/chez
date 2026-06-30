#!/usr/bin/env bash
# Ground-truth poll across the whole fleet: sum newest-shard-dir bytes per host,
# positions = bytes/35. Trips (exit 0) when combined positions >= TARGET_POS.
TARGET_POS="${1:-40000000}"
SIZECMD='d=$(ls -td /tmp/selfplay_shards.*/ 2>/dev/null | head -1); stat -c%s ${d}shard_*.bin 2>/dev/null | awk "{s+=\$1} END{print s+0}"'

# host spec: "label|ssh_target|ssh_opts"  (ssh_target empty => localhost)
HOSTS=(
  "dgx|admin-paul@10.83.242.109|-i $HOME/.ssh/blocklist -p 2222"
  "local||"
  "c223|root@51.159.203.223|-i $HOME/.ssh/cloud"
  "c219|root@51.159.203.219|-i $HOME/.ssh/cloud"
  "c133|root@51.158.36.133|-i $HOME/.ssh/cloud"
  "c6|root@51.158.37.6|-i $HOME/.ssh/cloud"
)

read_host() { # $1=target $2=opts
  if [ -z "$1" ]; then bash -c "$SIZECMD" 2>/dev/null
  else echo "$SIZECMD" | ssh $2 -o ConnectTimeout=15 "$1" bash -s 2>/dev/null; fi
}

for n in $(seq 1 240); do
  total=0; line=""
  for spec in "${HOSTS[@]}"; do
    IFS='|' read -r lbl tgt opts <<<"$spec"
    b=$(read_host "$tgt" "$opts"); b=${b:-0}
    total=$(( total + b ))
    line="$line $lbl=$(( b/35 ))"
  done
  pos=$(( total / 35 ))
  echo "$(date +%H:%M:%S) iter=$n total_pos=$pos |$line"
  if [ "$pos" -ge "$TARGET_POS" ]; then echo "TARGET_REACHED total_pos=$pos"; exit 0; fi
  sleep 180
done
echo "POLL_TIMEOUT total_pos=$pos"
