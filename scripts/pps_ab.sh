#!/usr/bin/env bash
# pps_ab.sh BASE CAND NET GAMES THREADS DEPTH NODES REPS
# Interleaved base-vs-candidate self-play throughput A/B (positions/second).
# Interleaving base/cand within each rep cancels host drift. Run via bash on the
# remote (login shell may be fish).
set -uo pipefail
base="$1"; cand="$2"; net="$3"; games="$4"; threads="$5"; depth="$6"; nodes="$7"; reps="$8"
run() {  # $1 binary -> echoes "pos ms"
  "$1" --num_games "$games" --depth "$depth" --nodes "$nodes" \
       --num_threads "$threads" --tt_bits 21 --eval "$net" >/dev/null 2>/tmp/r.err
  local pos ms
  pos=$(grep -aoE '[0-9]+ positions' /tmp/r.err | grep -oE '[0-9]+' | head -1)
  ms=$(grep -aoE '[0-9]+ms' /tmp/r.err | grep -oE '[0-9]+' | head -1)
  echo "$pos $ms"
}
bsum=0; csum=0; cnt=0
echo "# games=$games threads=$threads depth=$depth nodes=$nodes reps=$reps"
for r in $(seq 1 "$reps"); do
  read -r bp bm < <(run "$base")
  read -r cp cm < <(run "$cand")
  bpps=$(awk -v p="$bp" -v m="$bm" 'BEGIN{printf "%.1f", (m>0)?p*1000/m:0}')
  cpps=$(awk -v p="$cp" -v m="$cm" 'BEGIN{printf "%.1f", (m>0)?p*1000/m:0}')
  bsum=$(awk -v s="$bsum" -v v="$bpps" 'BEGIN{print s+v}')
  csum=$(awk -v s="$csum" -v v="$cpps" 'BEGIN{print s+v}')
  cnt=$((cnt+1))
  printf "rep %d  base pos=%-6s ms=%-7s pps=%-7s | cand pos=%-6s ms=%-7s pps=%-7s | d=%+.1f%%\n" \
    "$r" "$bp" "$bm" "$bpps" "$cp" "$cm" "$cpps" \
    "$(awk -v b="$bpps" -v c="$cpps" 'BEGIN{print (b>0)?(c/b-1)*100:0}')"
done
awk -v b="$bsum" -v c="$csum" -v n="$cnt" \
  'BEGIN{bm=b/n; cm=c/n; printf "MEAN  base=%.1f  cand=%.1f  delta=%+.2f%%\n", bm, cm, (cm/bm-1)*100}'
