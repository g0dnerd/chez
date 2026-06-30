#!/usr/bin/env bash
# tt_sweep_pps.sh -- sweep selfplay per-worker TT size, measure positions/second
# (the real data-gen throughput) AND nps. Interleaves configs across reps.
# Usage: tt_sweep_pps.sh <bin> <net> <games> <threads> <depth> <nodes> <reps> <bits...>
set -uo pipefail
bin="$1"; net="$2"; games="$3"; threads="$4"; depth="$5"; nodes="$6"; reps="$7"; shift 7
bits_list=("$@")

declare -A pps_sum nps_sum cnt
echo "# games=$games threads=$threads depth=$depth nodes=$nodes reps=$reps"
for r in $(seq 1 "$reps"); do
  for b in "${bits_list[@]}"; do
    "$bin" --num_games "$games" --depth "$depth" --nodes "$nodes" \
      --num_threads "$threads" --tt_bits "$b" --eval "$net" >/dev/null 2>run.err
    pos=$(grep -aoE '[0-9]+ positions' run.err | grep -oE '[0-9]+' | head -1)
    ms=$(grep -aoE '[0-9]+ms' run.err | grep -oE '[0-9]+' | head -1)
    nps=$(grep -aoE '[0-9]+ nps' run.err | grep -oE '[0-9]+' | head -1)
    pps=$(awk -v p="$pos" -v m="$ms" 'BEGIN{ if(m>0) printf "%.1f", p*1000/m; else print 0 }')
    pps_sum[$b]=$(awk -v s="${pps_sum[$b]:-0}" -v v="$pps" 'BEGIN{print s+v}')
    nps_sum[$b]=$(( ${nps_sum[$b]:-0} + ${nps:-0} ))
    cnt[$b]=$(( ${cnt[$b]:-0} + 1 ))
    printf "rep %d  bits=%-3s  pos=%-5s ms=%-7s  pps=%-7s nps=%s\n" "$r" "$b" "$pos" "$ms" "$pps" "$nps"
  done
done

echo "----  means (pos/s is the throughput metric)  ----"
base=""
for b in "${bits_list[@]}"; do
  pm=$(awk -v s="${pps_sum[$b]}" -v c="${cnt[$b]}" 'BEGIN{printf "%.1f", s/c}')
  nm=$(( ${nps_sum[$b]} / ${cnt[$b]} ))
  [ -z "$base" ] && base=$pm
  awk -v b="$b" -v pm="$pm" -v nm="$nm" -v base="$base" \
    'BEGIN{printf "bits=%-3s  pos/s=%-8s (%+.2f%%)   nps=%d\n", b, pm, (pm/base-1)*100, nm}'
done
