#!/usr/bin/env bash
# thread_sweep.sh -- sweep selfplay worker-thread count, measure positions/second
# (throughput per box). games scales with threads to keep wall-time ~constant.
# Usage: thread_sweep.sh <bin> <net> <games_per_thread> <depth> <nodes> <bits> <reps> <threads...>
set -uo pipefail
bin="$1"; net="$2"; gpt="$3"; depth="$4"; nodes="$5"; bits="$6"; reps="$7"; shift 7
thr_list=("$@")

declare -A pps_sum cnt
echo "# games_per_thread=$gpt depth=$depth nodes=$nodes bits=$bits reps=$reps"
for r in $(seq 1 "$reps"); do
  for t in "${thr_list[@]}"; do
    g=$(( gpt * t ))
    "$bin" --num_games "$g" --depth "$depth" --nodes "$nodes" \
      --num_threads "$t" --tt_bits "$bits" --eval "$net" >/dev/null 2>run.err
    pos=$(grep -aoE '[0-9]+ positions' run.err | grep -oE '[0-9]+' | head -1)
    ms=$(grep -aoE '[0-9]+ms' run.err | grep -oE '[0-9]+' | head -1)
    pps=$(awk -v p="$pos" -v m="$ms" 'BEGIN{ if(m>0) printf "%.1f", p*1000/m; else print 0 }')
    pps_sum[$t]=$(awk -v s="${pps_sum[$t]:-0}" -v v="$pps" 'BEGIN{print s+v}')
    cnt[$t]=$(( ${cnt[$t]:-0} + 1 ))
    printf "rep %d  threads=%-3s  g=%-5s pos=%-5s ms=%-7s  pos/s=%s\n" "$r" "$t" "$g" "$pos" "$ms" "$pps"
  done
done

echo "----  means  ----"
base=""
for t in "${thr_list[@]}"; do
  pm=$(awk -v s="${pps_sum[$t]}" -v c="${cnt[$t]}" 'BEGIN{printf "%.1f", s/c}')
  [ -z "$base" ] && base=$pm
  awk -v t="$t" -v pm="$pm" -v base="$base" \
    'BEGIN{printf "threads=%-3s  pos/s=%-8s (%+.2f%% vs first)\n", t, pm, (pm/base-1)*100}'
done
