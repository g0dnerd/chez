#!/usr/bin/env bash
# nps_ab_awk.sh <binA> <binB> <reps> <bench-args...>   (bc-free; awk only)
# Interleaves A,B runs pinned to one core; reports per-rep NPS + B/A, mean ratio,
# noise (CoV), and verifies the reproducible node-count signature matches.
set -uo pipefail
binA="$1"; binB="$2"; reps="$3"; shift 3
args=("$@")
CORE="${CORE:-3}"

declare -a A=() B=()
nodesA=""; nodesB=""; node_mismatch=0
nps()   { grep -E "^NPS:"   | grep -oE "[0-9]+" | head -1; }
nodes() { grep -E "^Nodes:" | grep -oE "[0-9]+" | head -1; }

for r in $(seq 1 "$reps"); do
  oa=$(taskset -c "$CORE" "$binA" "${args[@]}" 2>/dev/null)
  ob=$(taskset -c "$CORE" "$binB" "${args[@]}" 2>/dev/null)
  na=$(echo "$oa" | nps); nb=$(echo "$ob" | nps)
  xa=$(echo "$oa" | nodes); xb=$(echo "$ob" | nodes)
  A+=("$na"); B+=("$nb")
  [ -z "$nodesA" ] && nodesA="$xa"; [ -z "$nodesB" ] && nodesB="$xb"
  [ "$xa" != "$xb" ] && node_mismatch=1
  awk -v r="$r" -v a="$na" -v b="$nb" 'BEGIN{printf "  rep %2d:  A=%-9s B=%-9s  B/A=%.4f\n", r, a, b, b/a}'
done

stats() { printf '%s\n' "$@" | awk '{x[NR]=$1; s+=$1} END{m=s/NR; for(i=1;i<=NR;i++)v+=(x[i]-m)^2; sd=sqrt(v/NR); printf "%.0f %.2f", m, 100*sd/m}'; }
read am acov <<<"$(stats "${A[@]}")"
read bm bcov <<<"$(stats "${B[@]}")"
printf "  ----\n"
printf "  A: mean=%-9s NPS  CoV=%s%%\n" "$am" "$acov"
printf "  B: mean=%-9s NPS  CoV=%s%%\n" "$bm" "$bcov"
awk -v a="$am" -v b="$bm" 'BEGIN{printf "  B/A = %.4f  (%+.2f%%)\n", b/a, (b/a-1)*100}'
if [ "$node_mismatch" = "1" ]; then
  printf "  *** NODE SIGNATURE MISMATCH: A=%s B=%s (TREE CHANGED) ***\n" "$nodesA" "$nodesB"
else
  printf "  node signature identical (A=B=%s) -> tree-identical\n" "$nodesA"
fi
