#!/usr/bin/env bash
# nps_ab.sh <binA> <binB> <reps> <bench-args...>
# Interleaves A,B,A,B,... bench runs (cancels host drift), pinned to one core.
# Reports per-rep NPS + B/A ratio, mean ratio, noise (CoV), and verifies the
# reproducible node-count signature matches between A and B (tree-identical check).
set -uo pipefail
binA="$1"; binB="$2"; reps="$3"; shift 3
args=("$@")
CORE="${CORE:-3}"

declare -a A=() B=()
nodesA=""; nodesB=""; node_mismatch=0
extract_nps() { grep -E "^NPS:" | grep -oE "[0-9]+" | head -1; }
extract_nodes() { grep -E "^Nodes:" | grep -oE "[0-9]+" | head -1; }

for r in $(seq 1 "$reps"); do
  oa=$(taskset -c "$CORE" "$binA" "${args[@]}" 2>/dev/null)
  ob=$(taskset -c "$CORE" "$binB" "${args[@]}" 2>/dev/null)
  na=$(echo "$oa" | extract_nps); nb=$(echo "$ob" | extract_nps)
  xa=$(echo "$oa" | extract_nodes); xb=$(echo "$ob" | extract_nodes)
  A+=("$na"); B+=("$nb")
  [ -z "$nodesA" ] && nodesA="$xa"; [ -z "$nodesB" ] && nodesB="$xb"
  [ "$xa" != "$xb" ] && node_mismatch=1
  printf "  rep %2d:  A=%-9s B=%-9s  B/A=%s\n" "$r" "$na" "$nb" "$(echo "scale=4; $nb/$na" | bc -l)"
done

stats() { printf '%s\n' "$@" | awk '{x[NR]=$1; s+=$1} END{m=s/NR; for(i=1;i<=NR;i++)v+=(x[i]-m)^2; sd=sqrt(v/NR); printf "%.0f %.0f %.2f", m, sd, 100*sd/m}'; }
read am asd acov <<<"$(stats "${A[@]}")"
read bm bsd bcov <<<"$(stats "${B[@]}")"
printf "  ----\n"
printf "  A: mean=%-9s NPS  CoV=%s%%\n" "$am" "$acov"
printf "  B: mean=%-9s NPS  CoV=%s%%\n" "$bm" "$bcov"
printf "  B/A = %.4f  (%+.2f%%)\n" "$(echo "$bm/$am"|bc -l)" "$(echo "($bm/$am-1)*100"|bc -l)"
if [ "$node_mismatch" = "1" ]; then
  printf "  *** NODE SIGNATURE MISMATCH: A=%s B=%s (TREE CHANGED) ***\n" "$nodesA" "$nodesB"
else
  printf "  node signature identical (A=B=%s) -> tree-identical\n" "$nodesA"
fi
