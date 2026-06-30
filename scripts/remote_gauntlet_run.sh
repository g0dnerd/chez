#!/usr/bin/env bash
# Remote-side gauntlet runner. Runs a capped-Stockfish gauntlet for ONE Chez
# binary using fastchess directly (no Python). Designed to be copied to a fresh
# host alongside prebuilt stockfish + fastchess binaries and an opening book.
#
# Layout it assumes (created by scripts/remote_gauntlet.sh):
#   $ROOT/bin/{stockfish,fastchess,<uci>}
#   $ROOT/books/<book>
#   $ROOT/results/<label>/
#
# Usage:
#   remote_gauntlet_run.sh ROOT UCI_NAME LABEL TC THREADS ROUNDS CONC BOOK "ELO1 ELO2 ..."
set -u
ROOT="$1"; UCI_NAME="$2"; LABEL="$3"; TC="$4"; TH="$5"; ROUNDS="$6"; CONC="$7"; BOOK="$8"; shift 8
ELOS="$*"

BIN="$ROOT/bin"
OUT="$ROOT/results/$LABEL"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
UCI="$BIN/$UCI_NAME"
SF="$BIN/stockfish"
FC="$BIN/fastchess"
BOOKPATH="$ROOT/books/$BOOK"

{
  echo "# gauntlet $LABEL  uci=$UCI_NAME  TC=$TC  threads=$TH  rounds=$ROUNDS  conc=$CONC  book=$BOOK"
  echo "# host=$(hostname)  cores=$(nproc)  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$SUMMARY"

for elo in $ELOS; do
  log="$OUT/SF_${elo}.log"
  pgn="$OUT/Chez_vs_SF_${elo}.pgn"
  echo ">>> [$LABEL] vs SF_${elo} starting $(date -u +%H:%M:%S)" >> "$SUMMARY"
  "$FC" \
    -engine name=Chez cmd="$UCI" option.Threads=$TH option.OwnBook=false proto=uci \
    -engine name=SF_${elo} cmd="$SF" proto=uci option.Threads=1 option.UCI_LimitStrength=true option.UCI_Elo=${elo} \
    -each tc=$TC restart=on timemargin=300 \
    -rounds $ROUNDS -repeat -recover -concurrency $CONC \
    -openings file="$BOOKPATH" format=pgn order=random \
    -draw movenumber=40 movecount=10 score=5 -resign movecount=5 score=1000 \
    -ratinginterval 10 -output format=cutechess -pgnout file="$pgn" \
    > "$log" 2>&1
  # final "Score of Chez vs SF_xxxx: W - L - D  [pct] N"
  final=$(grep -E "Score of Chez" "$log" | tail -1)
  echo "    SF_${elo}: ${final:-NO_RESULT}" >> "$SUMMARY"
done

echo "# DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
