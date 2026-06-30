#!/usr/bin/env bash
# Runs ON the DGX, detached. At 08:00 local: stop selfplay, start the normal
# docker services so the box is usable after the weekend. Self-contained so it
# does not depend on any external session.
LOG="$HOME/dgx_resume_8am.log"
exec >>"$LOG" 2>&1
echo "=== resume job launched $(date) (pid $$) ==="
target=$(date -d "today 08:00" +%s)
now=$(date +%s)
wait=$(( target - now ))
echo "now=$(date) target=$(date -d "@$target") sleep=${wait}s"
if [ "$wait" -gt 0 ]; then sleep "$wait"; fi
echo "=== FIRING $(date) ==="
echo ">> stopping selfplay"
pkill -f "[s]elfplay --num_games"; sleep 3; pkill -9 -f "[s]elfplay --num_games"; sleep 1
echo "selfplay remaining: $(pgrep -cf '[s]elfplay --num_games')"
echo ">> starting docker services"
cd /opt/services && docker compose -f /opt/services/docker-compose.yml up -d
echo "docker compose up -d exit=$?"
docker compose -f /opt/services/docker-compose.yml ps
echo "=== DONE $(date) ==="
