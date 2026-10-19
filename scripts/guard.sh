#!/bin/bash
# Memory guard and lifecycle for the DSV4.1 serve.
#
#   bash scripts/guard.sh status        # one snapshot: container, endpoint, memory, log tail
#   bash scripts/guard.sh watch [ctr] [stage] [log] [floor_kib] [swap_base_kib]
#   bash scripts/guard.sh stop          # container down, sysctls restored
#
# The floor is not optional. A serve without this guard armed is out of policy:
# 0.5 GiB MemAvailable, swap growth <= 64 MiB, sampled every 0.05 s (a 0.25 s
# loop overshot the floor by up to 90 MiB; kills landed at 308/363 MiB).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-status}"
CONTAINER="${2:-${CONTAINER:-dsv41-tp1}}"
STAGE="${3:-/tmp/none}"
LOG="${4:-/dev/null}"
FLOOR="${5:-524288}"          # 0.5 GiB in KiB
SWAP_BASE="${6:-0}"           # pre-boot SwapTotal-SwapFree, KiB
SWAP_MAX=$((64 * 1024))       # 64 MiB growth allowance, KiB
API_PORT="${API_PORT:-8000}"
# if env.sh is present, take its container/port
[ -f "$HERE/env.sh" ] && . "$HERE/env.sh" >/dev/null 2>&1 || true

swap_used() { awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{printf "%d", t-f}' /proc/meminfo; }
avail_kib() { awk '/^MemAvailable:/{printf "%d", $2}' /proc/meminfo; }

snapshot() {
  echo "date: $(date -Is)"
  echo "container: $(sudo -n docker ps -a --filter "name=$CONTAINER" --format '{{.Status}}' | head -1)"
  echo "endpoint: $(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null)"
  awk '/MemAvailable/{printf "mem_available_gib: %.2f\n", $2/1048576} /SwapTotal{t=$2} /SwapFree{f=$2} END{printf "swap_used_kib: %d\n", t-f}' /proc/meminfo
  echo "guard procs: $(pgrep -fc "[g]uard.sh watch" || true)"
  echo "log_tail:"
  sudo -n docker logs --tail 4 "$CONTAINER" 2>&1 | cut -c1-160
}

case "$MODE" in
  status) snapshot; exit 0 ;;
  stop)
    sudo -n docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    pkill -f "[g]uard.sh watch" 2>/dev/null || true
    # measured host defaults on DGX OS
    sudo -n sysctl -q vm.compaction_proactiveness=20 vm.watermark_boost_factor=15000 vm.swappiness=60
    echo "stopped; sysctls restored: $(for k in vm.compaction_proactiveness vm.watermark_boost_factor vm.swappiness; do printf '%s=%s ' "$k" "$(cat "/proc/sys/${k//.//}")"; done)"
    exit 0 ;;
  watch) : ;;
  *) echo "usage: guard.sh status|watch|stop" >&2; exit 2 ;;
esac

echo "guard: waiting for $CONTAINER at $(date +%H:%M:%S) swap_base=${SWAP_BASE}KiB" >>"$LOG"
seen=0
for _ in $(seq 1 480); do
  if sudo -n docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
    seen=1; break
  fi
  sleep 0.05
done
if (( seen == 0 )); then
  echo "guard: $CONTAINER never started, exiting" >>"$LOG"
  exit 0
fi
echo "guard: armed on $CONTAINER at $(date +%H:%M:%S)" >>"$LOG"
CPID=$(sudo -n docker inspect -f '{{.State.Pid}}' "$CONTAINER" 2>/dev/null || echo 0)
LAST_ATTR=0

# top-3 swap consumers, sampled at most every 4 s (only when swap grows)
attr() {
  local now; now=$(date +%s)
  (( now - LAST_ATTR < 4 )) && return
  LAST_ATTR=$now
  local out="" s pid
  out=$(for p in /proc/[0-9]*; do
          s=$(awk '/^VmSwap:/{print $2}' "$p/status" 2>/dev/null) || continue
          [ -n "${s:-}" ] && (( s > 1024 )) && printf "%s %s %s\n" "$s" "${p#/proc/}" "$(cat "$p/comm" 2>/dev/null)"
        done | sort -rn | head -3 | tr '\n' ';')
  echo "guard-attr $(date +%H:%M:%S) ctr_pid=$CPID top_swap=[$out]" >>"$LOG"
}

TICK=0
while true; do
  # `docker inspect` costs ~20 ms; poll it once a second, not every 50 ms sample
  if (( TICK % 20 == 0 )); then
    running=$(sudo -n docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo missing)
  fi
  TICK=$(( TICK + 1 ))
  avail=$(avail_kib)
  swap=$(swap_used)
  stage=$(cat "$STAGE" 2>/dev/null || echo unknown)
  echo "guard $(date +%H:%M:%S.%3N) stage=$stage avail=$((avail/1024))MiB swap=${swap}KiB" >>"$LOG"

  if (( avail < FLOOR )); then
    echo "guard: KILL(avail) stage=$stage avail=$((avail/1024))MiB swap=${swap}KiB" >>"$LOG"
    sudo -n docker kill "$CONTAINER" >>"$LOG" 2>&1
    exit 0
  fi
  (( swap - SWAP_BASE > 0 )) && attr
  if (( swap - SWAP_BASE > SWAP_MAX )); then
    echo "guard: KILL(swap) stage=$stage avail=$((avail/1024))MiB swap=${swap}KiB base=${SWAP_BASE}KiB" >>"$LOG"
    sudo -n docker kill "$CONTAINER" >>"$LOG" 2>&1
    exit 0
  fi
  if [[ "$running" != "true" ]]; then
    echo "guard: $CONTAINER ran and stopped, exiting" >>"$LOG"
    exit 0
  fi
  sleep 0.05
done