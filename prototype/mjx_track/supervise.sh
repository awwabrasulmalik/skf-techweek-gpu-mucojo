#!/bin/bash
# supervise.sh -- GPU-track training supervisor (this fork only).
# Every 5 min: run alive -> heartbeat; finished -> eval+video -> decide ->
# launch next (promote / continue / preset-reshape) or BLOCKED (needs agent).
# Kill switch: touch runs_mjx/SUPERVISOR_STOP. Single instance via lock dir.
# Launch: setsid -f bash prototype/mjx_track/supervise.sh </dev/null >>runs_mjx/supervisor.log 2>&1
# One test cycle: bash prototype/mjx_track/supervise.sh --once
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1
PY=.venv-mjx/bin/python
TRACK=prototype/mjx_track
VARS=runs_mjx/supervisor.vars
STATE=runs_mjx/SUPERVISOR_STATE.md
SLOG=runs_mjx/supervisor.log
INTERVAL=300

log() { echo "[$(date '+%m%d %H:%M:%S')] SUP: $*" >> "$SLOG"; }
note() { echo "- [$(date '+%m%d %H:%M')] $*" >> "$STATE"; }

alive_out() {
  pgrep -af '\.venv-mjx/bin/python prototype/mjx_track/train_g1\.py --phas[e]' 2>/dev/null \
    | grep -v pgrep | head -1 | sed -n 's/.*--out \([^ ]*\).*/\1/p'
}

latest_ckpt() { ls "$1" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1; }

save_vars() {
  cat > "$VARS" <<EOF
PHASE=$PHASE
TAG=$TAG
OUT=$OUT
LOGF=$LOGF
LAST_GATE=$LAST_GATE
LAST_LEN=$LAST_LEN
FLAT=$FLAT
RESHAPED=$RESHAPED
CRASHES=$CRASHES
STATUS=$STATUS
LAST_CMD="$LAST_CMD"
EOF
}

launch() { # $1=phase $2=timesteps $3=envs $4=resume $5=extra-args $6=tagprefix
  local tag="${6}_$(date +%m%d_%H%M)" out="checkpoints_mjx/$tag" logf="runs_mjx/$tag.log"
  local cmd="$PY $TRACK/train_g1.py --phase $1 --timesteps $2 --num-envs $3 --resume $4 --out $out --log $logf $5"
  # shellcheck disable=SC2086
  setsid -f bash -c "cd $REPO && $cmd >> $logf.stdout.log 2>&1" </dev/null
  sleep 25
  if [ "$(alive_out)" = "$out" ]; then
    PHASE=$1 TAG=$tag OUT=$out LOGF=$logf CRASHES=0 STATUS=RUNNING LAST_CMD="$cmd"
    if [ "$1" = "walk" ]; then LAST_GATE=none LAST_LEN= FLAT=0 RESHAPED=0; fi
    save_vars
    log "launched $tag ($1, resume $4, $5)"
    note "launched \`$tag\` ($1, resume \`$4\`) $5"
  else
    STATUS=BLOCKED save_vars
    log "LAUNCH FAILED $tag (see $logf.stdout.log)"
    note "BLOCKED: launch of \`$tag\` failed"
  fi
}

decide() { # $1=gate $2=total $3=len $4=ckpt
  local gate=$1 total=$2 len=$3 ckpt=$4 need=10 extra=""
  [ "$PHASE" = "walk" ] && need=7
  [ "$RESHAPED" = "1" ] && extra="--pushes off"
  if [ "$gate" -ge "$need" ]; then
    if [ "$PHASE" = "stand" ]; then
      log "PROMOTE stand->walk"; note "PROMOTED to walk"
      launch walk 100000000 2048 "$OUT/$ckpt" "" g1_walk
    else
      STATUS=BLOCKED save_vars
      log "walk gate passed: PROMOTE-TO-STAIRS needs agent (no stairs env yet)"
      note "WALK PASSED - needs agent to build stairs env"
    fi
    return
  fi
  if [ -z "$LAST_LEN" ] || [ "$len" -gt $((LAST_LEN * 12 / 10)) ]; then
    LAST_GATE="$gate/$total" LAST_LEN=$len FLAT=0 save_vars
    log "improving (len $len), continue $PHASE"; note "improving, continuing $PHASE"
    launch "$PHASE" 50000000 2048 "$OUT/$ckpt" "$extra" "g1_${PHASE}c"
    return
  fi
  FLAT=$((FLAT + 1)) LAST_GATE="$gate/$total" LAST_LEN=$len save_vars
  if [ "$FLAT" -ge 2 ]; then
    if [ "$PHASE" = "stand" ] && [ "$RESHAPED" = "0" ]; then
      RESHAPED=1 FLAT=0 save_vars
      log "flat twice -> reshape stand-easy (pushes off)"
      note "flat twice -> reshape: stand-easy (pushes off)"
      launch stand 50000000 2048 "$OUT/$ckpt" "--pushes off" "g1_stande"
    else
      STATUS=BLOCKED save_vars
      log "BLOCKED: $PHASE flat, needs agent judgment"
      note "BLOCKED: $PHASE flat (gate $gate/$total len $len), needs agent"
    fi
  else
    log "flat 1x (len $len), continue once more"
    note "flat 1x, continuing $PHASE once more"
    launch "$PHASE" 50000000 2048 "$OUT/$ckpt" "$extra" "g1_${PHASE}c"
  fi
}

cycle() {
  [ -f runs_mjx/SUPERVISOR_STOP ] && { log "STOP flag present, exiting"; exit 0; }
  # shellcheck disable=SC1090
  source "$VARS"
  if [ "$STATUS" = "BLOCKED" ] || [ "$STATUS" = "PACKAGING" ]; then
    log "idle ($STATUS)"; return
  fi
  if [ "$(date +%H%M)" -ge 1300 ]; then
    STATUS=PACKAGING save_vars
    log "13:00 packaging rule: no new launches"; note "PACKAGING MODE from 13:00"
    return
  fi
  now_out=$(alive_out)
  if [ -n "$now_out" ]; then
    if [ "$now_out" != "$OUT" ]; then
      log "foreign run alive ($now_out), standing down"; return
    fi
    last=$(grep "eval_rew" "$LOGF" 2>/dev/null | tail -1)
    log "alive $TAG :: ${last:-compiling}"
    return
  fi
  if ! grep -q "SAVED " "$LOGF" 2>/dev/null; then
    CRASHES=$((CRASHES + 1))
    if [ "$CRASHES" -ge 2 ]; then
      STATUS=BLOCKED save_vars
      log "BLOCKED: crash loop ($TAG)"; note "BLOCKED: \`$TAG\` crashed twice"
      return
    fi
    save_vars
    log "crash of $TAG, relaunching same cmd (attempt $CRASHES)"
    note "crash of \`$TAG\`, relaunching same settings"
    # shellcheck disable=SC2086
    setsid -f bash -c "cd $REPO && $LAST_CMD >> $LOGF.stdout.log 2>&1" </dev/null
    sleep 25
    return
  fi
  ckpt=$(latest_ckpt "$OUT")
  log "finished $TAG, eval ckpt $ckpt"
  ev_out=$(timeout 900 $PY $TRACK/eval_g1.py --ckpt "$OUT/$ckpt" --phase "$PHASE" \
    --episodes 10 --save-traj "videos_mjx/${TAG}_eval.npz" 2>&1 | tail -14)
  gate=$(echo "$ev_out" | sed -n 's/.*GATE: success \([0-9]*\)\/.*/\1/p')
  total=$(echo "$ev_out" | sed -n 's/.*GATE: success [0-9]*\/\([0-9]*\).*/\1/p')
  len=$(echo "$ev_out" | sed -n 's/.*mean len \([0-9]*\)\/.*/\1/p')
  if [ -z "$gate" ] || [ -z "$len" ]; then
    STATUS=BLOCKED save_vars
    log "BLOCKED: eval output unparseable"; note "BLOCKED: eval of \`$TAG/$ckpt\` unparseable"
    return
  fi
  timeout 600 $PY $TRACK/render_g1.py --traj "videos_mjx/${TAG}_eval.npz" \
    --out "videos_mjx/${TAG}_eval.mp4" >>"$SLOG" 2>&1
  note "gate \`$TAG/$ckpt\`: $gate/$total, len $len/1000, video \`videos_mjx/${TAG}_eval.mp4\`"
  log "gate $TAG/$ckpt: $gate/$total len=$len"
  decide "$gate" "$total" "$len" "$ckpt"
}

if [ "${1:-}" = "--once" ]; then cycle; exit 0; fi
if ! mkdir runs_mjx/supervisor.lock 2>/dev/null; then
  echo "supervisor already running (stale lock? rmdir runs_mjx/supervisor.lock)"
  exit 1
fi
trap 'rmdir runs_mjx/supervisor.lock 2>/dev/null' EXIT
log "supervisor started (interval ${INTERVAL}s)"
while true; do cycle; sleep "$INTERVAL"; done
