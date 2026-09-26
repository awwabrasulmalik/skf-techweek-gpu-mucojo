#!/bin/sh
# Fetch Unitree MuJoCo assets (pinned commit) into prototype/assets/.
# Assets are git-ignored; this script is the reproducible source.
set -e
REF=1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d
URL=https://github.com/UnitreeRobotics/unitree_mujoco
DST=prototype/assets/unitree_mujoco
if [ -d "$DST" ]; then echo "assets present: $DST"; exit 0; fi
mkdir -p prototype/assets
git clone --filter=blob:none --no-checkout "$URL" "$DST"
git -C "$DST" sparse-checkout set unitree_robots/g1
git -C "$DST" checkout "$REF"
echo "fetched $REF"
