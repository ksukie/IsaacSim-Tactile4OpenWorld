#!/usr/bin/env bash
set -e

cd ~/IsaacLab-v2.3.2

rm -rf gelsight_original_record

./isaaclab.sh -p experiments/sensors/openworldtactile_gelsight_original_sensor.py \
  --use_tactile_rgb \
  --use_tactile_ff \
  --contact_object_type nut \
  --num_envs 1 \
  --save_viz \
  --save_viz_dir gelsight_original_record \
  --enable_cameras
