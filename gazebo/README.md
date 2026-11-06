# Gazebo / Real-UAV Deployment — R2C + Hybrid Swarm Controllers

This package flies the exact controllers used in the experiments
(`r2c/controller.py` R2C, `r2c/hybrid.py` cluster-belief hybrid) on
Gazebo-simulated or real drones. One core bridge (`SwarmBridge`) serves all
targets; only the pose source and the velocity sink change.

```
        poses in                     velocities out
  Gazebo / mocap / GPS ──► SwarmBridge ──► gz VelocityControl | MAVROS setpoints
                              │
                    R2CController | HybridController   (unchanged r2c/ code)
                              │
                    RF shadow (sim) | onboard RSSI/SNR sensing (real)
```

## Path A — fallback simulator (no ROS/Gazebo, runs anywhere)

Validates the full bridge code path kinematically:

```bash
PYTHONPATH=gazebo/uav_swarm_r2c:. python \
  gazebo/uav_swarm_r2c/uav_swarm_r2c/fallback_sim.py \
  --drones 20 --controller hybrid --jammer reactive
```

Smoke test: `PYTHONPATH=gazebo/uav_swarm_r2c:. python -m test.test_bridge`

## Path B — Gazebo Sim + ROS 2 (this package)

Requires ROS 2 Humble + Gazebo Sim (gz-sim >= 7) + ros_gz. On a machine
without root, RoboStack provides both via conda/mamba.

```bash
# 1. environment (RoboStack, user-space)
mamba create -n ros -c robostack-staging ros-humble-desktop ros-humble-ros-gzgarden python=3.10
mamba activate ros

# 2. build the bridge package
cd gazebo/uav_swarm_r2c
pip install numpy scipy torch          # r2c deps (torch CPU build is fine)
colcon build --symlink-install
source install/setup.bash

# 3. launch Gazebo + bridges + controller
ros2 launch ../launch/swarm_bringup.launch.py n_drones:=20 controller:=hybrid jammer_mode:=reactive

# 4. spawn drones (velocity-controlled sphere models)
for i in $(seq 0 19); do
  x=$((RANDOM % 900 + 50)); y=$((RANDOM % 900 + 50))
  bash ../spawn_drone.bash drone_$i $x $y > /tmp/drone_$i.sdf
  gz service -s /world/swarm_arena/create \
    --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
    --timeout 300 --req "sdf_filename: '/tmp/drone_$i.sdf'"
done
```

The bridge consumes poses on `/swarm/pose/{i}` (PoseStamped). The launch file
starts a `ros_gz_bridge` TF bridge; for a clean per-drone PoseStamped feed,
run the tiny relay from the repo root:

```bash
python gazebo/pose_relay.py   # /world/*/pose -> /swarm/pose/{i}
```

Velocities are published to `/swarm/cmd_vel/{i}` (Twist), consumed by each
model's `VelocityControl` plugin. With `gz gui` you can watch the swarm
self-heal after the bridge injects failures at `failure_time`.

## Path C — PX4 / ArduPilot SITL (flight-stack fidelity)

Use the vendor multi-vehicle SITL launchers (they bring their own Gazebo
worlds), then run the bridge against MAVROS:

```bash
# after PX4 SITL (Tools/simulation/gazebo-classic/sitl_multiple_run.sh -n 20)
ros2 run uav_swarm_r2c bridge_node --ros-args \
  -p n_drones:=20 -p controller:=hybrid -p use_mavros:=true \
  -p failure_time:=100 -p failure_ratio:=0.5
```

With `use_mavros:=true` setpoints go to
`/uav{i}/mavros/setpoint_velocity/cmd_vel`; put each vehicle in OFFBOARD.
Poses come from each vehicle's local position — remap `/swarm/pose/{i}` in a
relay as in Path B.

## Path D — real drones

Same bridge, same topics:

- **Poses in**: GPS (NavSatFix->local), motion-capture (VRPN/Qualisys), or
  UWB anchors; publish `PoseStamped` on `/swarm/pose/{i}` at >= 10 Hz.
- **Velocities out**: `use_mavros:=true`, vehicles in OFFBOARD mode.
- **RF layer**: in simulation the quality map comes from `r2c/rf.py`. On real
  hardware replace `RFLayer.sensed_jammer_power` with measured per-node
  jammer/interference power (RSSI/SNR from the radio). The hybrid's
  cluster-belief design was built for exactly this: each cluster member
  contributes one sensing sample (divided slots), fused into the shared
  belief — no ground truth ever enters the controller.
- **Dynamics**: the paper uses v_max = 1 m/s. Real multirotors fly faster;
  scale `R2CParams.v_max`, `r_c`, and the control rate together, and keep
  accel limits in the flight stack (the bridge emits velocity setpoints; the
  stack enforces feasibility).

## Scaling notes

- Gazebo comfortably runs ~20–50 velocity-controlled models headless
  (null rendering: `gz sim -r --render-engine ogre2` without GUI). 200 models
  crawl in one Gazebo instance; use the numeric simulator (`r2c.sim`,
  validated against the paper) for swarm-scale statistics and Gazebo for
  dynamic-fidelity demos.
- The control loop is 10 Hz with O(n^2) geometry at n=200 — trivial for any
  onboard computer; the IST/GCN training every 50 steps is the heavy part
  and runs fine on CPU (~0.2 s) or the Jetson-class GPU the R2C paper used.

## What is simulated vs real

| Component | Simulation (this repo) | Real deployment |
|---|---|---|
| Motion | Gazebo physics / kinematic fallback | flight stack |
| RF quality map | `r2c/rf.py` SINR-grid shadow | measured RSSI/SNR |
| Jammer position | world truth into the RF shadow only | n/a (belief is estimated) |
| Controller belief | estimated from sensed powers (anti-cheat) | same — from radio measurements |
| Failures | `inject_failures` masks | landed/crashed detection -> `set_active` |
