# One velocity-controlled drone model for Gazebo Sim (gz-sim >= 7).
# The VelocityControl system consumes /model/{name}/cmd_vel (gz Twist) and
# drives the model at that planar velocity — exactly what the bridge publishes.
MODEL_NAME=$1
X=$2
Y=$3

cat <<EOF
<?xml version="1.0"?>
<sdf version="1.9">
  <model name="${MODEL_NAME}">
    <pose>${X} ${Y} 1 0 0 0</pose>
    <plugin filename="gz-sim-velocity-control-system"
            name="gz::sim::systems::VelocityControl">
      <topic>/model/${MODEL_NAME}/cmd_vel</topic>
    </plugin>
    <link name="base">
      <inertial>
        <mass>1.5</mass>
        <inertia><ixx>0.02</ixx><iyy>0.02</iyy><izz>0.04</izz></inertia>
      </inertial>
      <collision name="c">
        <geometry><sphere><radius>0.25</radius></sphere></geometry>
      </collision>
      <visual name="v">
        <geometry><sphere><radius>0.25</radius></sphere></geometry>
        <material><ambient>0.2 0.2 0.8 1</ambient></material>
      </visual>
    </link>
  </model>
</sdf>
EOF
