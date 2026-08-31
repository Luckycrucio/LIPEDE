# FAST-LIPEDE

FAST-LIPEDE (fats LiDAR PEople DEtector) is a ROS 2 Python node that runs one LARS/LSK3DNet inference per
incoming Ouster cloud and removes points classified as people. It subscribes to
`/ouster/points`, publishes the filtered cloud on `/ouster/points/processed`,
and publishes only positively classified people points on
`/ouster/points/people`.

The ROS package name is `fast_lipede` because ROS 2 package names cannot contain
hyphens.

## AUTOSWEEP USAGE

Launch the fast-lipede ros2 node:
- lipede
- ros2 launch fast_lipede fast_lipede.launch.py

Luanch a SLAM algorithm like GLIM:
- glim
- ros2 run glim_ros glim_rosnode \
  --ros-args \
  -p config_path:=/home/autosweep/glim_ws/src/glim/config \
  -r /ouster/points:=/ouster/points/processed

Play the bag:
- jazzy
- ros2 bag play   /home/autosweep/autosweep/dataset22jul/coverage1 



## Data path through the node

For each `sensor_msgs/msg/PointCloud2` message, the node:

1. Creates NumPy views for the `x`, `y`, `z`, and `intensity` fields directly
   over the ROS message buffer. The fields may have arbitrary byte offsets and
   the message may contain additional Ouster fields.
2. Builds the model's contiguous `float32` `[x, y, z, intensity]` array. This is
   the only unavoidable representation conversion before preprocessing; no
   temporary `.bin` file is written.
3. Removes invalid points and applies the training crop (`x/y` within 50 m and
   `z` between -4 and 4 m), normalizes intensity, and computes the same
   range-image normals used by `test_autosweep.py`.
4. Runs exactly one inference (`model.eval()` plus `torch.inference_mode()`).
5. Removes model classes listed in `people_class_ids`. The default is `[16, 17]`,
   corresponding to the person/followers and kid-like learned categories for
   the 21-class AutoSweep model. Confirm these IDs against the label mapping
   used to train your checkpoint.
6. Splits the original point records into two complementary messages:
   `/ouster/points/processed` contains everything except people, while
   `/ouster/points/people` contains only positive people detections.
7. Copies every selected original `point_step`-byte record byte-for-byte. Thus
   timestamps, frame ID, field layout, and all retained Ouster point attributes
   are preserved exactly. Width, row step, and data length necessarily change.
   Both outputs are unorganized (`height=1`) clouds.

Points outside the model crop are retained because the network cannot classify
them. If processing fails, the original message is published by default.

## Included inference files

- `models/model_8rwkv.0pth`: copied LARS checkpoint.
- `config/model.yaml`: minimal architecture and crop configuration.
- `vendor/network`: model, voxelization, RWKV, and TorchSparse compatibility
  implementation required by the checkpoint.
- `vendor/utils`: checkpoint loader and range-normal preprocessing only.
- `vendor/c_utils`: the normal-map C++ source/build definition and the existing
  Python 3.11 compiled extension. A vectorized NumPy fallback is used if its ABI
  does not match the ROS Python interpreter.

The checked-in `.so` is Python/architecture specific. Rebuild it when the target
does not use CPython 3.11 x86-64.

## Requirements

- ROS 2 with `rclpy`, `sensor_msgs`, and `ament_python`
- Python 3.11 for the included fast normal extension, or use/rebuild the fallback
- CUDA-capable GPU and CUDA-enabled PyTorch
- `numpy`, `PyYAML`, OpenCV, `torch-scatter`, and TorchSparse

The model stack is not installed automatically by `colcon`. Install versions
compatible with the CUDA/PyTorch environment used by the original LARS project.

## Build

From a ROS 2 workspace (this repository can be the workspace source directory,
or this folder can be linked under `src`):

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd /path/to/ros2_ws
colcon build --symlink-install --packages-select fast_lipede
source install/setup.bash
```

For a conventional workspace layout:

```bash
mkdir -p ~/fast_lipede_ws/src
ln -s /path/to/LARS/FAST-LIPEDE ~/fast_lipede_ws/src/FAST-LIPEDE
cd ~/fast_lipede_ws
colcon build --symlink-install --packages-select fast_lipede
source install/setup.bash
```

## Run the node only

```bash
ros2 run fast_lipede fast_lipede_node
```

## Launch the node and RViz

```bash
ros2 launch fast_lipede fast_lipede.launch.py
```

The included RViz YAML configuration displays:

- `/ouster/points` in translucent gray;
- `/ouster/points/people` in yellow with larger points;
- `/ouster/points/processed` in blue, initially disabled to avoid visually
  covering the gray input. Enable it in the RViz Displays panel when needed.

The default RViz fixed frame is `os_sensor`. If the incoming cloud uses another
`header.frame_id`, change **Global Options → Fixed Frame** and the view's target
frame in RViz, then save the configuration if desired.

Launch arguments can override the node topics and device:

```bash
ros2 launch fast_lipede fast_lipede.launch.py \
  input_topic:=/ouster/points \
  output_topic:=/ouster/points/processed \
  people_topic:=/ouster/points/people \
  device:=cuda
```

The launch file also applies these topic overrides as RViz remappings.

Useful parameters:

```bash
ros2 run fast_lipede fast_lipede_node --ros-args \
  -p input_topic:=/ouster/points \
  -p output_topic:=/ouster/points/processed \
  -p people_topic:=/ouster/points/people \
  -p intensity_field:=intensity \
  -p people_class_ids:="[16,17]" \
  -p device:=cuda
```

If the Ouster driver exposes reflectivity under another field name, set
`intensity_field`, for example `-p intensity_field:=reflectivity`. The numerical
distribution must match the intensity feature used during training for good
predictions.

Other parameters:

| Parameter | Default | Purpose |
|---|---|---|
| `config_path` | installed `config/model.yaml` | Model architecture and crop |
| `checkpoint_path` | installed checkpoint | Override model weights |
| `passthrough_on_error` | `true` | Publish the unchanged input after an error |

## Validate the stream

```bash
ros2 topic hz /ouster/points/processed
ros2 topic hz /ouster/points/people
ros2 topic echo /ouster/points/processed --once
```

The node uses sensor-data-style best-effort QoS with queue depth 1. This favors
fresh scans and bounds backlog during real-time operation. ROS Python callbacks
are synchronous, so an incoming scan may be dropped by DDS while inference is
busy rather than accumulating latency.

## Optional `.bin` representation

The model input is equivalent to a SemanticKITTI-style `.bin` file:

```python
xyzi.astype(np.float32).tofile("scan.bin")
```

FAST-LIPEDE deliberately does not perform this disk write. It passes the same
`N x 4` values directly to preprocessing, avoiding filesystem latency and a
second read/copy.

## Performance notes

- Keep `people_class_ids` short; filtering uses one vectorized comparison.
- The first scan is slower due to CUDA initialization and allocator warm-up.
- Rebuild the normal-map extension for maximum throughput if the node reports a
  Python ABI mismatch; the portable vectorized fallback is slower.
- Latency is logged at most once per second.
- Input fields are viewed without conversion. A compact output buffer must be
  allocated because deleting arbitrary point records cannot be represented as a
  zero-copy `PointCloud2` message.
