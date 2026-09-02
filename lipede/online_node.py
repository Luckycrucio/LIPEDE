"""ROS 2 PointCloud2 semantic people-removal node."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField


_ROS_DTYPES = {
    PointField.INT8: "i1",
    PointField.UINT8: "u1",
    PointField.INT16: "i2",
    PointField.UINT16: "u2",
    PointField.INT32: "i4",
    PointField.UINT32: "u4",
    PointField.FLOAT32: "f4",
    PointField.FLOAT64: "f8",
}


def _field_map(msg: PointCloud2) -> Dict[str, PointField]:
    return {field.name: field for field in msg.fields}


def _numeric_field(msg: PointCloud2, name: str) -> np.ndarray:
    """Return a zero-copy, row-major view of one scalar PointCloud2 field."""
    fields = _field_map(msg)
    if name not in fields:
        raise ValueError(f"PointCloud2 has no '{name}' field")
    field = fields[name]
    if field.count != 1 or field.datatype not in _ROS_DTYPES:
        raise ValueError(f"Field '{name}' must be one supported scalar numeric value")
    endian = ">" if msg.is_bigendian else "<"
    dtype = np.dtype(endian + _ROS_DTYPES[field.datatype])
    values = np.ndarray(
        shape=(msg.height, msg.width), dtype=dtype, buffer=msg.data,
        offset=field.offset, strides=(msg.row_step, msg.point_step),
    )
    return values.reshape(-1) if msg.row_step == msg.width * msg.point_step else values.ravel()


def decode_xyzi(msg: PointCloud2, intensity_field: str) -> np.ndarray:
    """Decode x/y/z/intensity into the float32 Nx4 representation used by LARS."""
    columns = [_numeric_field(msg, name) for name in ("x", "y", "z", intensity_field)]
    return np.column_stack(columns).astype(np.float32, copy=False)


def filter_records(msg: PointCloud2, keep: np.ndarray) -> PointCloud2:
    """Copy retained point records byte-for-byte into a compact unorganized cloud."""
    count = msg.height * msg.width
    if keep.shape != (count,):
        raise ValueError(f"keep mask has {keep.size} entries, expected {count}")
    rows = np.ndarray(
        shape=(msg.height, msg.width, msg.point_step), dtype=np.uint8,
        buffer=msg.data, strides=(msg.row_step, msg.point_step, 1),
    ).reshape(count, msg.point_step) if msg.row_step == msg.width * msg.point_step else np.concatenate([
        np.frombuffer(msg.data, dtype=np.uint8, count=msg.width * msg.point_step,
                      offset=row * msg.row_step).reshape(msg.width, msg.point_step)
        for row in range(msg.height)
    ])
    kept = np.ascontiguousarray(rows[keep])
    output = PointCloud2()
    output.header = copy.deepcopy(msg.header)
    output.height = 1
    output.width = int(kept.shape[0])
    output.fields = copy.deepcopy(msg.fields)
    output.is_bigendian = msg.is_bigendian
    output.point_step = msg.point_step
    output.row_step = output.width * output.point_step
    output.data = kept.tobytes()
    output.is_dense = msg.is_dense
    return output


class LarsInference:
    def __init__(self, config_path: str, checkpoint_path: str, device: str):
        import torch
        import yaml

        installed_vendor = Path(get_package_share_directory("lipede")) / "vendor"
        source_vendor = Path(__file__).resolve().parents[1] / "vendor"
        vendor = installed_vendor if installed_vendor.exists() else source_vendor
        normal_build = vendor / "c_utils" / "build"
        sys.path.insert(0, str(vendor))
        sys.path.insert(0, str(normal_build))
        from network.largekernel_model import get_model_class
        from utils.load_save_util import load_checkpoint_compatible
        from utils.normalmap import USING_NATIVE_NORMALS, compute_normals_range

        with open(config_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested, but CUDA is unavailable")
        self.device = torch.device(device)
        self.torch = torch
        self.compute_normals = compute_normals_range
        self.using_native_normals = USING_NATIVE_NORMALS
        architecture = self.config["model_params"]["model_architecture"]
        self.model = get_model_class(architecture)(self.config)
        self.model = load_checkpoint_compatible(checkpoint_path, self.model, self.device)
        self.model.to(self.device).eval()
        bounds = self.config["dataset_params"]
        self.minimum = np.asarray(bounds["min_volume_space"], dtype=np.float32)
        self.maximum = np.asarray(bounds["max_volume_space"], dtype=np.float32)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def predict(self, xyzi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return predictions and original indices for finite, in-range points."""
        torch = self.torch
        finite = np.isfinite(xyzi).all(axis=1)
        nonzero_range = np.linalg.norm(xyzi[:, :3], axis=1) > 1e-6
        in_range = ((xyzi[:, :3] > self.minimum) & (xyzi[:, :3] < self.maximum)).all(axis=1)
        indices = np.flatnonzero(finite & nonzero_range & in_range)
        if indices.size == 0:
            return np.empty(0, dtype=np.int64), indices

        points = np.ascontiguousarray(xyzi[indices], dtype=np.float32)
        intensity_max = float(np.max(np.abs(points[:, 3])))
        if intensity_max > 0.0:
            points[:, 3] /= intensity_max
        normals = np.ascontiguousarray(self.compute_normals(points), dtype=np.float32)
        data = {
            "points": torch.from_numpy(points).to(self.device, non_blocking=True),
            "normal": torch.from_numpy(normals).to(self.device, non_blocking=True),
            "batch_idx": torch.zeros(indices.size, dtype=torch.long, device=self.device),
            "batch_size": 1,
        }
        with torch.inference_mode():
            logits = self.model(data)["logits"]
            prediction = logits.argmax(dim=1).cpu().numpy()
        return prediction, indices


class FastLipedeNode(Node):
    def __init__(self):
        super().__init__("lipede")
        share = Path(get_package_share_directory("lipede"))
        self.declare_parameter("input_topic", "/ouster/points")
        self.declare_parameter("output_topic", "/ouster/points/processed")
        self.declare_parameter("people_topic", "/ouster/points/people")
        self.declare_parameter("config_path", str(share / "config" / "model.yaml"))
        self.declare_parameter("checkpoint_path", str(share / "models" / "model_8rwkv.0pth"))
        self.declare_parameter("device", "cuda")
        self.declare_parameter("intensity_field", "intensity")
        self.declare_parameter("people_class_ids", [16, 17])
        self.declare_parameter("passthrough_on_error", True)

        self.intensity_field = self.get_parameter("intensity_field").value
        self.people_ids = np.asarray(self.get_parameter("people_class_ids").value, dtype=np.int64)
        self.passthrough_on_error = self.get_parameter("passthrough_on_error").value
        self.engine = LarsInference(
            self.get_parameter("config_path").value,
            self.get_parameter("checkpoint_path").value,
            self.get_parameter("device").value,
        )
        if not self.engine.using_native_normals:
            self.get_logger().warning(
                "Native normal-map extension is ABI-incompatible or missing; using NumPy fallback"
            )
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        output_topic = self.get_parameter("output_topic").value
        people_topic = self.get_parameter("people_topic").value
        input_topic = self.get_parameter("input_topic").value
        self.publisher = self.create_publisher(PointCloud2, output_topic, qos)
        self.people_publisher = self.create_publisher(PointCloud2, people_topic, qos)
        self.subscription = self.create_subscription(PointCloud2, input_topic, self._callback, qos)
        self.get_logger().info(
            f"Filtering {input_topic} -> {output_topic}; detections -> {people_topic}; "
            f"people IDs={self.people_ids.tolist()}"
        )

    def _callback(self, msg: PointCloud2) -> None:
        started = time.perf_counter()
        try:
            xyzi = decode_xyzi(msg, self.intensity_field)
            predictions, inferred_indices = self.engine.predict(xyzi)
            people = np.zeros(msg.height * msg.width, dtype=bool)
            people[inferred_indices[np.isin(predictions, self.people_ids)]] = True
            self.publisher.publish(filter_records(msg, ~people))
            self.people_publisher.publish(filter_records(msg, people))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.get_logger().info(
                f"points={people.size} removed={int(people.sum())} latency={elapsed_ms:.1f} ms",
                throttle_duration_sec=1.0,
            )
        except Exception as error:
            self.get_logger().error(f"Point cloud processing failed: {error}")
            if self.passthrough_on_error:
                self.publisher.publish(msg)
                self.people_publisher.publish(
                    filter_records(msg, np.zeros(msg.height * msg.width, dtype=bool))
                )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FastLipedeNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
