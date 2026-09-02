"""Lossless, sequential FAST-LIPEDE processing of a ROS 2 bag."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import PointCloud2

from lipede.online_node import LarsInference, decode_xyzi, filter_records


def _default_output(input_bag: Path) -> Path:
    """Return a sibling URI without assuming a particular rosbag storage plugin."""
    name = input_bag.name.rstrip("/")
    return input_bag.parent / f"{name}_lipede"


class OfflineFastLipedeNode(Node):
    """Read, filter, and rewrite a bag without real-time message loss."""

    def __init__(self):
        super().__init__("lipede_offline")
        share = Path(get_package_share_directory("lipede"))
        self.declare_parameter("bag_path", "")
        self.declare_parameter("output_bag_path", "")
        self.declare_parameter("input_topic", "/ouster/points")
        self.declare_parameter("output_topic", "/ouster/points/processed")
        self.declare_parameter("people_topic", "/ouster/points/people")
        self.declare_parameter("config_path", str(share / "config" / "model.yaml"))
        self.declare_parameter("checkpoint_path", str(share / "models" / "model_8rwkv.0pth"))
        self.declare_parameter("device", "cuda")
        self.declare_parameter("intensity_field", "intensity")
        self.declare_parameter("people_class_ids", [16, 17])
        self.declare_parameter("passthrough_on_error", True)
        self.declare_parameter("overwrite_output", False)

        bag_value = self.get_parameter("bag_path").value
        if not bag_value:
            raise ValueError("offline mode requires bag_path:=/path/to/input_bag")
        self.input_bag = Path(bag_value).expanduser().resolve()
        output_value = self.get_parameter("output_bag_path").value
        self.output_bag = (
            Path(output_value).expanduser().resolve()
            if output_value else _default_output(self.input_bag)
        )
        self.input_topic = self.get_parameter("input_topic").value
        self.intensity_field = self.get_parameter("intensity_field").value
        self.people_ids = np.asarray(
            self.get_parameter("people_class_ids").value, dtype=np.int64
        )
        self.passthrough_on_error = self.get_parameter("passthrough_on_error").value
        self.overwrite_output = self.get_parameter("overwrite_output").value

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        self.original_publisher = self.create_publisher(PointCloud2, self.input_topic, qos)
        self.filtered_publisher = self.create_publisher(
            PointCloud2, self.get_parameter("output_topic").value, qos
        )
        self.people_publisher = self.create_publisher(
            PointCloud2, self.get_parameter("people_topic").value, qos
        )
        self.engine = LarsInference(
            self.get_parameter("config_path").value,
            self.get_parameter("checkpoint_path").value,
            self.get_parameter("device").value,
        )
        self.done = False
        # Give RViz time to start and discover the visualization publishers.
        self.start_timer = self.create_timer(1.0, self._run_once)

    def _filter(self, msg: PointCloud2) -> tuple[PointCloud2, PointCloud2, int]:
        xyzi = decode_xyzi(msg, self.intensity_field)
        predictions, inferred_indices = self.engine.predict(xyzi)
        people = np.zeros(msg.height * msg.width, dtype=bool)
        people[inferred_indices[np.isin(predictions, self.people_ids)]] = True
        return filter_records(msg, ~people), filter_records(msg, people), int(people.sum())

    def _run_once(self) -> None:
        self.start_timer.cancel()
        try:
            self._convert_bag()
        except Exception as error:
            self.get_logger().fatal(f"Offline conversion failed: {error}")
            self.exit_code = 1
        else:
            self.exit_code = 0
        self.done = True

    def _convert_bag(self) -> None:
        import rosbag2_py

        if not self.input_bag.exists():
            raise FileNotFoundError(f"input bag does not exist: {self.input_bag}")
        if self.output_bag == self.input_bag:
            raise ValueError("output_bag_path must differ from bag_path")
        if self.output_bag.exists():
            if not self.overwrite_output:
                raise FileExistsError(
                    f"output bag already exists: {self.output_bag}; choose another path or "
                    "set overwrite_output:=true"
                )
            shutil.rmtree(self.output_bag) if self.output_bag.is_dir() else self.output_bag.unlink()

        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(self.input_bag), storage_id=""),
            rosbag2_py.ConverterOptions("", ""),
        )
        topics = reader.get_all_topics_and_types()
        topic_types = {topic.name: topic.type for topic in topics}
        if self.input_topic not in topic_types:
            raise ValueError(f"bag has no topic named {self.input_topic}")
        if topic_types[self.input_topic] != "sensor_msgs/msg/PointCloud2":
            raise TypeError(
                f"{self.input_topic} is {topic_types[self.input_topic]}, not PointCloud2"
            )

        storage_id = reader.get_metadata().storage_identifier
        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=str(self.output_bag), storage_id=storage_id),
            rosbag2_py.ConverterOptions("", ""),
        )
        for topic in topics:
            writer.create_topic(topic)

        clouds = removed = records = failures = 0
        started = time.perf_counter()
        self.get_logger().info(
            f"Converting {self.input_bag} -> {self.output_bag}; replacing {self.input_topic}"
        )
        while reader.has_next():
            topic, serialized, bag_timestamp = reader.read_next()
            records += 1
            if topic == self.input_topic:
                original = deserialize_message(serialized, PointCloud2)
                try:
                    filtered, people, removed_now = self._filter(original)
                except Exception as error:
                    failures += 1
                    if not self.passthrough_on_error:
                        raise
                    self.get_logger().error(
                        f"Cloud at {bag_timestamp} failed, copying unchanged: {error}"
                    )
                    filtered = original
                    people = filter_records(
                        original, np.zeros(original.height * original.width, dtype=bool)
                    )
                    removed_now = 0
                writer.write(topic, serialize_message(filtered), bag_timestamp)
                self.original_publisher.publish(original)
                self.filtered_publisher.publish(filtered)
                self.people_publisher.publish(people)
                clouds += 1
                removed += removed_now
                if clouds % 25 == 0:
                    self.get_logger().info(
                        f"processed clouds={clouds} removed points={removed} records={records}"
                    )
            else:
                # Serialized payload and recorded publication time are copied exactly.
                writer.write(topic, serialized, bag_timestamp)

        elapsed = time.perf_counter() - started
        self.get_logger().info(
            f"Finished {clouds} clouds/{records} records in {elapsed:.1f}s; "
            f"removed {removed} points; failures={failures}; output={self.output_bag}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    exit_code = 1
    try:
        node = OfflineFastLipedeNode()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
        exit_code = getattr(node, "exit_code", 1)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
