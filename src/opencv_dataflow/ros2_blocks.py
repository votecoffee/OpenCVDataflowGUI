"""Optional ROS 2 Lyrical/Fast DDS endpoints and portable image converters."""

import copy
import re

from .core import Block, Port, Result
from .geometry3d_common import cloud_graphics
from .parameter_validation import NumericRange, ValidationError
from .payloads import NO_VALUE, frame
from .ros2_calibration import ROS2_CALIBRATION_BLOCKS
from .ros2_client import ROSClient, ROSPublishUnknown
from .ros2_clouds import decode_cloud, encode_cloud
from .ros2_images import decode_image, encode_image
from .ros2_support.cloud_wire import ROSPointCloud
from .ros2_support.wire import ENCODINGS, MAX_DATA_BYTES, ROSImage, json_bytes, parse_json
from .tickets import DIRECT_CLOCK
from .visual_style import INDUSTRIAL

MESSAGE_TYPES = (
    "std_msgs/msg/String",
    "std_msgs/msg/Bool",
    "std_msgs/msg/Int32",
    "std_msgs/msg/Int64",
    "std_msgs/msg/Float32",
    "std_msgs/msg/Float64",
    "std_msgs/msg/Float64MultiArray",
    "geometry_msgs/msg/PoseStamped",
    "geometry_msgs/msg/Twist",
    "geometry_msgs/msg/TransformStamped",
    "geometry_msgs/msg/PoseArray",
    "sensor_msgs/msg/JointState",
    "sensor_msgs/msg/CameraInfo",
    "nav_msgs/msg/Odometry",
    "rosgraph_msgs/msg/Clock",
    "custom",
)
TYPE_PATTERN = r"[a-z][a-z0-9_]*/msg/[A-Za-z][A-Za-z0-9_]*"
NAME_PATTERN = r"/?[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z_][A-Za-z0-9_]*)*"


class ROSBlock(Block):
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    preview_safe = False
    requires_serial_execution = True
    validation_skip_output = True
    defaults = {
        "topic": "/opencv_dataflow/demo",
        "namespace": "/opencv_dataflow",
        "domain_id": 71,
        "discovery_range": "localhost",
        "queue_depth": 4,
        "queue_bytes": 33554432,
        "reliability": "reliable",
        "durability": "volatile",
        "deadline_seconds": 0.0,
        "lifespan_seconds": 0.0,
        "timeout_seconds": 0.25,
        "startup_timeout_seconds": 30.0,
        "use_sim_time": False,
        "publication_mode": "asynchronous",
        "consume": "fifo",
        "max_age_seconds": 0.0,
    }
    parameter_choices = {
        "discovery_range": ("localhost", "subnet"),
        "reliability": ("reliable", "best_effort"),
        "durability": ("volatile", "transient_local"),
        "consume": ("fifo", "latest"),
        "publication_mode": ("asynchronous", "synchronous", "auto"),
    }
    parameter_ranges = {
        "domain_id": NumericRange(0, 232, whole=True),
        "queue_depth": NumericRange(1, 64, whole=True),
        "queue_bytes": NumericRange(1024, 67108864, whole=True),
        "timeout_seconds": NumericRange(0.01, 30),
        "startup_timeout_seconds": NumericRange(1, 60),
        **{
            key: NumericRange(0, 3600)
            for key in ("deadline_seconds", "lifespan_seconds", "max_age_seconds")
        },
    }
    parameter_help = {
        "topic": (
            "ROS topic. Endpoints must match domain, topic, installed interface and compatible QoS."
        ),
        "namespace": "Node namespace; relative topics resolve here. / is allowed.",
        "domain_id": (
            "Independent DDS domain. Demo default 71; choose the same domain as your peer."
        ),
        "discovery_range": (
            "Localhost by default; subnet opts in to network discovery. DDS XML may "
            "also affect scope."
        ),
        "queue_depth": (
            "KEEP_LAST DDS depth and maximum buffered received messages; oldest entries "
            "are dropped."
        ),
        "queue_bytes": (
            "Local subscription buffer budget, including encoded image data; oldest "
            "entries are dropped."
        ),
        "reliability": (
            "Best effort suits cameras. Reliable adds DDS delivery acknowledgments, not "
            "application acceptance."
        ),
        "durability": "Transient local can replay cached messages while a publisher remains alive.",
        "deadline_seconds": (
            "Requested/offered DDS deadline; zero uses the middleware's infinite default."
        ),
        "lifespan_seconds": "DDS sample lifespan; zero uses the middleware's infinite default.",
        "timeout_seconds": "Receive wait, or each discovery/acknowledgment wait for a publication.",
        "startup_timeout_seconds": (
            "Bounded worker startup. ROS runs in its own configured Python environment."
        ),
        "use_sim_time": (
            "Use /clock for this node's clock. Does not reinterpret incoming header timestamps."
        ),
        "publication_mode": (
            "Fast DDS publishing mode. Asynchronous bounds normal publication latency."
        ),
        "consume": "FIFO consumes oldest; latest discards superseded buffered samples.",
        "max_age_seconds": (
            "Reject buffered messages older than this local arrival age; zero disables it."
        ),
    }
    role, kind = "read", "data"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = self._settings = None

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for field in ("topic", "namespace"):
            value = p[field]
            if (
                not isinstance(value, str)
                or len(value) > 240
                or "__" in value
                or not (field == "namespace" and value == "/")
                and not re.fullmatch(NAME_PATTERN, value)
            ):
                raise ValidationError("Use a ROS name with slash-separated identifiers", field)
        for field in (
            "enabled",
            "write_enabled",
            "use_sim_time",
            "require_subscriber",
            "wait_for_ack",
        ):
            if field in p and type(p[field]) is not bool:
                raise ValidationError(f"{field} must be boolean", field)
        if "message_type" in p and p["message_type"] == "custom":
            if not isinstance(p["custom_message_type"], str) or not re.fullmatch(
                TYPE_PATTERN, p["custom_message_type"]
            ):
                raise ValidationError("Use an installed package/msg/Type", "custom_message_type")
        if (
            "data_mode" in p
            and p["data_mode"] == "json_string"
            and self.message_type(p) != "std_msgs/msg/String"
        ):
            raise ValidationError(
                "JSON string mode requires std_msgs/msg/String", "data_mode", "message_type"
            )
        if "ticket_field" in p:
            if not isinstance(p["ticket_field"], str) or (
                p["ticket_field"]
                and not re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", p["ticket_field"]
                )
            ):
                raise ValidationError(
                    "Ticket field must be a dotted object field path", "ticket_field"
                )
            if p["ticket_mode"] == "field" and not p["ticket_field"]:
                raise ValidationError("Choose a ticket field", "ticket_field")
        if "frame_id" in p and (
            not isinstance(p["frame_id"], str) or "\0" in p["frame_id"] or len(p["frame_id"]) > 1024
        ):
            raise ValidationError("Frame ID must be text up to 1024 characters", "frame_id")

    def message_type(self, p=None):
        p = self.parameters if p is None else p
        if self.kind == "image":
            return "sensor_msgs/msg/Image"
        if self.kind == "cloud":
            return "sensor_msgs/msg/PointCloud2"
        return p["custom_message_type"] if p["message_type"] == "custom" else p["message_type"]

    def connection(self):
        p = self.parameters
        if self._client is None or self._settings != p:
            self.close()
            config = {key: p[key] for key in ROSBlock.defaults}
            config.update(role=self.role, kind=self.kind, message_type=self.message_type())
            self._client = ROSClient(config, p["startup_timeout_seconds"], self._cancelled)
            self._settings = copy.deepcopy(p)
        return self._client

    def close(self):
        client, self._client = self._client, None
        self._settings = None
        if client is not None:
            client.close()

    def reset_runtime(self):
        self.close()

    def waiting(self, message):
        return Result({port.name: NO_VALUE for port in self.outputs}, message=message)


class ReadROS(ROSBlock):
    defaults = {**ROSBlock.defaults, "enabled": False, "ticket_mode": "auto", "ticket_field": ""}
    parameter_choices = {
        **ROSBlock.parameter_choices,
        "ticket_mode": ("auto", "header_stamp", "field"),
    }
    parameter_help = {
        **ROSBlock.parameter_help,
        "enabled": (
            "Explicitly start a subscription. Disabled sources do not create workers or tickets."
        ),
        "ticket_mode": (
            "Auto acquisition ticket; header frame+stamp string; or an external data field."
        ),
        "ticket_field": (
            "Dotted path in decoded data for an external string/number ticket; not a "
            "ROS timestamp conversion."
        ),
    }

    def issue_source_ticket(self):
        return None

    def acquire(self, inputs):
        if not self.parameters["enabled"]:
            self.close()
            return None
        try:
            timeout = self.parameters["timeout_seconds"]
            result = self.connection().request(
                "poll", timeout + 2, self._cancelled, timeout=timeout
            )
        except InterruptedError:
            self._close_safely()
            return None
        except Exception:
            self._close_safely()
            raise
        return result if result["sample"] is not None else None

    def assign_ticket(self, inputs, value, header):
        p = self.parameters
        mode = p["ticket_mode"]
        if mode == "auto":
            self._source_ticket = Block.issue_source_ticket(self)
        else:
            if mode == "header_stamp":
                from .ros2_support.wire import header_value

                header = header_value(header)
                stamp = header["stamp"]
                value = f"{header['frame_id']}|{stamp['sec']}:{stamp['nanosec']:09d}"
            else:
                try:
                    for field in p["ticket_field"].split("."):
                        value = value[field]
                except (KeyError, TypeError) as exc:
                    raise ValidationError(
                        "ROS data lacks its ticket field", "ticket_field"
                    ) from exc
                if type(value) not in (str, int, float):
                    raise ValidationError("Ticket field must be a string or number", "ticket_field")
            self._source_ticket = getattr(self, "_clock", DIRECT_CLOCK).issue(
                self.id, "external", value
            )
        self._report_activity_start(inputs)
        self.report_event("ticket_assigned", values={"ticket": self._source_ticket})


class ReadROSImage(ReadROS):
    type_id, title = "input.ros2_image", "Read ROS 2 image"
    kind = "image"
    outputs = (Port("image"), Port("message", "data"), Port("trigger", "boolean"))
    flow_primary = ("image",)
    defaults = {
        **ReadROS.defaults,
        "topic": "/opencv_dataflow/camera/image",
        "consume": "latest",
        "reliability": "best_effort",
    }
    parameter_choices = {**ReadROS.parameter_choices, "ticket_mode": ("auto", "header_stamp")}

    def process(self, inputs):
        received = self.acquire(inputs)
        if received is None:
            return self.waiting("ROS image disabled or waiting for a message")
        packet = ROSImage.from_wire(received["sample"])
        image = decode_image(packet)
        image.metadata["spatial"] = frame([packet.width, packet.height], f"{self.id}:image")
        image.metadata["ros2"].update(received["info"])
        self.assign_ticket(inputs, None, packet.header)
        return Result({"image": image, "message": received["info"], "trigger": True})


class ReadROSData(ReadROS):
    type_id, title = "input.ros2_data", "Read ROS 2 data"
    outputs = (Port("data", "any"), Port("message", "data"), Port("trigger", "boolean"))
    flow_primary = ("data",)
    defaults = {
        **ReadROS.defaults,
        "message_type": MESSAGE_TYPES[0],
        "custom_message_type": "",
        "data_mode": "fields",
    }
    parameter_choices = {
        **ReadROS.parameter_choices,
        "message_type": MESSAGE_TYPES,
        "data_mode": ("fields", "json_string"),
    }
    parameter_help = {
        **ReadROS.parameter_help,
        "message_type": (
            "Installed ROS interface; custom interfaces must be installed in the ROS runtime."
        ),
        "custom_message_type": "Installed package/msg/Type when Custom is selected.",
        "data_mode": (
            "Fields returns the ROS message object; JSON string parses String.data as strict JSON."
        ),
    }

    def process(self, inputs):
        received = self.acquire(inputs)
        if received is None:
            return self.waiting("ROS data disabled or waiting for a message")
        fields = received["sample"]
        value = fields
        if self.parameters["data_mode"] == "json_string":
            raw = fields["data"].encode("utf-8")
            if len(raw) > MAX_DATA_BYTES:
                raise ValidationError("ROS JSON string exceeds 1 MiB")
            value = parse_json(raw)
        self.assign_ticket(inputs, value, fields.get("header"))
        payload = {
            "ros2": {
                **received["info"],
                **({"header": fields["header"]} if "header" in fields else {}),
            }
        }
        return Result(
            {"data": value, "message": received["info"], "trigger": True},
            payloads={"data": payload},
        )


class PublishROS(ROSBlock):
    role = "write"
    outputs = (Port("published", "boolean"), Port("delivery", "data"))
    flow_primary = ("published",)
    defaults = {
        **ROSBlock.defaults,
        "write_enabled": False,
        "require_subscriber": True,
        "wait_for_ack": True,
    }
    parameter_help = {
        **ROSBlock.parameter_help,
        "write_enabled": (
            "Also wire true Enable for each intended publication; preview cannot publish."
        ),
        "require_subscriber": (
            "Wait for a matching subscriber before publishing; no match emits no output."
        ),
        "wait_for_ack": (
            "For reliable QoS, wait for DDS acknowledgments. This does not acknowledge robot work."
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._uncertain = False

    def reset_runtime(self):
        super().reset_runtime()
        self._uncertain = False

    def copy_runtime_from(self, other):
        self._uncertain = other._uncertain

    def prepare_value(self, inputs):
        raise NotImplementedError

    def process(self, inputs):
        p = self.parameters
        if not p["write_enabled"] or inputs["enable"] is not True or self._cancelled():
            return self.waiting("ROS write disabled")
        if self._uncertain:
            raise ValidationError("Inspect the unconfirmed ROS publish before resetting flow state")
        value = self.prepare_value(inputs)  # Portable validation does not start ROS.
        try:
            client = self.connection()
            timeout = p["timeout_seconds"]
            delivery = client.request(
                "publish",
                timeout * 2 + 3,
                self._cancelled,
                value=value,
                timeout=timeout,
                require_subscriber=p["require_subscriber"],
                wait_for_ack=p["wait_for_ack"],
                stamp_mode=p.get("stamp_mode", "preserve"),
            )
        except ROSPublishUnknown:
            self._uncertain = True
            self._close_safely()
            raise
        except InterruptedError:
            self._close_safely()
            return self.waiting("ROS write stopped before publication")
        if not delivery["published"]:
            return self.waiting("Waiting for a compatible ROS subscriber")
        return Result({"published": True, "delivery": delivery}, message="ROS message published")


class PublishROSImage(PublishROS):
    type_id, title = "industrial.ros2_image_publish", "Publish ROS 2 image"
    kind = "image"
    inputs = (Port("image"), Port("enable", "boolean"))
    defaults = {
        **PublishROS.defaults,
        "topic": "/opencv_dataflow/camera/image",
        "reliability": "best_effort",
        "encoding": "auto",
        "frame_id": "",
        "stamp_mode": "preserve",
    }
    parameter_choices = {
        **PublishROS.parameter_choices,
        "encoding": ("auto", *ENCODINGS),
        "stamp_mode": ("preserve", "zero", "node_clock"),
    }
    parameter_help = {
        **PublishROS.parameter_help,
        "encoding": (
            "Auto preserves native sample depth and tagged color order; explicit "
            "RGB/BGR can swap channels."
        ),
        "frame_id": (
            "Optional header frame override, with no implied physical transform or calibration."
        ),
        "stamp_mode": (
            "Preserve incoming ROS header (otherwise zero), clear it, or stamp using "
            "the ROS node clock."
        ),
    }

    def prepare_value(self, inputs):
        return encode_image(
            inputs["image"],
            **{key: self.parameters[key] for key in ("encoding", "frame_id", "stamp_mode")},
        ).to_wire()


class PublishROSData(PublishROS):
    type_id, title = "industrial.ros2_data_publish", "Publish ROS 2 data"
    inputs = (Port("data", "any"), Port("enable", "boolean"))
    defaults = {
        **PublishROS.defaults,
        "message_type": MESSAGE_TYPES[0],
        "custom_message_type": "",
        "data_mode": "fields",
    }
    parameter_choices = {
        **PublishROS.parameter_choices,
        "message_type": MESSAGE_TYPES,
        "data_mode": ("fields", "json_string"),
    }
    parameter_help = {
        **PublishROS.parameter_help,
        **{
            key: ReadROSData.parameter_help[key]
            for key in ("message_type", "custom_message_type", "data_mode")
        },
    }

    def prepare_value(self, inputs):
        value = inputs["data"]
        raw = json_bytes(value, MAX_DATA_BYTES)
        if self.parameters["data_mode"] == "json_string":
            value = {"data": raw.decode("utf-8")}
        if not isinstance(value, dict):
            raise ValidationError(
                "ROS fields must be a JSON object; use JSON string mode for arbitrary data"
            )
        json_bytes(value, MAX_DATA_BYTES)
        return value


class EncodeROSImage(Block):
    type_id, title = "industrial.ros2_image_encode", "Encode ROS 2 image"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    inputs = (Port("image"),)
    outputs = (Port("message", "any", visual_type="image"), Port("details", "data"))
    defaults = {"encoding": "auto", "frame_id": "", "stamp_mode": "preserve"}
    parameter_choices = {"encoding": ("auto", *ENCODINGS), "stamp_mode": ("preserve", "zero")}
    parameter_help = {key: PublishROSImage.parameter_help[key] for key in defaults}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        if not isinstance(p["frame_id"], str) or len(p["frame_id"]) > 1024 or "\0" in p["frame_id"]:
            raise ValidationError("Frame ID must be text up to 1024 characters", "frame_id")

    def process(self, inputs):
        packet = encode_image(inputs["image"], **self.parameters)
        return Result({"message": packet, "details": packet.describe()})


class DecodeROSImage(Block):
    type_id, title = "industrial.ros2_image_decode", "Decode ROS 2 image"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    inputs = (Port("message", "any", visual_type="image"),)
    outputs = (Port("image"), Port("details", "data"))

    def process(self, inputs):
        packet = inputs["message"]
        return Result({"image": decode_image(packet), "details": packet.describe()})


class CloudDecodeOptions:
    defaults = {
        "units": "m",
        "invalid_points": "drop",
        "colors": "auto",
        "normals": "normalize",
        "empty_frame": "ros_cloud",
    }
    choices = {
        "units": ("m", "mm"),
        "invalid_points": ("drop", "reject"),
        "colors": ("auto", "ignore"),
        "normals": ("normalize", "ignore"),
    }
    help = {
        "units": "Declared incoming XYZ units; no scaling is performed when decoding.",
        "invalid_points": (
            "Drop invalid XYZ/normals or reject the sample. A false dense claim is rejected."
        ),
        "colors": "Decode packed RGB/RGBA or scalar r/g/b bytes; alpha stays in the raw message.",
        "normals": (
            "Normalize complete normal_x/y/z or nx/ny/nz fields, or leave them in the raw message."
        ),
        "empty_frame": (
            "Fallback cloud frame only when ROS header.frame_id is empty; no transform is implied."
        ),
    }

    @staticmethod
    def validate(p):
        value = p["empty_frame"]
        if not isinstance(value, str) or not value or "\0" in value or len(value) > 1024:
            raise ValidationError(
                "Name an empty-header fallback frame (1..1024 characters)", "empty_frame"
            )


class ReadROSCloud(ReadROS):
    type_id, title = "input.ros2_cloud", "Read ROS 2 point cloud"
    kind = "cloud"
    outputs = (
        Port("cloud", "cloud"),
        Port("message", "any", visual_type="cloud"),
        Port("details", "data"),
        Port("trigger", "boolean"),
    )
    flow_primary = ("cloud",)
    defaults = {
        **ReadROS.defaults,
        **CloudDecodeOptions.defaults,
        "topic": "/opencv_dataflow/camera/points",
        "consume": "latest",
        "reliability": "best_effort",
    }
    parameter_choices = {
        **ReadROS.parameter_choices,
        **CloudDecodeOptions.choices,
        "ticket_mode": ("auto", "header_stamp"),
    }
    parameter_help = {**ReadROS.parameter_help, **CloudDecodeOptions.help}

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        CloudDecodeOptions.validate(self.parameters if parameters is None else parameters)

    def process(self, inputs):
        received = self.acquire(inputs)
        if received is None:
            return self.waiting("ROS point cloud disabled or waiting for a message")
        packet = ROSPointCloud.from_wire(received["sample"])
        cloud, details = decode_cloud(
            packet, **{k: self.parameters[k] for k in CloudDecodeOptions.defaults}
        )
        cloud.metadata["ros2"].update(received["info"])
        self.assign_ticket(inputs, None, packet.header)
        cloud.metadata["ticket"] = copy.deepcopy(self._source_ticket)
        packet.metadata = copy.deepcopy(cloud.metadata)
        return Result(
            {
                "cloud": cloud,
                "message": packet,
                "details": {**details, **received["info"]},
                "trigger": True,
            },
            graphics=cloud_graphics(self, cloud),
        )


class EncodeROSCloud(Block):
    type_id, title = "industrial.ros2_cloud_encode", "Encode ROS 2 point cloud"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    inputs = (Port("cloud", "cloud"),)
    outputs = (Port("message", "any", visual_type="cloud"), Port("details", "data"))
    defaults = {"organization": "unorganized", "byte_order": "little", "stamp_mode": "preserve"}
    parameter_choices = {
        "organization": ("unorganized", "source_pixels"),
        "byte_order": ("little", "big"),
        "stamp_mode": ("preserve", "zero"),
    }
    parameter_help = {
        "organization": (
            "Packed rows or retained source pixels with NaN holes. Duplicate pixels are rejected."
        ),
        "byte_order": "Canonical XYZ/normal float32 and packed uint32 RGB field byte order.",
        "stamp_mode": (
            "Keep incoming ROS acquisition stamp (otherwise zero) or clear it; "
            "frame comes from cloud.frame."
        ),
    }
    input_help = {
        "cloud": "Physical m/mm cloud. XYZ is published in meters; unitless clouds are rejected."
    }

    def process(self, inputs):
        cloud = inputs["cloud"]
        cloud.metadata.update(copy.deepcopy(self.input_payloads["cloud"]))
        packet = encode_cloud(cloud, **self.parameters)
        return Result({"message": packet, "details": {**packet.describe(), "units": "m"}})


class DecodeROSCloud(Block):
    type_id, title = "industrial.ros2_cloud_decode", "Decode ROS 2 point cloud"
    category = block_type = INDUSTRIAL
    subtype = "ROS 2"
    inputs = (Port("message", "any", visual_type="cloud"),)
    outputs = (Port("cloud", "cloud"), Port("details", "data"))
    defaults = CloudDecodeOptions.defaults
    parameter_choices = CloudDecodeOptions.choices
    parameter_help = CloudDecodeOptions.help

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        CloudDecodeOptions.validate(self.parameters if parameters is None else parameters)

    def process(self, inputs):
        packet = inputs["message"]
        if not isinstance(packet, ROSPointCloud):
            raise ValidationError("Expected a portable ROSPointCloud packet")
        packet.metadata.update(copy.deepcopy(self.input_payloads["message"]))
        cloud, details = decode_cloud(packet, **self.parameters)
        return Result({"cloud": cloud, "details": details}, graphics=cloud_graphics(self, cloud))


class PublishROSCloud(PublishROS):
    type_id, title = "industrial.ros2_cloud_publish", "Publish ROS 2 point cloud"
    kind = "cloud"
    inputs = (
        Port("cloud", "cloud", required=False, required_when_connected=True),
        Port("message", "any", required=False, visual_type="cloud", required_when_connected=True),
        Port("enable", "boolean"),
    )
    defaults = {
        **PublishROS.defaults,
        **EncodeROSCloud.defaults,
        "topic": "/opencv_dataflow/camera/points",
        "reliability": "best_effort",
        "input_mode": "cloud",
    }
    parameter_choices = {
        **PublishROS.parameter_choices,
        **EncodeROSCloud.parameter_choices,
        "stamp_mode": ("preserve", "zero", "node_clock"),
        "input_mode": ("cloud", "message"),
    }
    parameter_help = {
        **PublishROS.parameter_help,
        **EncodeROSCloud.parameter_help,
        "stamp_mode": PublishROSImage.parameter_help["stamp_mode"],
        "input_mode": (
            "Cloud rebuilds canonical fields in meters. Message sends the exact portable packet; "
            "organization/byte order are retained and the caller owns raw units. "
            "Wire only the selected input."
        ),
    }
    input_help = EncodeROSCloud.input_help

    def prepare_value(self, inputs):
        if self.parameters["input_mode"] == "message":
            packet = inputs.get("message")
            if not isinstance(packet, ROSPointCloud):
                raise ValidationError("Message mode requires a portable ROSPointCloud packet")
            packet = copy.deepcopy(packet)
            if self.parameters["stamp_mode"] == "zero":
                packet.header["stamp"] = {"sec": 0, "nanosec": 0}
            return packet.to_wire()
        if "cloud" not in inputs:
            raise ValidationError("Cloud mode requires a typed cloud input")
        inputs["cloud"].metadata.update(copy.deepcopy(self.input_payloads["cloud"]))
        return encode_cloud(
            inputs["cloud"], **{k: self.parameters[k] for k in EncodeROSCloud.defaults}
        ).to_wire()


ROS2_BLOCKS = (
    ReadROSImage,
    ReadROSData,
    PublishROSImage,
    PublishROSData,
    EncodeROSImage,
    DecodeROSImage,
    ReadROSCloud,
    PublishROSCloud,
    EncodeROSCloud,
    DecodeROSCloud,
    *ROS2_CALIBRATION_BLOCKS,
)
