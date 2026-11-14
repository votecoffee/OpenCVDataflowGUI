"""Pure scene generation. The app owns pick order, target grid and perception."""
import hashlib
import json
import math
from pathlib import Path
import re
from xml.etree import ElementTree as ET


def default_scene():
    parts = []
    for stack, y in enumerate((-0.12, 0.0, 0.12)):
        for level in range(2):
            parts.append({"id": f"part_{2 * stack + level + 1:02d}", "size": [0.04] * 3,
                          "pose": [0.45, y, 0.2205 + level * 0.041, 0, 0, 0], "mass": 0.05})
    return {"schema": "opencv.bin_picking_scene/v1", "units": "m", "world": "opencv_bin_picking",
            "world_frame": "world", "robot_model": "ur", "tcp": "gripper_tcp", "group": "ur_manipulator",
            "static_boxes": [
                {"id": "table_top", "size": [0.50, 0.94, 0.03], "pose": [0.51, 0.15, 0.185, 0, 0, 0]},
                {"id": "bin_floor", "size": [0.24, 0.48, 0.01], "pose": [0.45, 0, 0.195, 0, 0, 0]},
                {"id": "bin_x_min", "size": [0.01, 0.48, 0.10], "pose": [0.325, 0, 0.25, 0, 0, 0]},
                {"id": "bin_x_max", "size": [0.01, 0.48, 0.10], "pose": [0.575, 0, 0.25, 0, 0, 0]},
                {"id": "bin_y_min", "size": [0.24, 0.01, 0.10], "pose": [0.45, -0.245, 0.25, 0, 0, 0]},
                {"id": "bin_y_max", "size": [0.24, 0.01, 0.10], "pose": [0.45, 0.245, 0.25, 0, 0, 0]}],
            "parts": parts,
            "camera": {"pose": [0.45, 0, 1.25, 0, math.pi / 2, 0], "width": 960, "height": 720,
                       "horizontal_fov": 1.2, "near": 0.05, "far": 3.0, "rate_hz": 15,
                       "optical_frame": "opencv_fixture_camera_optical_frame"}}


def validate(scene):
    if scene.get("schema") != "opencv.bin_picking_scene/v1" or scene.get("units") != "m":
        raise ValueError("Expected a metre-based opencv.bin_picking_scene/v1 scene")
    names = set()
    for name in [scene["world"], scene["world_frame"], scene["robot_model"], scene["tcp"], scene["group"],
                 scene["camera"]["optical_frame"]]:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
            raise ValueError("Invalid scene/frame identifier")
    if not 1 <= len(scene["parts"]) <= 64:
        raise ValueError("Fixture supports 1..64 individually named parts")
    for item in scene["static_boxes"] + scene["parts"]:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", item["id"]) or item["id"] in names:
            raise ValueError("Invalid or duplicate object ID")
        names.add(item["id"])
        if len(item["size"]) != 3 or len(item["pose"]) != 6:
            raise ValueError("Boxes need three dimensions and xyz/rpy poses")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   for v in item["size"] + item["pose"]):
            raise ValueError("Scene geometry must be finite numeric values")
        if not all(0 < v <= 5 for v in item["size"]):
            raise ValueError("Box dimensions must be positive and at most 5 m")
        if item in scene["parts"] and not 0 < item["mass"] <= 10:
            raise ValueError("Part mass must be positive and at most 10 kg")
    camera = scene["camera"]
    if len(camera["pose"]) != 6 or not all(math.isfinite(v) for v in camera["pose"]):
        raise ValueError("Camera pose must be finite xyz/rpy")
    if not (16 <= camera["width"] <= 1920 and 16 <= camera["height"] <= 1080
            and 0 < camera["horizontal_fov"] < math.pi and 0 < camera["near"] < camera["far"] <= 20
            and 0 < camera["rate_hz"] <= 60):
        raise ValueError("Invalid camera geometry/rate")
    return scene


def digest(scene):
    return hashlib.sha256(json.dumps(validate(scene), sort_keys=True, allow_nan=False).encode()).hexdigest()


def child(parent, tag, value=None, **attributes):
    element = ET.SubElement(parent, tag, attributes)
    if value is not None:
        element.text = " ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
    return element


def plugin(parent, library, system, **settings):
    element = child(parent, "plugin", filename=f"gz-sim-{library}-system", name=f"gz::sim::systems::{system}")
    for name, value in settings.items():
        child(element, name, value)
    return element


def box(world, item, static):
    model = child(world, "model", name=item["id"])
    child(model, "static", "true" if static else "false")
    child(model, "pose", item["pose"])
    link = child(model, "link", name="body")
    if not static:
        inertial = child(link, "inertial"); mass = item["mass"]
        child(inertial, "mass", mass)
        inertia = child(inertial, "inertia"); x, y, z = item["size"]
        for name, value in {"ixx": mass * (y*y + z*z)/12, "iyy": mass * (x*x + z*z)/12,
                            "izz": mass * (x*x + y*y)/12, "ixy": 0, "ixz": 0, "iyz": 0}.items():
            child(inertia, name, value)
    for kind in ("collision", "visual"):
        element = child(link, kind, name=kind)
        child(child(child(element, "geometry"), "box"), "size", item["size"])
        if kind == "visual":
            material = child(element, "material")
            color = "0.32 0.36 0.4 1" if static else "0.85 0.36 0.12 1"
            child(material, "ambient", color); child(material, "diffuse", color)
    if not static:
        plugin(model, "pose-publisher", "PosePublisher", publish_model_pose="true", publish_link_pose="false",
               use_pose_vector_msg="true", update_frequency=30, topic=f"/opencv_fixture/evaluation/{item['id']}/pose")


def world_xml(scene):
    validate(scene)
    sdf = ET.Element("sdf", version="1.12"); world = child(sdf, "world", name=scene["world"])
    child(world, "gravity", "0 0 -9.81")
    physics = child(world, "physics", name="fixture_physics", type="ignored")
    child(physics, "max_step_size", 0.001); child(physics, "real_time_factor", 1)
    plugin(world, "physics", "Physics"); plugin(world, "user-commands", "UserCommands")
    plugin(world, "scene-broadcaster", "SceneBroadcaster")
    plugin(world, "sensors", "Sensors", render_engine="ogre2")
    light = child(world, "light", name="fixture_sun", type="directional")
    child(light, "pose", "0 0 5 0 0 0"); child(light, "direction", "0.2 0.1 -1")
    child(light, "diffuse", "0.9 0.9 0.9 1"); child(light, "specular", "0.1 0.1 0.1 1")
    attenuation = child(light, "attenuation"); child(attenuation, "range", 100)
    floor = {"id": "ground", "size": [4, 4, 0.05], "pose": [0, 0, -0.025, 0, 0, 0]}
    box(world, floor, True)
    for item in scene["static_boxes"]: box(world, item, True)
    for item in scene["parts"]: box(world, item, False)
    camera = scene["camera"]
    model = child(world, "model", name="fixture_camera"); child(model, "static", "true")
    child(model, "pose", camera["pose"]); link = child(model, "link", name="camera_link")
    sensor = child(link, "sensor", name="rgbd", type="rgbd_camera")
    child(sensor, "frame_id", camera["optical_frame"])
    child(sensor, "always_on", "true"); child(sensor, "update_rate", camera["rate_hz"])
    child(sensor, "topic", "/opencv_fixture/source/camera")
    settings = child(sensor, "camera"); child(settings, "horizontal_fov", camera["horizontal_fov"])
    image = child(settings, "image"); child(image, "width", camera["width"])
    child(image, "height", camera["height"]); child(image, "format", "R8G8B8")
    clip = child(settings, "clip"); child(clip, "near", camera["near"]); child(clip, "far", camera["far"])
    ET.indent(sdf)
    return ET.tostring(sdf, encoding="unicode", xml_declaration=True) + "\n"


def add_assistance(urdf, scene):
    """Add unconnected joints to this generated robot copy, preserving original bytes elsewhere."""
    validate(scene); root = ET.fromstring(urdf)
    if root.tag != "robot" or root.find("link[@name='gripper_base_link']") is None:
        raise ValueError("Expected the native generated UR5e parallel-gripper URDF")
    gazebo = child(root, "gazebo")
    child(child(root, "gazebo", reference="gripper_mount_base"), "preserveFixedJoint", "true")
    for part in scene["parts"]:
        prefix = f"/opencv_fixture/assistance/{part['id']}"
        plugin(gazebo, "detachable-joint", "DetachableJoint", parent_link="gripper_base_link",
               child_model=part["id"], child_link="body", initial_attach="false",
               attach_topic=prefix + "/attach", detach_topic=prefix + "/detach", output_topic=prefix + "/state")
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def bridge_config(scene):
    def row(ros, gz, ros_type, gz_type, direction="GZ_TO_ROS"):
        return {"ros_topic_name": ros, "gz_topic_name": gz, "ros_type_name": ros_type,
                "gz_type_name": "gz.msgs." + gz_type, "direction": direction}
    rows = [row("/opencv_fixture/source/rgb", "/opencv_fixture/source/camera/image", "sensor_msgs/msg/Image", "Image"),
            row("/opencv_fixture/source/depth", "/opencv_fixture/source/camera/depth_image", "sensor_msgs/msg/Image", "Image"),
            row("/opencv_fixture/source/camera_info", "/opencv_fixture/source/camera/camera_info", "sensor_msgs/msg/CameraInfo", "CameraInfo")]
    for part in scene["parts"]:
        prefix = f"/opencv_fixture/assistance/{part['id']}"
        for command in ("attach", "detach"):
            rows.append(row(prefix + "/" + command, prefix + "/" + command, "std_msgs/msg/Empty", "Empty", "ROS_TO_GZ"))
        rows.append(row(prefix + "/state", prefix + "/state", "std_msgs/msg/String", "StringMsg"))
        pose = f"/opencv_fixture/evaluation/{part['id']}/pose"
        rows.append(row(pose, pose, "tf2_msgs/msg/TFMessage", "Pose_V"))
    return rows


def load(path):
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))


def quaternion_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return (sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy)


def camera_transform(scene):
    """World-from-optical; Gazebo camera link is +X forward, +Y left, +Z up."""
    a = quaternion_rpy(*scene["camera"]["pose"][3:])
    b = quaternion_rpy(-math.pi/2, 0, -math.pi/2)
    x, y, z, w = a; X, Y, Z, W = b
    q = [w*X + x*W + y*Z - z*Y, w*Y - x*Z + y*W + z*X,
         w*Z + x*Y - y*X + z*W, w*W - x*X - y*Y - z*Z]
    return {"parent": scene["world_frame"], "child": scene["camera"]["optical_frame"],
            "translation_m": list(scene["camera"]["pose"][:3]), "quaternion_xyzw": q,
            "origin": "scene-defined fixed camera extrinsics; not estimated calibration"}
