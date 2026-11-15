"""Original metric board geometry and wrist camera for the calibration workcell."""

import math
from xml.etree import ElementTree as ET


def sub(parent, tag, value=None, **attrs):
    item = ET.SubElement(parent, tag, attrs)
    if value is not None:
        item.text = " ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
    return item


def visual(link, name, xyz, size, color, collision=False):
    item = sub(link, "visual", name=name)
    sub(item, "pose", [*xyz, 0, 0, 0])
    sub(sub(item, "geometry"), "box").append(
        ET.fromstring(f"<size>{' '.join(map(str, size))}</size>")
    )
    material = sub(item, "material")
    sub(material, "ambient", color)
    sub(material, "diffuse", color)
    sub(material, "specular", "0 0 0 1")
    if collision:
        item = sub(link, "collision", name=name)
        sub(item, "pose", [*xyz, 0, 0, 0])
        sub(sub(sub(item, "geometry"), "box"), "size", size)


def world_xml(scene):
    root = ET.Element("sdf", version="1.12")
    world = sub(root, "world", name="hand_eye")
    sub(world, "gravity", "0 0 -9.81")
    physics = sub(world, "physics", name="physics", type="ignored")
    sub(physics, "max_step_size", 0.001)
    sub(physics, "real_time_factor", 1)
    for library, system in [
        ("physics", "Physics"),
        ("user-commands", "UserCommands"),
        ("scene-broadcaster", "SceneBroadcaster"),
        ("sensors", "Sensors"),
    ]:
        plugin = sub(
            world, "plugin", filename=f"gz-sim-{library}-system", name=f"gz::sim::systems::{system}"
        )
        if system == "Sensors":
            sub(plugin, "render_engine", "ogre2")
    sub(sub(world, "scene"), "ambient", "0.8 0.8 0.8 1")
    light = sub(world, "light", name="light", type="directional")
    sub(light, "pose", "0 0 4 0 0 0")
    sub(light, "direction", "0.2 0.4 -1")
    sub(light, "diffuse", "0.8 0.8 0.8 1")
    sub(light, "cast_shadows", "false")
    ground = sub(world, "model", name="ground")
    sub(ground, "static", "true")
    visual(
        sub(ground, "link", name="ground"),
        "ground",
        [0, 0, -0.025],
        [4, 4, 0.05],
        ".4 .45 .5 1",
        True,
    )
    board = sub(world, "model", name="charuco_board")
    sub(board, "static", "true")
    sub(board, "pose", scene["board_pose_xyzrpy"])
    link = sub(board, "link", name="plate")
    visual(link, "white_backing", [0.105, 0.075, 0.002], [0.25, 0.19, 0.004], "1 1 1 1", True)
    # Metric black rectangles reproduce the generated target's pixel edges exactly.
    for i, (x, y, width, height) in enumerate(scene["black_rectangles_m"]):
        visual(
            link,
            f"black_{i}",
            [x + width / 2, y + height / 2, -0.00006],
            [width, height, 0.0001],
            "0 0 0 1",
        )
    return ET.tostring(root, encoding="unicode")


def add_camera(description, scene):
    root = ET.fromstring(description)
    camera = sub(root, "link", name="hand_eye_camera_link")
    inertial = sub(camera, "inertial")
    ET.SubElement(inertial, "mass", {"value": "0.02"})
    sub(inertial, "inertia", ixx="0.00001", ixy="0", ixz="0", iyy="0.00001", iyz="0", izz="0.00001")
    for kind in ("visual", "collision"):
        box = sub(camera, kind)
        sub(sub(box, "geometry"), "box", size="0.025 0.035 0.035")
    joint = sub(root, "joint", name="hand_eye_camera_mount", type="fixed")
    sub(joint, "parent", link="tool0")
    sub(joint, "child", link="hand_eye_camera_link")
    pose = scene["tool_from_camera_body_xyzrpy"]
    sub(joint, "origin", xyz=" ".join(map(str, pose[:3])), rpy=" ".join(map(str, pose[3:])))
    sub(root, "link", name="hand_eye_camera_optical_frame")
    joint = sub(root, "joint", name="hand_eye_optical_axes", type="fixed")
    sub(joint, "parent", link="hand_eye_camera_link")
    sub(joint, "child", link="hand_eye_camera_optical_frame")
    sub(joint, "origin", xyz="0 0 0", rpy=f"{-math.pi / 2} 0 {-math.pi / 2}")
    gazebo = sub(root, "gazebo", reference="hand_eye_camera_link")
    sensor = sub(gazebo, "sensor", name="hand_eye_camera", type="camera")
    sub(sensor, "always_on", "true")
    sub(sensor, "update_rate", 10)
    sub(sensor, "topic", "/hand_eye/camera")
    sub(sensor, "gz_frame_id", "hand_eye_camera_optical_frame")
    cam = sub(sensor, "camera")
    sub(cam, "horizontal_fov", 2 * math.atan(1280 / 2000))
    image = sub(cam, "image")
    sub(image, "width", 1280)
    sub(image, "height", 960)
    sub(image, "format", "R8G8B8")
    clip = sub(cam, "clip")
    sub(clip, "near", 0.05)
    sub(clip, "far", 3)
    return ET.tostring(root, encoding="unicode")


def bridges():
    rows = []
    for gz, ros, rt, gt in [
        ("/hand_eye/camera", "/hand_eye/camera/image", "sensor_msgs/msg/Image", "Image"),
        (
            "/hand_eye/camera_info",
            "/hand_eye/camera/camera_info",
            "sensor_msgs/msg/CameraInfo",
            "CameraInfo",
        ),
    ]:
        rows.append(
            dict(
                gz_topic_name=gz,
                ros_topic_name=ros,
                ros_type_name=rt,
                gz_type_name="gz.msgs." + gt,
                direction="GZ_TO_ROS",
            )
        )
    return rows
