"""Derive assets/g1/g1_walk.xml from the vendored upstream assets/g1/g1.xml.

The only change: the 12 lower-body leg joint actuators are converted from
MuJoCo <position> actuators (upstream default: kp=500 dampratio=1, one
gain shared by every joint) to <motor> (direct torque) actuators, so our
own PD controller (control/pd_controller.py) can apply distinct
kp/kd per leg-joint group instead of MuJoCo's built-in uniform gain.

ctrlrange/forcerange on each new motor actuator equals that joint's own
`actuatorfrcrange` from g1.xml -- no torque limit is changed, only the
control mode. Waist and arm actuators are left untouched (<position>,
upstream gains): they are held at a fixed pose and are not RL-controlled
in this project (see docs/DECISIONS.md ADR-002), so MuJoCo's internal
position PD is sufficient for them.

This mirrors the same transform already applied and validated in the
sibling ../standing-rl project (assets/g1/README.md there documents the
identical technique for the identical robot); we re-derive it ourselves
here rather than copying the derived file, since owning the transform is
part of understanding the model (CLAUDE.md "own the task-specific parts").

Run: python scripts/build_walk_model.py
Regenerate whenever assets/g1/g1.xml (upstream) changes.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "assets" / "g1" / "g1.xml"
DST = REPO_ROOT / "assets" / "g1" / "g1_walk.xml"

LEG_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


def find_joint_frcrange(root: ET.Element, joint_name: str) -> tuple[str, str]:
    for joint in root.iter("joint"):
        if joint.get("name") == joint_name:
            frcrange = joint.get("actuatorfrcrange")
            if frcrange is None:
                raise ValueError(f"joint {joint_name} has no actuatorfrcrange")
            lo, hi = frcrange.split()
            return lo, hi
    raise ValueError(f"joint not found: {joint_name}")


def main() -> None:
    tree = ET.parse(SRC)
    root = tree.getroot()

    actuator_section = root.find("actuator")
    if actuator_section is None:
        raise ValueError("no <actuator> section found in g1.xml")

    converted = 0
    for actuator in list(actuator_section):
        name = actuator.get("name")
        if name not in LEG_JOINTS:
            continue
        if actuator.tag != "position":
            raise ValueError(f"expected <position> actuator for {name}, got <{actuator.tag}>")

        joint_name = actuator.get("joint")
        assert joint_name == name, f"actuator/joint name mismatch: {name} vs {joint_name}"
        lo, hi = find_joint_frcrange(root, joint_name)

        index = list(actuator_section).index(actuator)
        actuator_section.remove(actuator)
        motor = ET.Element(
            "motor",
            {
                "class": "g1",
                "name": name,
                "joint": joint_name,
                "ctrlrange": f"{lo} {hi}",
                "forcerange": f"{lo} {hi}",
            },
        )
        actuator_section.insert(index, motor)
        converted += 1

    if converted != len(LEG_JOINTS):
        raise ValueError(f"expected to convert {len(LEG_JOINTS)} actuators, converted {converted}")

    root.set("model", root.get("model", "") + "_walk")
    ET.indent(tree, space="  ")
    tree.write(DST, encoding="utf-8", xml_declaration=False)
    print(f"wrote {DST} ({converted} leg actuators converted position -> motor)")


if __name__ == "__main__":
    main()
