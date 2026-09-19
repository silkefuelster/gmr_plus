"""
Convert egoallo egocentric motion output to an AMASS-style SMPL-H .npz file.

egoallo stores local joint rotations as wxyz quaternions (body_quats,
left_hand_quats, right_hand_quats, each shape (num_hypotheses, T, J, 4)) plus a
world-root SE3 trajectory Ts_world_root (num_hypotheses, T, 7) in wxyz_xyz
order. This is not the axis-angle root_orient/pose_body/trans format the rest
of this codebase (load_smplh_file) expects, so this script bridges the two.

Usage:
    python scripts/convert_egoallo_to_amass_smplh.py \\
        --egoallo_file <path/to/egoallo_output> \\
        --save_path <output.npz>
"""

import argparse
import pathlib

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot


def _load_egoallo_file(egoallo_file):
    path = pathlib.Path(egoallo_file)
    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu")
    else:
        data = dict(np.load(path, allow_pickle=True))
    return data


def _field(data, key):
    """Fetch a field from either a dict-like or attribute-style container."""
    if isinstance(data, dict):
        value = data[key]
    else:
        value = getattr(data, key)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _quats_wxyz_to_rotvec(quats_wxyz):
    """Convert an array of wxyz quaternions to axis-angle rotation vectors.

    Args:
        quats_wxyz: (..., 4) array of quaternions in wxyz order.

    Returns:
        (..., 3) array of axis-angle rotation vectors.
    """
    flat = quats_wxyz.reshape(-1, 4)
    rotvecs = sRot.from_quat(flat, scalar_first=True).as_rotvec()
    return rotvecs.reshape(quats_wxyz.shape[:-1] + (3,))


def main():
    parser = argparse.ArgumentParser(
        description="Convert egoallo output to AMASS-style SMPL-H .npz"
    )
    parser.add_argument(
        "--egoallo_file",
        type=str,
        required=True,
        help="Path to egoallo output file (.pt or .npz).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the converted SMPL-H .npz file.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="SMPL-H gender to record in the output file.",
    )
    parser.add_argument(
        "--hypothesis",
        type=int,
        default=0,
        help="Index into egoallo's sampled batch of hypotheses.",
    )
    args = parser.parse_args()

    data = _load_egoallo_file(args.egoallo_file)
    h = args.hypothesis

    body_quats = _field(data, "body_quats")[h]  # (T, J_body, 4) wxyz
    left_hand_quats = _field(data, "left_hand_quats")[h]  # (T, J_hand, 4) wxyz
    right_hand_quats = _field(data, "right_hand_quats")[h]  # (T, J_hand, 4) wxyz
    Ts_world_root = _field(data, "Ts_world_root")[h]  # (T, 7) wxyz_xyz
    betas = _field(data, "betas")[h]  # (T, 16), re-estimated per frame
    timestamps_ns = _field(data, "timestamps_ns")
    if timestamps_ns.ndim > 1:
        timestamps_ns = timestamps_ns[h]

    num_frames = body_quats.shape[0]

    pose_body = _quats_wxyz_to_rotvec(body_quats).reshape(num_frames, -1)
    left_hand_pose = _quats_wxyz_to_rotvec(left_hand_quats).reshape(num_frames, -1)
    right_hand_pose = _quats_wxyz_to_rotvec(right_hand_quats).reshape(num_frames, -1)

    root_orient = sRot.from_quat(Ts_world_root[:, :4], scalar_first=True).as_rotvec()
    trans = Ts_world_root[:, 4:]

    # egoallo re-estimates shape every frame via guidance optimization;
    # downstream retargeting wants a single fixed shape per sequence.
    betas_mean = betas.mean(axis=0)

    dt_ns = np.median(np.diff(timestamps_ns.astype(np.float64)))
    mocap_frame_rate = 1e9 / dt_ns

    save_dict = {
        "root_orient": root_orient.astype(np.float32),
        "pose_body": pose_body.astype(np.float32),
        "left_hand_pose": left_hand_pose.astype(np.float32),
        "right_hand_pose": right_hand_pose.astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": betas_mean.astype(np.float32),
        "gender": args.gender,
        "mocap_frame_rate": np.array(mocap_frame_rate),
    }

    save_path = pathlib.Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **save_dict)
    print(
        f"Saved converted SMPL-H motion ({num_frames} frames, "
        f"{mocap_frame_rate:.2f} fps) to {save_path}"
    )


if __name__ == "__main__":
    main()
