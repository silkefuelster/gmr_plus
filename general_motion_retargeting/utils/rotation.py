import numpy as np
from scipy.spatial.transform import Rotation as R


def smooth_human_data_frames(frames: list[dict], window: int = 5) -> list[dict]:
    """Moving-average-smooth a sequence of {body_name: (pos, quat)} frames.
    """
    if window < 2:
        return frames
    half = window // 2

    body_names = list(frames[0].keys())
    n = len(frames)
    smoothed = [dict() for _ in range(n)]

    for body_name in body_names:
        pos = np.stack([np.asarray(f[body_name][0], dtype=float) for f in frames])  # (n, 3)
        quat = np.stack([np.asarray(f[body_name][1], dtype=float) for f in frames])  # (n, 4) wxyz

        # Enforce hemisphere continuity so q and -q (the same rotation) don't
        # get averaged as if they were opposite values.
        for t in range(1, n):
            if np.dot(quat[t], quat[t - 1]) < 0:
                quat[t] = -quat[t]

        pos_padded = np.pad(pos, ((half, half), (0, 0)), mode="edge")
        quat_padded = np.pad(quat, ((half, half), (0, 0)), mode="edge")

        pos_smooth = np.empty_like(pos)
        quat_smooth = np.empty_like(quat)
        for t in range(n):
            pos_smooth[t] = pos_padded[t : t + window].mean(axis=0)
            q = quat_padded[t : t + window].mean(axis=0)
            quat_smooth[t] = q / np.linalg.norm(q)

        for t in range(n):
            smoothed[t][body_name] = (pos_smooth[t], quat_smooth[t])

    return smoothed


def swing_twist_decompose(rotation: R, twist_axis: np.ndarray) -> tuple[R, R]:
    """Split `rotation` into `swing * twist`, where `twist` is the component
    of `rotation` that is a pure rotation about `twist_axis` (expressed in
    the same frame `rotation` operates in).
    """
    twist_axis = twist_axis / np.linalg.norm(twist_axis)
    x, y, z, w = rotation.as_quat()
    proj = np.dot([x, y, z], twist_axis)
    twist_quat = np.array([*(proj * twist_axis), w])
    norm = np.linalg.norm(twist_quat)
    if norm < 1e-8:
        twist = R.identity()
    else:
        twist = R.from_quat(twist_quat / norm)
    swing = rotation * twist.inv()
    return swing, twist
