import numpy as np
from scipy.spatial.transform import Rotation as R


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
