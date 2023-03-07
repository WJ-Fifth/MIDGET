import numpy as np
import pickle
from smplx import SMPL
import smplx
import scipy
from scipy.spatial.transform import Rotation

from features import bvh, quat


def aa2rotmat(angles):
    """Converts axis angles to rotation matrices.
  Args:
    angles: np array of shape [..., 3].
  Returns:
    np array of shape [..., 9].
  """
    input_shape = angles.shape
    assert input_shape[-1] == 3, (f"input shape is not valid! got {input_shape}")
    output_shape = input_shape[:-1] + (9,)

    r = Rotation.from_rotvec(angles.reshape(-1, 3))
    if scipy.__version__ < "1.4.0":
        # as_dcm is renamed to as_matrix in scipy 1.4.0 and will be
        # removed in scipy 1.6.0
        output = r.as_dcm().reshape(output_shape)
    else:
        output = r.as_matrix().reshape(output_shape)
    return output


class BVHData(object):
    """Struct of a BVH data.
  A container for properties: ${global_trans} the 3d translation of the joint,
  ${axis_angles} the joint rotation represented in axis angles,
  ${euler_angles} the joint rotation represented in euler angles.
  """

    def __init__(self, global_trans, axis_angles):
        """Initialize with joints location and axis angle."""
        self.global_trans = global_trans
        self.axis_angles = axis_angles.reshape([-1, 3])
        self.euler_angles = np.ones_like(self.axis_angles)
        for index, axis_angle in enumerate(self.axis_angles):
            self.euler_angles[index] = aa2rotmat(axis_angle)

    def output(self):
        return self.euler_angles

def get_smpl_skeleton():
    kinematic_tree = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 4],
        [2, 5],
        [3, 6],
        [4, 7],
        [5, 8],
        [6, 9],
        [7, 10],
        [8, 11],
        [9, 12],
        [9, 13],
        [9, 14],
        [12, 15],
        [13, 16],
        [14, 17],
        [16, 18],
        [17, 19],
        [18, 20],
        [19, 21],
        [20, 22],
        [21, 23],
    ])

    names = [
        "Pelvis",
        "Left_hip",
        "Right_hip",
        "Spine1",
        "Left_knee",
        "Right_knee",
        "Spine2",
        "Left_ankle",
        "Right_ankle",
        "Spine3",
        "Left_foot",
        "Right_foot",
        "Neck",
        "Left_collar",
        "Right_collar",
        "Head",
        "Left_shoulder",
        "Right_shoulder",
        "Left_elbow",
        "Right_elbow",
        "Left_wrist",
        "Right_wrist",
        "Left_palm",
        "Right_palm",
    ]

    return kinematic_tree, names


if __name__ == "__main__":
    smpl = SMPL(model_path='../smpl/', gender='MALE', batch_size=1)

    parents = smpl.parents.detach().cpu().numpy()
    rest = smpl()
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24, :]
    print(rest_pose.shape)
    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 100

    scaling = None

    pkl_filename = "../experiments/GPT_BA_BCE_1/eval/pkl_files/ep000075/gHO_sBM_cAll_d21_mHO5_ch02.json.pkl"
    try:
        with open(pkl_filename, "rb") as f:
            data = pickle.load(f)
    except EOFError as e:
        message = "Aboring reading file %s due to: %s" % (pkl_filename, str(e))
        raise ValueError(message)

    joints = data['pred_position'].reshape(-1, 24, 3)
    print(joints.shape)

    trans = np.zeros((joints.shape[0], 3), dtype=np.float32)

    root_joint = joints[:, 0, :]

    # to quaternion
    rots = aa2rotmat(joints)

    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    positions[:, 0] += trans * 100
    rotations = np.degrees(quat.to_euler(rots, order=order))

    fps = 60

    _, names = get_smpl_skeleton()

    bvh_data = {
        "rotations": rotations,
        "positions": joints,
        "offsets": offsets,
        "parents": parents,
        "names": names,
        "order": order,
        "frametime": 1 / fps,
    }

    bvh.save("../experiments/GPT_BA_BCE_1/eval/bvh/gHO_sBM_cAll_d21_mHO5_ch02.bvh", bvh_data)

