import scipy.signal as scisignal
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.ticker as ticker

from smplx import SMPL
from scipy.spatial.transform import Rotation as R
import torch

from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema

from scipy.interpolate import interp1d


def eye(n, batch_shape):
    iden = np.zeros(np.concatenate[batch_shape, [n, n]])
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


smpl = SMPL(model_path="./smpl/", gender="MALE", batch_size=1)


def motion_peak_onehot(joints):
    """Calculate motion beats.
    Kwargs:
        joints: [nframes, njoints, 3]
    Returns:
        - peak_onhot: motion beats.
    """
    # Calculate velocity.
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=10)  # 10 for 60FPS
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1

    return peak_onehot


def recover_motion_to_keypoints(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]  # (seq_len, 24, 3)
    return keypoints3d


def recover_to_axis_angles(motion):
    batch_size, seq_len, dim = motion.shape
    assert dim == 225
    transl = motion[:, :, 6:9]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


def calc_db(keypoints):
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats[0], kinetic_vel


if __name__ == "__main__":
    FPS = 60
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    y, sr = librosa.load('./aist_plusplus_final/all_musics/mWA0.wav', sr=SR)

    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=o_env, sr=sr, hop_length=512)
    times = librosa.times_like(o_env, sr=sr)
    onest_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

    motion_result = "./experiments/GPT_BA_BCE_2/eval/pkl/ep000100/gWA_sBM_cAll_d25_mWA0_ch01.npy"
    motion_result = np.load(motion_result, allow_pickle=True).item()['pred_position'][:, :]
    keypoints3d = motion_result.reshape(-1, 24, 3)
    # motion_beats = motion_peak_onehot(keypoints3d)[:2881]
    # motion_beats = motion_peak_onehot(keypoints3d)[15:]
    # motion_beats = motion_peak_onehot(keypoints3d)[:-11]
    # motion_beats = motion_peak_onehot(keypoints3d)[7:]
    pre_times = times
    motion_beats, kinetic_vel = calc_db(keypoints3d)

    plt.figure(figsize=(8, 3))
    # plt.rcParams['font.family'] = 'cursive'
    plt.rcParams['font.weight'] = 'bold'

    f = interp1d(times, o_env, kind='cubic')
    smoothed_y = f(times)

    plt.plot(times, o_env, alpha=0.5, label="Onset Strength or envelope")

    # plt.plot(times, smoothed_y, label="Onset Strength or envelope")

    plt.vlines(times[motion_beats[:-10]], ymin=-5, ymax=20, color='m', alpha=0.9,
               linestyle='--', label='motion beats')
    plt.vlines(times[beats], ymin=-5, ymax=20, color='g', alpha=0.9,
               linestyle='--', label='music beats')

    plt.ylim(-5, 25)
    plt.xlim(10, 15)

    plt.tight_layout()
    plt.legend(frameon=False, ncol=3, loc='upper center')
    plt.yticks([])
    # plt.axis('off')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.gca().spines['left'].set_visible('zero')
    plt.gca().spines['bottom'].set_position('zero')

    plt.xlabel('time', loc='right')
    plt.ylabel('dance velocity', loc='top')

    plt.xticks([])

    # arrow_props = dict(arrowstyle='->', color='black')
    #
    # plt.annotate('Annotation Text', xy=(0, 25), xytext=(3, -2),
    #              arrowprops=arrow_props)
    # plt.annotate('Annotation Text', xy=(40, 0), xytext=(3, -2),
    #              arrowprops=arrow_props)

    plt.show()
