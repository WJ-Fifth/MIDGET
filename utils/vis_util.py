import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def draw_skeleton(ax, kpts, parents=[], is_right=[], cols=["#3498db", "#e74c3c"], marker='o', line_style='-',
                  label=None):
    """

    :param kpts: joint_n*(3 or 2)
    :param parents:
    :return:
    """
    # ax = plt.subplot(111)
    joint_n, dims = kpts.shape
    # by default it is human 3.6m joints

    if dims > 2:
        ax.view_init(75, 90)
        ax.set_zlabel('Z Label')

    is_label = True
    for i in range(len(parents)):
        if parents[i] < 0:
            continue
        # if dims == 2:
        #     if not (parents[i] in idx_choosed and i in idx_choosed):
        #         continue

        if dims == 2:
            # ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]], c=cols[is_right[i]],
            #         linestyle=line_style,
            #         alpha=0.5 if is_right[i] else 1, linewidth=3)
            if label is not None and is_label:
                ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]], c=cols[is_right[i]],
                        linestyle=line_style,
                        alpha=1 if is_right[i] else 0.6, label=label)
                is_label = False
            else:
                ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]], c=cols[is_right[i]],
                        linestyle=line_style,
                        alpha=1 if is_right[i] else 0.6)
        else:
            if label is not None and is_label:
                ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]],
                        [kpts[parents[i], 2], kpts[i, 2]], linestyle=line_style, c=cols[is_right[i]],
                        alpha=1 if is_right[i] else 0.6, linewidth=3, label=label)
                is_label = False
            else:
                ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]],
                        [kpts[parents[i], 2], kpts[i, 2]], linestyle=line_style, c=cols[is_right[i]],
                        alpha=1 if is_right[i] else 0.6, linewidth=3)

    return None