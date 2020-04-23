import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=None, cmap=plt.cm.Blues, linewidths=0, edgecolors='k')
    # plt.axis([0, 300, 0, 300])
    plt.colorbar()


def load_image(path):
    if not os.path.isfile(path):
        return None, None

    image = cv2.imread(path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # flip since using opencv
    gray = np.transpose(gray)
    gray = np.fliplr(gray)

    # threshold black values
    idx = gray < 200
    gray[idx] = 0

    obstacles_indices = np.where(gray == 0)
    ox = obstacles_indices[0]
    oy = obstacles_indices[1]
    return gray, ox, oy

