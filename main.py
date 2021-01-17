from sift import SIFT
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.io import imread

if __name__ == '__main__':
    im = imread('1.jpeg')
    s = SIFT(im)
    ozellik = s.get_ozellik()

    ky = s.anahtarNoktPiramit

    ky, ax = plt.subplots(1, s.oktavNo)

    for i in range(s.oktavNo):
        ax[i].imshow(im)
        olcek = ky[i] * (2 ** i)
        ax[i].scatter(olcek[:, 0], olcek[:, 1], c='r', s=2.5)

    plt.imshow(im),plt.show()
