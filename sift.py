from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
import cv2 as cv

from GaussOktav.gaussFiltresi import gaussF
from GaussOktav.oktav import gauss_piramidi
from GaussOktav.gaussOktav import createDogPiramit
from ExtremaDetect.adayAnahtarNokt import getAnahtarNokt
from YonAtama.yonAta import yonAta
from yerelTanimlayici.yerelTanim import getYerelTanimlayici


class SIFT(object):
    def __init__(self, im, s=3, oktavNo=4, s0=1.3, sigma=1.6, r_th=10, t_c=0.03, w=16):
        self.im = convolve(rgb2gray(im), gaussF(s0))
        self.s = s
        self.sigma = sigma
        self.oktavNo = oktavNo
        self.t_c = t_c
        self.R_th = (r_th + 1) ** 2 / r_th
        self.w = w

    def get_ozellik(self):
        gaussPiramit = gauss_piramidi(self.im, self.oktavNo, self.s, self.sigma)
        DoG_pyr = createDogPiramit(gaussPiramit)

        anahtarNoktPiramit = getAnahtarNokt(DoG_pyr, self.R_th, self.t_c, self.w)
        ozellikler = []

        for i, DoG_oktav in enumerate(DoG_pyr):
            anahtarNoktPiramit[i] = yonAta(anahtarNoktPiramit[i], DoG_oktav)
            ozellikler.append(getYerelTanimlayici(anahtarNoktPiramit[i], DoG_oktav))

        self.anahtarNoktPiramit = anahtarNoktPiramit
        self.ozellikler = ozellikler

        return ozellikler
