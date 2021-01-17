import numpy as np
import numpy.linalg as lcebir
from GaussOktav.gaussFiltresi import gaussF
from YonAtama.yonAta import gradyanHesap, nicel_yonelim


def getYamaGradyan(yama):
    r1 = np.zeros_like(yama)
    r1[-1] = yama[-1]
    r1[:-1] = yama[1:]

    r2 = np.zeros_like(yama)
    r2[0] = yama[0]
    r2[1:] = yama[:-1]

    dy = r1 - r2

    r1[:, -1] = yama[:, -1]
    r1[:, :-1] = yama[:, 1:]
    r2[:, 0] = yama[:, 0]
    r2[:, 1:] = yama[:, :-1]

    dx = r1 - r2

    return dx, dy


def getAltBolgeHistogram(m, teta, gradyanBolmeSayisi, referansAcisi, bolmeGenislik, altBolgeAgirlik):
    histogram = np.zeros(gradyanBolmeSayisi, dtype=np.float32)

    c = altBolgeAgirlik / 2 - .5

    for i, (mag, aci) in enumerate(zip(m, teta)):
        aci = (aci - referansAcisi) % 360
        bolmeNo = nicel_yonelim(teta, gradyanBolmeSayisi)
        secim = mag

        histogram_interp = 1 - abs(aci - (bolmeNo * bolmeGenislik + bolmeGenislik / 2)) / (bolmeGenislik / 2)

        secim *= max(histogram_interp, 1e-6)

        gy, gx = np.unravel_index(i, (altBolgeAgirlik, altBolgeAgirlik))
        x_interp = max(1 - abs(gx - c) / c, 1e-6)
        y_interp = max(1 - abs(gy - c) / c, 1e-6)
        secim *= x_interp * y_interp
        histogram[bolmeNo] += secim

    histogram /= max(1e-6, lcebir.norm(histogram))
    histogram[histogram > 0.2] = 0.2
    histogram /= max(1e-6, lcebir.norm(histogram))

    return histogram


def getYerelTanimlayici(anahtarNoktalar, oktav, w=16, altBolgeNo=4, bolmeNo=8):
    tanimlayicilar = []
    bolmeGenislik = 360 // bolmeNo

    for anahtarNokta in anahtarNoktalar:
        cx, cy, s = int(anahtarNokta[0]), int(anahtarNokta[1]), int(anahtarNokta[2])
        s = np.clip(s, 0, oktav.shape[2] - 1)
        kernel = gaussF(w / 6)  # gaussian_filter sigma'yı 3 ile çarpar
        L = oktav[..., s]
        t, l = max(0, cy - w // 2), max(0, cx - w // 2)
        b, r = min(L.shape[0], cy + w // 2 + 1), min(L.shape[1], cx + w // 2 + 1)
        yama = L[t:b, l:r]
        dx, dy = getYamaGradyan(yama)

        if dx.shape[0] < w + 1:
            if t == 0:
                kernel = kernel[kernel.shape[0] - dx.shape[0]:]
            else:
                kernel = kernel[:dx.shape[0]]

        if dx.shape[1] < w + 1:
            if l == 0:
                kernel = kernel[kernel.shape[1] - dx.shape[1]:]
            else:
                kernel = kernel[:dx.shape[1]]

        if dy.shape[0] < w + 1:
            if t == 0:
                kernel = kernel[kernel.shape[0] - dy.shape[0]:]
            else:
                kernel = kernel[:dy.shape[0]]

        if dy.shape[1] < w + 1:
            if l == 0:
                kernel = kernel[kernel.shape[1] - dy.shape[1]:]
            else:
                kernel = kernel[:dy.shape[1]]

        m, teta = gradyanHesap(dx, dy)
        dx, dy = dx * kernel, dy * kernel

        altBolgeAgirlik = w // altBolgeNo
        ozellikVek = np.zeros(bolmeNo * altBolgeNo ** 2, dtype=np.float32)

        for i in range(0, altBolgeAgirlik):
            for j in range(0, altBolgeAgirlik):
                t, l = i * altBolgeAgirlik, j * altBolgeAgirlik
                b, r = min(L.shape[0], (i + 1) * altBolgeAgirlik), min(L.shape[1], (j + 1) * altBolgeAgirlik)
                histogram = getAltBolgeHistogram(m[t:b, l:r].ravel(), teta[t:b, l:r].ravel(), bolmeNo, anahtarNokta[3],
                                                 bolmeGenislik, altBolgeAgirlik)

                ozellikVek[i * altBolgeAgirlik * bolmeNo + j * bolmeNo: i * altBolgeAgirlik * bolmeNo + (
                            j + 1) * bolmeNo] = histogram.flatten()

        ozellikVek /= max(1e-6, lcebir.norm(ozellikVek))
        ozellikVek[ozellikVek > 0.2] = 0.2
        ozellikVek /= max(1e-6, lcebir.norm(ozellikVek))
        tanimlayicilar.append(ozellikVek)

    return np.array(tanimlayicilar)
