from scipy.ndimage.filters import convolve
from GaussOktav.gaussFiltresi import gaussF

def createOctave(init_level, s,sigma):
    oktav = [init_level]
    k = 2**(1/s)
    kernel = gaussF(k * sigma)

    for _ in range(s+2):
        next_level = convolve(oktav[-1], kernel)
        oktav.append(next_level)

    return oktav


def gauss_piramidi(im,num_oktav,s,sigma):
    pyr = []

    for _ in range(num_oktav):
        oktav = createOctave(im,s,sigma)
        pyr.append(oktav)
        im = oktav[-3][::2, ::2]

    return pyr

