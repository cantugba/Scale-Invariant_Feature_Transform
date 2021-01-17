from GaussOktav.oktav import *
import numpy as np
# def createGaussPiramid(im,num_oktav,s,sigma):
#     pyr = []
#
#     for _ in range(num_oktav):
#         oktav = createOctave(im,s,sigma)
#         pyr.append(oktav)
#         im = oktav[-3][::2, ::2]
#
#     return pyr


def dog_oktav(gauss_oktav):
    oktav = []

    for i in range(1,len(gauss_oktav)):
        oktav.append(gauss_oktav[i] - gauss_oktav[i - 1])

    return np.concatenate([o[:,:,np.newaxis] for o in oktav], axis=2)


def createDogPiramit(gauss_piramit):
    pyr = []

    for gaussOktav in gauss_piramit:
        pyr.append(dog_oktav(gaussOktav))

    return pyr