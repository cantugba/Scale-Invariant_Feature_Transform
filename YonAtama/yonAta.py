import numpy as np
from numpy import linalg as lcebir

from GaussOktav import gaussFiltresi

def yonAta(anahtarNoktalar,oktav,gradyanBolmeSayisi = 36):
    yeni_anahtarNokt = []
    bolme_genislik = 360//gradyanBolmeSayisi

    for anahtarNokta in anahtarNoktalar:
        cx,cy,s = int(anahtarNokta[0]), int(anahtarNokta[1]),int(anahtarNokta[2])
        s = np.clip(s,0,oktav.shape[2] - 1)
        #clip Bir aralık verildiğinde, aralığın dışındaki değerler aralık kenarlarına kırpılır.
        # Örneğin, [0, 1] aralığı belirtilirse, 0'dan küçük değerler 0 olur ve 1'den büyük değerler 1 olur.

        sigma = anahtarNokta[2]*1.5
        w =int(2*np.ceil(sigma) + 1)
        kernel = gaussFiltresi(sigma)
        L = oktav[:,s] #oktav[...,s]

        histogram = np.zeros(gradyanBolmeSayisi,dtype=np.float32)

        for yYonu in range(-w,w+1):
            for xYonu in range(-w,w+1):
                x,y = cx + xYonu, cy + yYonu
                if x < 0 or x > oktav.shape[1] - 1: continue
                elif y < 0 or y > oktav.shape[0]-1: continue

                #gradyan buyuklugu ve yonelimi
                m,teta = getGradyan(L,x,y)
                agirlik = kernel[yYonu + w,xYonu +w] * m
                bolme = nicel_yonelim(teta,gradyanBolmeSayisi)
                histogram[bolme]+= agirlik


            maks_bolme = np.argmax(histogram)

            yeni_anahtarNokt.append([anahtarNokta[0],anahtarNokta[1],anahtarNokta[2],parabol(histogram,maks_bolme,bolme_genislik)])
            maks_deger = np.max(histogram)

            for bolmeNo,deger in enumerate(histogram):
                if bolmeNo == maks_bolme: continue

                if .8*maks_deger <= deger:
                    yeni_anahtarNokt.append([anahtarNokta[0],anahtarNokta[1],anahtarNokta[2],parabol(histogram,bolmeNo,bolme_genislik)])


        return np.array(yeni_anahtarNokt)



def getGradyan(L,x,y):
    dy = L[min(L.shape[0]-1,y+1),x] -L[max(0,y-1),x]
    dx = L[y,min(L.shape[1] -1,x+1)] - L[y,max(0,x-1)]

    return gradyanHesap(dx,dy)


def gradyanHesap(dx,dy):
    m = np.sqrt(dx**2 + dy**2)
    teta = (np.arctan2(dy,dx) + np.pi) *180/np.pi

    return m,teta


#gradyanın sürekli açısını bir histogram bölmesine dönüştüren nicelleştirme_oriantasyon vardır:

def nicel_yonelim(teta,gradyanBolmeSayisi):
    bolme_genislik = 360 // gradyanBolmeSayisi
    return int(np.floor(teta)//bolme_genislik)

 #maksimuma en yakın üç histogram değerine bir parabol sığdırmanız gerekmekte
# En küçük kareler çözümünü elde ederiz, burada maksimum histogram bölmesinin merkezi ve iki bitişik bölmesi bağımsız değişken olarak
# ve bu histogramdaki değer bağımlı değişken olarak alınır. Parabolün katsayıları bulunduğunda,
# iyileştirilmiş yönelimi elde etmek için -b / 2a'yı kullanın.

def parabol(histogram,bolmeNo,bolmeGenislik):
    merkeziDeger = bolmeNo*bolmeGenislik + bolmeGenislik /2.

    if bolmeNo == len(histogram) - 1:
        sagDeger = 360 + bolmeGenislik / 2.
    else:
        sagDeger = (bolmeNo + 1)*bolmeGenislik + bolmeGenislik/2.

    if bolmeNo == 0:
        solDeger =-bolmeGenislik/2.
    else:
        solDeger = (bolmeNo-1)* bolmeGenislik + bolmeGenislik/2.


    A = np.array(
        [merkeziDeger**2,merkeziDeger,1],
        [sagDeger**2,sagDeger,1],
        [solDeger**2,solDeger,1]
    )

    b = np.array(
        histogram[bolmeNo],
        histogram[(bolmeNo+1) % len(histogram)],
        histogram[(bolmeNo-1)% len(histogram)]
    )

    x = lcebir.lstsq(A,b,rcond=None)[0]

    if x[0] == 0:
        x[0] = 1e-6
        return -x[1]/(2*x[0])