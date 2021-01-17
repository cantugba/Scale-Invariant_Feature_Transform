import numpy as np
import numpy.linalg as lcebir

def getAdayNokt(DoG, w=16):
    adayNoktalar = []
    DoG[:,:0]=0
    DoG[:,:-1] =0


    for i in range(w//2 +1,DoG.shape[0]-w//2-1):
        for j in range(w//2+1,DoG.shape[1]-w//2-1):
            for k in range(1,DoG.shape[2] - 1):
                parca = DoG[i-1:i+2,j-1:j+2,k-1:k+2]
                if np.argmax(parca) == 13 or np.argmin(parca) == 13:
                    adayNoktalar.append([i,j,k])


    return adayNoktalar


def yerelAnahtarNokt(DoG,x,y,s):
    dx = (DoG[y,x+1,s] - DoG[y,x-1,s]) / 2.
    dy = (DoG[y+1,x,s] - DoG[y-1,x,s]) / 2.
    ds = (DoG[y,x,s+1] - DoG[y,x,s-1]) / 2.

    dxx =DoG[y,x+1,s] - 2*DoG[y,x,s] + DoG[y,x-1,s]
    dyy = DoG[y+1,x,s] - 2*DoG[y,x,s] + DoG[y-1,x,s]
    dss = DoG[y,x,s+1] - 2*DoG[y,x,s] + DoG[y,x,s-1]

    # dxy = ((DoG[y + 1, x + 1, s] - DoG[y + 1, x - 1, s]) — (DoG[y-1, x+1, s] - DoG[y-1, x-1, s])) / 4.
    dxy = ((DoG[y+1,x+1,s] - DoG[y+1,x-1,s]) - (DoG[y-1,x+1,s] - DoG[y-1,x-1,s]))/4.
    dys = ((DoG[y+1,x,s+1] - DoG[y-1,x,s+1]) - (DoG[y+1,x,s-1] - DoG[y-1,x,s-1]))/4.
    dxs = ((DoG[y,x+1,s+1] - DoG[y,x-1,s+1]) - (DoG[y,x+1,s-1] - DoG[y,x-1,s-1]))/4.

    Jacobian = np.array(dx,dy,ds)
    Hessian = np.array([[dxx,dxy,dxs],[dxy,dyy,dys],[dxs,dys,dss]])
    expr = -lcebir.inv(Hessian).dot(Jacobian)

    return expr,Jacobian,Hessian[:2,:2],x,y,s


def anahtarNoktaBul(DoG,R_th,t_c,w):
    adayNoktalar = getAdayNokt(DoG,w)
    anahtarNokta = []

    for i,aday in enumerate(adayNoktalar):
        y,x,s = aday[0],aday[1,],aday[2]
        expr,Jacobian,Hessian,x,y,s, =yerelAnahtarNokt(DoG,x,y,s)

        kontrast = DoG[y,x,s] + 5* Jacobian.dot(expr)
        if abs(kontrast) < t_c: continue

        w,v = lcebir.eig(Hessian) # özdeğer ve özvektör hesabı özvektörler, sütun v [:, i], w [i] özdeğerine karşılık gelen özvektördür.
        r = w[1] / w[0]
        R = (r+1) **2 /r

        if R>R_th : continue

        aNokt = np.array([x,y,s]) + expr
        anahtarNokta.append(aNokt)


    return np.array(anahtarNokta)



#DoG piramidinin tamamı için anahtar noktaları hesapla
def getAnahtarNokt(DoG_piramidi,R_th,t_c,w):
 anaharNoktalar = []

 for DoG in DoG_piramidi:
     anaharNoktalar.append(anahtarNoktaBul(DoG,R_th,t_c,w))

 return anaharNoktalar


