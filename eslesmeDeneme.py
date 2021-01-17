# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# img1 = cv.imread('deneme.png',cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.imread('bir.png',cv.IMREAD_GRAYSCALE) # trainImage
# # Initiate SIFT detector
# sift = cv.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# # BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# # cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()


import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pysift

img = cv2.imread('cat.jpeg')
sift = cv2.SIFT_create()
gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kp,ds = pysift.computeKeypointsAndDescriptors(img)
#kp, ds = sift.detectAndCompute(gr, None)

# def locateForgery(img,key_points,descriptors,eps=40,min_sample=2):
# 		clusters=DBSCAN(eps=eps, min_samples=min_sample).fit(descriptors) # Find clusters using DBSCAN
# 		size=np.unique(clusters.labels_).shape[0]-1                       # Identify the number of clusters formed
# 		forgery=img.copy()                                               # Create another image for marking forgery
# 		if (size==0) and (np.unique(clusters.labels_)[0]==-1):
# 			print('No Forgery Found!!')
# 			return None                                               # If no clusters are found return
# 		if size==0:
# 			size=1
# 		cluster_list= [[] for i in range(size)]       # List of list to store points belonging to the same cluster
# 		for idx in range(len(key_points)):
# 		    if clusters.labels_[idx]!=-1:
# 		        cluster_list[clusters.labels_[idx]].append((int(key_points[idx].pt[0]),int(key_points[idx].pt[1])))
# 		for points in cluster_list:
# 		    if len(points)>1:
# 		        for idx1 in range(1,len(points)):
# 		            cv2.line(forgery,points[0],points[idx1],(255,0,0),5)  # Draw line between the points in a same cluster
# 		return forgery

clusters = DBSCAN(eps=40, min_samples=2).fit(ds)  # Find clusters using DBSCAN
size = np.unique(clusters.labels_).shape[0] - 1  # Identify the number of clusters formed
forgery = img.copy()  # Create another image for marking forgery
if (size == 0) and (np.unique(clusters.labels_)[0] == -1):
    print('No Forgery Found!!')
# return None                                               # If no clusters are found return
if size == 0:
    size = 1
cluster_list = [[] for i in range(size)]  # List of list to store points belonging to the same cluster
for idx in range(len(kp)):
    if clusters.labels_[idx] != -1:
        cluster_list[clusters.labels_[idx]].append((int(kp[idx].pt[0]), int(kp[idx].pt[1])))
for points in cluster_list:
    if len(points) > 1:
        for idx1 in range(1, len(points)):
            cv2.line(forgery, points[0], points[idx1], (255, 0, 0), 3)  # Draw line between the points in a same cluster
plt.imshow(forgery), plt.show()

# image = locateForgery(img,kp,ds,eps=40,min_sample=2)
# imgplot = plt.imshow(image)

# import cv2
#
# img1 = cv2.imread('bir.png', 0)
# img2 = cv2.imread('iki.png', 0)
#
# orb = cv2.ORB_create(nfeatures=500)
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
#
# # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
# # draw first 50 matches
# match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
# cv2.imshow('Matches', match_img)
# cv2.waitKey()
