import numpy as np
import cv2
basePath = "/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /Projects/RBE549_CV_Projects/ajayamoorthy_p1/Phase1/Data/Train/Set1/" #1.jpg"
Image1_path = basePath +"1.jpg"
Image2_path = basePath +"2.jpg"
Image3_path = basePath +"3.jpg"
cv2.ocl.setUseOpenCL(False)


img1 = cv2.imread(Image1_path,0)
img2 = cv2.imread(Image2_path,0)

img3 = img1.copy()

# Initiate ORB detector
orb = cv2.ORB_create()

# compute the descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)
print(matches[1])
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)


# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)

cv2.imshow("Matches",img3)
cv2.waitKey(-1) 