#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 1 Starter Code


Author(s):
Shiva Kumar Tekumatla (stekumatla@wpi.edu)
MS in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

from email.headerregistry import ContentTransferEncodingHeader
from socketserver import ThreadingMixIn
from ssl import SSLSocket
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt 
from skimage import feature 
from scipy.spatial import distance
import random 
# Add any python libraries here
class MyPano:
	def __init__(self) -> None:
		pass
	def plot_image(self,Image,name):
		cv2.imshow(name,Image)
		if cv2.waitKey(0) & 0xff == 27:
			cv2.destroyAllWindows()
	def draw_matches(self,Image1,Image2,matches):
		# Image1 = Img1.copy()
		# Image2 = Img2.copy()
		height1,width1 = Image1.shape[:2]
		height2,width2 = Image2.shape[:2]
		match_image = np.zeros((max(height1, height2), width1 + width2, 3), dtype="uint8")
		match_image[0:height1, 0:width1] = Image1
		match_image[0:height2, width1:] = Image2

		for match in matches:
			point1 = (match[0][1],match[0][0])
			point2 = (match[1][1]+width1,match[1][0])
			cv2.circle(match_image,point1,2,255, -1)
			cv2.circle(match_image,point2,2,255, -1)
			cv2.line(match_image, point1, point2, (0, 0, 255), 1)
		return match_image

	def corner_detection(self,Image,block_size,sobel_size):
		"""Using Harris Corners to detect the corners in a given image
		Based on  https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
		"""
		img = Image.copy()
		gray_img = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))  # converting the image to gray scale and then to float32 
		dst = cv2.cornerHarris(gray_img,block_size,sobel_size,0.04)
		dst = cv2.dilate(dst,None)
		dst[dst<0.01*dst.max()] = 0  # Setting below threshold to zero 
		img[dst!=0]=[0,0,255]  
		return img, dst

	def ANMS(self,C_Img,N_best):
		#C_Image - Corner Score Image 
		#N_best - Number of best corners needed
		"""The objective of this step is to detect corners such that 
		they are equally distributed across the image in order to avoid 
		weird artifacts in warping."""
		#Find all local maxima see: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_peak_local_max.html
		coordinates = feature.peak_local_max(C_Img, min_distance=10,threshold_abs=0.1) #coordinates are stored 
		r = np.zeros((coordinates.shape[0], 1), dtype=np.float32) 
		r[:] = np.inf  
		index = 0 
		for i in coordinates:
			yi,xi = i #Inverting the indices to match row and column 
			ED = np.inf 
			for j in coordinates:
				yj,xj =j  #Inverting the indices to match row and column 
				if (C_Img[yj,xj] > C_Img[yi,xi]):
					ED = (xj - xi)**2 + (yj - yi)**2
				if (ED < r[index][0]):
					r[index][0] = ED
			index+=1
		best_coordinates = coordinates[np.flip(np.argsort(r[:,0]))] #descending order 
		return best_coordinates[0:N_best,:]

	def featureDescriptor(self,Image,coordinate,patch_size=40):
		i,j= coordinate  #Inverting the indices to match row and column 
		gray_img = np.float32(cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY))
		patch = gray_img[i-patch_size//2:i+patch_size//2, j-patch_size//2:j+patch_size//2]  #Take a patch of size 40×40 centered (this is very important) around the keypoint/feature point
		# print(patch)
		gaussian = cv2.GaussianBlur(patch,(3,3),cv2.BORDER_DEFAULT)  
		#Now, sub-sample the blurred output (this reduces the dimension) to 8×8.
		resampled_img = cv2.resize(gaussian,(8,8),interpolation = cv2.INTER_AREA)
		reshaped_img = resampled_img.reshape(-1) #Then reshape to obtain a 64×1 vector
		#Standardize the vector to have zero mean and variance of 1.
		# Standardization is used to remove bias and to achieve some amount of 
		# illumination invariance.
		# See https://bookdown.org/ndphillips/YaRrr/standardization-z-score.html
		mean = resampled_img.mean() 
		std = resampled_img.std()
		return (resampled_img -mean)/std 
	
	def featureMatching(self,Image_1,Coordinates_1,Image_2,Coordinates_2,ratio,patch = 40 ):
		"""finding feature correspondences between the 2 images"""
		def ssd(A,B):
			# Taken from https://stackoverflow.com/a/58589090/5658788
			dif = A.ravel() - B.ravel()
			return np.dot( dif, dif )
		def filter_coordinates(Coordinates,width,height):
			"""Filtering the coordinates to make sure these  coordinates as centers can accommodate patches """
			temp_array =[]
			for coordinate in Coordinates:
				i,j = coordinate 
				if (i - (patch/ 2) > 0) & (i + (patch / 2) < height) & (j - (patch/ 2) > 0) & (j + (patch / 2) < width):
					temp_array.append([i,j])
			return temp_array
	
		"""Pick a point in image 1, compute sum of square differences between 
		all points in image 2. Take the ratio of best match (lowest distance) to 
		the second best match (second lowest distance) and if this is below some 
		ratio keep the matched pair or reject it. Repeat this for all points in image 1.
		You will be left with only the confident feature correspondences and these points
		will be used to estimate the transformation between the 2 images, also called as 
		Homography"""
		# FIrst make sure the patches in the edges are ignored
		height1,width1 = Image_1.shape[:2]
		height2,width2 = Image_2.shape[:2]
		Coordinates_1 = filter_coordinates(Coordinates_1,width1,height1)
		Coordinates_2 = filter_coordinates(Coordinates_2,width2,height2)
	
		features2 = [self.featureDescriptor(Image_2,coordinate) for coordinate in Coordinates_2]
		matches =[]
		for coordinate in Coordinates_1:
			feature_1 = self.featureDescriptor(Image_1,coordinate)
			SSDs = np.array([ssd(feature_1,feature_2) for feature_2 in features2])
			ids = np.argpartition(SSDs, 2) # finding the indices of two smallest values 
			low_id = ids[0] 
			lowest,second_lowest = SSDs[ids[:2]]
			if lowest/second_lowest <ratio:
				matches.append([coordinate , Coordinates_2[low_id]])

		return Coordinates_1,Coordinates_2,matches 

	def RANSAC(self,matches,threshold,Nmax=10000):
		"""We now have matched all the features correspondences 
		but not all matches will be right. To remove incorrect matches, 
		we will use a robust method called Random Sample Concensus or RANSAC 
		to compute homography."""
		matches = np.asarray(matches)
		Pi = matches[:,0]
		Pi_dash = matches[:,1]  #Separating two coordinates in a pair 
		ones = np.ones((len(Pi), 1)) 
		inliers_best = 0
		for _ in range(Nmax):
			#step-1 : Select four feature pairs
			four_random_indices = random.sample(range(len(matches)),4) 
			four_random_pairs = matches[four_random_indices]
			Pi4 = four_random_pairs[:,0]
			Pi4_dash = four_random_pairs[:,1]
			#Step-2 : Compute homography H between the previously picked point pairs.
			H = cv2.getPerspectiveTransform(np.float32(Pi4), np.float32(Pi4_dash)) # See https://www.youtube.com/watch?v=PtCQH93GucA&ab_channel=Pysource to understand this concept 
			Hpi = np.dot(H,np.transpose(np.hstack((Pi,ones))))
			Hpi = np.transpose(Hpi[:2,:]/(Hpi[2,:] + 1e-10)) #normalizing the HPi
			#Step-3 : 
			SSDs = np.apply_along_axis(np.linalg.norm, 1, Pi_dash -Hpi)**2 #compute SSD
			SSDs[SSDs<=threshold] = 1
			SSDs[SSDs>threshold]  = 0
			inliers = np.sum(SSDs)
			if inliers > inliers_best:
				inliers_best =inliers
				H_best = H
				inlier_inds = np.where(SSDs== 1)
		return Pi[inlier_inds],Pi_dash[inlier_inds],matches[inlier_inds],H_best

	def Blending(self,Image1,Image2,filtered_matches,H):
		pass
		
def main():
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
	Args = Parser.parse_args()
	NumFeatures = Args.NumFeatures
	panorama = MyPano()

	basePath = "/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /Projects/RBE549_CV_Projects/ajayamoorthy_p1/Phase1/Data/Train/Set1/" #1.jpg"
	Image1_path = basePath +"1.jpg"
	Image2_path = basePath +"2.jpg"
	Image3_path = basePath +"3.jpg"

    # """
    # Read a set of images for Panorama stitching
    # """
	Image1 = cv2.imread(Image1_path)
	Image2 = cv2.imread(Image2_path)
	Image3 = cv2.imread(Image3_path)
	panorama.plot_image(Image1,"input1")
	panorama.plot_image(Image2,"input2")
	panorama.plot_image(Image3,"input3")

    # """
	# Corner Detection
	# Save Corner detection output as corners.png
	# """
	
	corner_score_image1,dst1 = panorama.corner_detection(Image1,2,3)
	corner_score_image2,dst2 = panorama.corner_detection(Image2,2,3)
	corner_score_image3,dst3 = panorama.corner_detection(Image3,2,3)
	panorama.plot_image(corner_score_image1,"corners")
	cv2.imwrite('corners1.png',corner_score_image1)
	cv2.imwrite('corners2.png',corner_score_image2)
	cv2.imwrite('corners3.png',corner_score_image3)

    # """
	# Perform ANMS: Adaptive Non-Maximal Suppression
	# Save ANMS output as anms.png
	# """
	anms1 = panorama.ANMS(dst1,200)
	anms2 = panorama.ANMS(dst2,200)
	anms3 = panorama.ANMS(dst3,200)
	img1 = Image1.copy()
	img2 = Image2.copy()
	img3 = Image3.copy()

	for coordinate in anms1:
		i,j= int(coordinate[1]),int(coordinate[0])
		cv2.circle(img1,(i,j),2,(0,0,255), -1)
	panorama.plot_image(img1,"anms1")
	cv2.imwrite('anms1.png',img1)

	for coordinate in anms2:
		i,j= int(coordinate[1]),int(coordinate[0])
		cv2.circle(img2,(i,j),2,(0,0,255), -1)
	panorama.plot_image(img2,"anms2")
	cv2.imwrite('anms2.png',img2)

	for coordinate in anms3:
		i,j= int(coordinate[1]),int(coordinate[0])
		cv2.circle(img3,(i,j),2,(0,0,255), -1)
	panorama.plot_image(img3,"anms3")
	cv2.imwrite('anms3.png',img3)
	
    # """
	# Feature Descriptors
	# Save Feature Descriptor output as FD.png
	# """

    # """
	# Feature Matching
	# Save Feature Matching output as matching.png
	# """

	k1,k2,matches12 = panorama.featureMatching(Image1,anms1,Image2,anms2,1)
	#Drawing matches  . cv2.Drawmatches did not work , hence writing my own match plotting
	match_image = panorama.draw_matches(Image1,Image2,matches12)
	panorama.plot_image(match_image,"Matches12")

	cv2.imwrite('matching12.png',match_image)
	k1,k2,matches23 = panorama.featureMatching(Image2,anms2,Image3,anms3,1)
	match_image = panorama.draw_matches(Image2,Image3,matches23)
	panorama.plot_image(match_image,"Matches23")
	cv2.imwrite('matching23.png',match_image)

	k1,k2,matches31 = panorama.featureMatching(Image3,anms3,Image1,anms1,1)
	match_image = panorama.draw_matches(Image3,Image1,matches31)
	panorama.plot_image(match_image,"Matches31")
	cv2.imwrite('matching31.png',match_image)
    # """
	# Refine: RANSAC, Estimate Homography
	# """
	k1,k2,filtered_matches12,H_best = panorama.RANSAC(matches12,100)
	inliers_image12 = panorama.draw_matches(Image1,Image2,filtered_matches12)
	panorama.plot_image(inliers_image12,"RANSAC12")
	cv2.imwrite('inliers12.png',inliers_image12)

	k1,k2,filtered_matches23,H_best = panorama.RANSAC(matches23,100)
	inliers_image23 = panorama.draw_matches(Image2,Image3,filtered_matches23)
	panorama.plot_image(inliers_image23,"RANSAC23")
	cv2.imwrite('inliers23.png',inliers_image23)

	k1,k2,filtered_matches31,H_best = panorama.RANSAC(matches31,100)
	inliers_image31 = panorama.draw_matches(Image3,Image1,filtered_matches31)
	panorama.plot_image(inliers_image31,"RANSAC31")
	cv2.imwrite('inliers31.png',inliers_image31)

    # """
	# Image Warping + Blending
	# Save Panorama output as mypano.png
	# """
	image_stitched = panorama.Blending(Image1,Image2,filtered_matches12,H_best)
	# panorama.plot_image(image_stitched,"stitch")
	
if __name__ == "__main__":
    main()
