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


import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt 
from skimage import feature 
from scipy.spatial import distance
import random 
import imutils 
import glob 
import itertools 
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
		# (H_best, status) = cv2.findHomography(Pi[inlier_inds],Pi_dash[inlier_inds], cv2.RANSAC, 4)
		return Pi[inlier_inds],Pi_dash[inlier_inds],matches[inlier_inds],H_best

	def Blending(self,Image1,Image2,Image_stitcher):  #,filtered_matches,H):
		Images = []
		Images.append(Image1)
		Images.append(Image2)
		err,stitched_image = Image_stitcher.stitch(Images)
		return err,stitched_image

		# pass
		
def main():
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
	Args = Parser.parse_args()
	NumFeatures = Args.NumFeatures
	panorama = MyPano()

	Path = glob.glob("/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /Projects/RBE549_CV_Projects/ajayamoorthy_p1/Phase1/Data/Train/Set1/*.jpg")
	
    # """
    # Read a set of images for Panorama stitching
    # """
	Images = []
	i=0
	for image in Path:
		img = cv2.imread(image)
		Images.append(img)
		panorama.plot_image(img,"input_"+str(i))
		i+=1
	# Image1 ,Image2 , Image3 = Images 

    # """
	# Corner Detection
	# Save Corner detection output as corners.png
	# """
	corner_score_images = []
	dst = []
	for i,img in enumerate(Images):
		Image = img.copy()
		cor_img,d = panorama.corner_detection(Image,2,3)
		corner_score_images.append(cor_img)
		dst.append(d)
		panorama.plot_image(cor_img,"corners"+str(i+1))
		cv2.imwrite(f'corners{i+1}.png',cor_img)

    # """
	# Perform ANMS: Adaptive Non-Maximal Suppression
	# Save ANMS output as anms.png
	# """
	# dst1,dst2,dst3 = dst 
	ANMS = []
	for id,d in enumerate(dst):
		img = Images[id].copy()
		anms = panorama.ANMS(d,200)
		ANMS.append(anms)
		for coordinate in anms:
			i,j= int(coordinate[1]),int(coordinate[0])
			cv2.circle(img,(i,j),2,(0,0,255), -1)
		panorama.plot_image(img,"anms"+str(id+1))
		cv2.imwrite(f'anms{id+1}.png',img)
	
    # """
	# Feature Descriptors
	# Save Feature Descriptor output as FD.png
	# """

    # """
	# Feature Matching
	# Save Feature Matching output as matching.png
	# """
	ids = list(itertools.combinations(range(len(Images)), 2))
	Matches ={}
	for i,j in ids:
		img1 =Images[i].copy()
		img2 = Images[j].copy()
		k1,k2,matches = panorama.featureMatching(img1,ANMS[i],img2,ANMS[j],1)
		Matches[f"{i}{j}"]= matches
		match_image = panorama.draw_matches(img1,img2,matches)
		panorama.plot_image(match_image,f"Matches{i+1}{j+1}")
	
    # """
	# Refine: RANSAC, Estimate Homography
	# """
	# print(Matches)
	for i,j in ids:
		img1 =Images[i].copy()
		img2 = Images[j].copy()
		match =Matches[f"{i}{j}"]
		k1,k2,filtered_matches,H_best = panorama.RANSAC(match,100)
		inliers_image = panorama.draw_matches(img1,img2,filtered_matches)
		panorama.plot_image(inliers_image,f"RANSAC{i+1}{j+1}")
		cv2.imwrite(f'inliers{i+1}{j+1}.png',inliers_image)

    # 
	# Image Warping + Blending
	# Save Panorama output as mypano.png
	# 

	Image_stitcher = cv2.Stitcher_create()

	# for i,j in ids:
	# 	img1 = Images[i].copy()
	# 	img2 = Images[j].copy()
	# 	err,image_stitched = panorama.Blending(img1,img2,Image_stitcher)
	# 	if not err:
	# 		panorama.plot_image(image_stitched,f"stitch{i+1}{j+1}")
	# 		cv2.imwrite(f'stitch{i+1}{j+1}.png',image_stitched)

	err,stitched_image = Image_stitcher.stitch(Images)
	panorama.plot_image(stitched_image,f"mypano")
	cv2.imwrite('mypano.png',stitched_image)

if __name__ == "__main__":
    main()
