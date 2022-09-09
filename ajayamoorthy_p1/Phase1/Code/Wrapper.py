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
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt 
from skimage import feature 

# Add any python libraries here

class MyPano:
	def __init__(self) -> None:
		pass
	def corner_detection(self,Image,block_size,sobel_size):
		"""Using Harris Corners to detect the corners in a given image
		Based on  https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
		"""
		gray_img = np.float32(cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY))  # converting the image to gray scale and then to float32 
		dst = cv2.cornerHarris(gray_img,block_size,sobel_size,0.04)
		dst = cv2.dilate(dst,None)
		dst[dst<0.01*dst.max()] = 0  # Setting below threshold to zero 
		Image[dst!=0]=[0,0,255]  
		cv2.imshow('corners',Image)
		if cv2.waitKey(0) & 0xff == 27:
			cv2.destroyAllWindows()
		return Image, dst

	def ANMS(self,C_Img,N_best):
		#C_Image - Corner Score Image 
		#N_best - Number of best corners needed
		"""The objective of this step is to detect corners such that 
		they are equally distributed across the image in order to avoid 
		weird artifacts in warping."""
		#Find all local maxima see: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_peak_local_max.html
		coordinates = feature.peak_local_max(C_Img, min_distance=10,threshold_abs=0.1) #coordinates are stored 
		r = np.zeros((coordinates.shape[0], 1), dtype=np.float32) 
		r[:] = np.inf #float("inf")
		index = 0 
		
		for i in coordinates:
			yi,xi = i
			ED = np.inf 
			for j in coordinates:
				yj,xj =j 
				if (C_Img[yj,xj] > C_Img[yi,xi]):
					ED = (xj - xi)**2 + (yj - yi)**2
				if (ED < r[index][0]):
					r[index][0] = ED
			index+=1
		best_coordinates = coordinates[np.flip(np.argsort(r[:,0]))] #descending order 
		return best_coordinates[0:N_best,:]


def main():
	# Add any Command Line arguments here
	# Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures
	
	image_path = "Phase1/Data/Train/Set1/1.jpg"
	image_path = "/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /Projects/RBE549_CV_Projects/ajayamoorthy_p1/Phase1/Data/Train/Set1/1.jpg"
	Image = cv2.imread(image_path)
	cv2.imshow('input',Image)
	if cv2.waitKey(0) & 0xff == 27:
			cv2.destroyAllWindows()
	# Image = cv2.imread(image_path,0)

	panorama = MyPano()

	corner_score_image,dst = panorama.corner_detection(Image,2,3)

	best_coordinates = panorama.ANMS(dst,100)
	Image = cv2.imread(image_path)
	for coord in best_coordinates:
		i,j= int(coord[1]),int(coord[0])
		# Image[i][j] = [0,0,255] 
		cv2.circle(Image,(i,j),3,255, -1)

	cv2.imshow('anms',Image)
	if cv2.waitKey(0) & 0xff == 27:
			cv2.destroyAllWindows()

	# panorama.ANMS(Image,50)

    # """
    # Read a set of images for Panorama stitching
    # """

    # """
	# Corner Detection
	# Save Corner detection output as corners.png
	# """

    # """
	# Perform ANMS: Adaptive Non-Maximal Suppression
	# Save ANMS output as anms.png
	# """

    # """
	# Feature Descriptors
	# Save Feature Descriptor output as FD.png
	# """

    # """
	# Feature Matching
	# Save Feature Matching output as matching.png
	# """

    # """
	# Refine: RANSAC, Estimate Homography
	# """

    # """
	# Image Warping + Blending
	# Save Panorama output as mypano.png
	# """
	
	


if __name__ == "__main__":
    main()
