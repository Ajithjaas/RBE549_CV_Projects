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
from skimage.feature import peak_local_max

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
		# Threshold for an optimal value, it may vary depending on the image.
		Image[dst>0.01*dst.max()]=[0,0,255]

		# cv2.imshow('dst',Image)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()
		return Image 

	def ANMS(self,C_Img,N_best):
		#C_Image - Corner Score Image 
		#N_best - Number of best corners needed

		"""The objective of this step is to detect corners such that 
		they are equally distributed across the image in order to avoid 
		weird artifacts in warping."""

		#Find all local maxima see: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_peak_local_max.html
		coordinates = peak_local_max(C_Img, min_distance=20) #coordinates are stored 
		r = np.zeros((coordinates.shape[0], 1), dtype=np.float32) 
		best_coordinates = np.zeros((coordinates.shape[0], 2), dtype=np.float32) #stores the best 
		r[:] = float("inf")
		index = 0 
		for i in coordinates:
			best_coordinates[i][0] = i[1]
			best_coordinates[i][1] = i[0]
			ED = float("inf")
			for j in coordinates:
				if (C_Img[j[0], j[1]] > C_Img[i[0], i[1]]):
					ED = (j[1] - i[1])**2 + (j[0] - i[0])**2
				if (ED < r[index][2]):
					r[index][0] = ED
			index+=1
		best_coordinates[np.argsort(r)]
		
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
	cv2.imshow('dst',Image)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()
	coordinates = peak_local_max(Image, min_distance=20)
	# print(coordinates.shape)



	# print(Image)

	# pano = MyPano()

	# corner_score_image = pano.corner_detection(Image,2,3)

	# print(pano.ANMS(Image,100))

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
