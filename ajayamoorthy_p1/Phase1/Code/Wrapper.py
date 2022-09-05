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

		# Image[dst>0.01 *dst.max()] = [0,0,255]  #Giving a color for corners
		return dst

		






def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""

    """
	Refine: RANSAC, Estimate Homography
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
