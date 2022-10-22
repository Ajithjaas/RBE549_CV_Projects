from plyfile import PlyData, PlyElement
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Resizing and combining the image
# path1 = '/home/ajith/Documents/git_repos/RBE549_CV_Projects/FaceSwap/'
# path2 = '/home/ajith/Documents/git_repos/RBE549_CV_Projects/FaceSwap/'

# face1 = cv2.imread(path1+'leo.webp')
# face2 = cv2.imread(path2+'Margot.webp')

# dim = (320, 320)

# face1_resized = cv2.resize(face1, dim, interpolation = cv2.INTER_AREA)
# face2_resized = cv2.resize(face2, dim, interpolation = cv2.INTER_AREA)

# img = np.hstack((face1_resized,face2_resized))
# cv2.imwrite(path1+'Image.jpg', img)


# Bisualization of 3D landmarks
path = '/home/ajith/Documents/git_repos/RBE549_CV_Projects/FaceSwap/Network_Outputs/'
lm1 = np.loadtxt(path+"Image_0.txt", dtype='i', delimiter=' ')
lm2 = np.loadtxt(path+"Image_1.txt", dtype='i', delimiter=' ')

x1 = lm1[0,:]
y1 = lm1[1,:]
z1 = lm1[2,:]

x2 = lm2[0,:]
y2 = lm2[1,:]
z2 = lm2[2,:]

fig = plt.figure(1)
ax = plt.axes(projection='3d')
# Data for three-dimensional scattered points
ax.scatter3D(x1, y1, z1, c=z1, cmap='winter')
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")


fig = plt.figure(2)
ax = plt.axes(projection='3d')
# Data for three-dimensional scattered points
ax.scatter3D(x2, y2, z2, c=z2, cmap='winter')
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
plt.show()



# # Visualizing point cloud file
# plydata1 = PlyData.read('/home/ajith/Documents/git_repos/RBE549_CV_Projects/FaceSwap/3DDFA/samples/test1_0.ply')
# plydata2 = PlyData.read('/home/ajith/Documents/git_repos/RBE549_CV_Projects/FaceSwap/3DDFA/samples/test1_1.ply')

# x1 = plydata1['vertex']['x']
# y1 = plydata1['vertex']['y']
# z1 = plydata1['vertex']['z']

# x2 = plydata2['vertex']['x']
# y2 = plydata2['vertex']['y']
# z2 = plydata2['vertex']['z']

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# # Data for three-dimensional scattered points
# ax.scatter3D(x1, y1, z1, c=z1, cmap='gray')
# ax.set_xlabel("x-axis")
# ax.set_ylabel("y-axis")
# ax.set_zlabel("z-axis")
# plt.show()