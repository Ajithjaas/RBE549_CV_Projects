import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation 
import numpy as np

def Plot3D(X,color,label):
    ax = plt.axes(projection='3d')
    # Data for three-dimensional scattered points
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2],c=color,s=1,label=label) 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return plt 

def DrawCameras(R, C,plt,ax,label):
    angle = Rotation.from_matrix(R).as_euler("XYZ")
    angle = np.rad2deg(angle)
    plt.plot(C[0],C[2],marker=(3, 0, int(angle[1])),markersize=15,linestyle='None')  # we are seeing in XZ plane , so rotation w.r.t Y axis 
    corr = -0.1 # adds a little correction to put annotation in marker's centrum
    ax.annotate(label,xy=(C[0]+corr,C[2]+corr))



