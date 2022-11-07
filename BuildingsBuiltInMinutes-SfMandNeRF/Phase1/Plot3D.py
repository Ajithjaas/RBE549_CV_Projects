import matplotlib.pyplot as plt 

def Plot3D(X,color,label):
    ax = plt.axes(projection='3d')
    # Data for three-dimensional scattered points
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2],c=color,s=1,label=label) 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return plt 


