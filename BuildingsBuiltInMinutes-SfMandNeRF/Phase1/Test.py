# import glob 
# import numpy as np  
# from GetInliersRANSAC import GetInliersRANSAC
# Matches = {}

# for file in glob.glob("P3Data/matches/*.txt"):
#     matches = np.loadtxt(file)
#     key = file.split("/")[-1].split(".")[0]
#     # Filtering the information 
#     Matches[key] = GetInliersRANSAC(matches[:,:2],matches[:,2:4])
# # Get the Fundamental Matrix from the first two images 
# x1 , x2 = Matches["matches12"]

# print(x1,x2)

import numpy as np 
match1 = np.loadtxt('P3Data/matches/matches12.txt')
x12 = match1[:,:2]
x21 = match1[:,2:4]

match1 = np.loadtxt('P3Data/matches/matches13.txt')
x13 = match1[:,:2]
x31 = match1[:,2:4]

X= np.random.randint(100,size=(x12.shape[0],3))

indices_x13 = [] 

indices_x12 = [] 

for i in x13:

    ind = np.argwhere(i==x12)

    if len(ind)>0:
        indices_x12.append(ind[0][0])
        indices_x13.append(np.argwhere(i==x13)[0][0])

print(np.hstack((X[indices_x12] , x31[indices_x13])))
print("X12 ind", len(indices_x12))

print("X13 ind", len(indices_x13))



