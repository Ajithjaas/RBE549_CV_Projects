import glob 
import numpy as np  
from GetInliersRANSAC import GetInliersRANSAC
Matches = {}

for file in glob.glob("P3Data/matches/*.txt"):
    matches = np.loadtxt(file)
    key = file.split("/")[-1].split(".")[0]
    # Filtering the information 
    Matches[key] = GetInliersRANSAC(matches[:,:2],matches[:,2:4])
# Get the Fundamental Matrix from the first two images 
x1 , x2 = Matches["matches12"]

print(x1,x2)