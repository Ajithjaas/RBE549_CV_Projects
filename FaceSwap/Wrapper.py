from importlib.util import source_hash
from multiprocessing.managers import BaseProxy
import posixpath
from re import X
import dlib 
import cv2
import numpy as np 
from scipy.interpolate import interp2d
import pry

class FaceSwap:
    def __init__(self) -> None:
        self.pred_data = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_detector = dlib.get_frontal_face_detector()
        # Indexing the detected face shapes based on : 
        #https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
        self.shapes = {"jaw":[i for i in range(17)],
                       "right_eyebrow":[i for i in range(17,22)],
                       "left_eyebrow":[i for i in range(22,27)],
                       "nose":[i for i in range(27,36)],
                    #    "nose_edge":[i for i in range(31,36)],
                       "right_eye":[i for i in range(36,42)],
                       "left_eye":[i for i in range(42,48)],
                       "outer_lips":[i for i in range(48,60)],
                       "inner_lips":[i for i in range(60,68)]
                       }
    def landmarks(self,Image):
        """Identifies and draws landmarks on a given face """
        img = Image.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray_img)
        facemark_coordinates =[] #this list contains landmark coordniates of all the faces 
        for face in faces:
            face_marks = self.pred_data(gray_img,face)
            coordinates = [] #landmark coordinates of only one face
            for n in range(0,68): 
                x,y = int(face_marks.part(n).x),int(face_marks.part(n).y )
                coordinates.append((x,y))
                cv2.circle(img,(x,y),3,(0,0,255),-1) # drawing circle at facial marks 
            coordinates = np.asarray(coordinates)
            facemark_coordinates.append(coordinates)
            for shape in self.shapes:
                #Drawing lines at the corresponding shapes 
                indices = np.array(self.shapes[shape])
                shape_connect = coordinates[indices]
                for start,end in zip(shape_connect,shape_connect[1:]):
                    cv2.line(img, start, end, (0, 255, 0), 2)
        return img,facemark_coordinates

    def delaunayTriangulation(self,Image,facemark_coordinates):
        """ Delaunay Triangulation tries the maximize the smallest angle in each triangle, we will obtain the same triangulation in both the images
        See https://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/
        """
        delaunay_color = (255, 255, 255)
        img = Image.copy()
        size = Image.shape
        r = (0, 0, size[1], size[0])
        def rect_contains(rect, point) :
            """Checks if a point is inside a rectangle"""
            if point[0] < rect[0] :return False
            elif point[1] < rect[1] :return False
            elif point[0] > rect[2] :return False
            elif point[1] > rect[3] :return False
            return True
        all_triangleList = []
        for points in facemark_coordinates:
            size = Image.shape 
            rect = (0, 0, size[1], size[0])
            subdiv = cv2.Subdiv2D(rect) 
            for p in points:
                pt = tuple([int(p[0]),int(p[1])])
                subdiv.insert(pt)
            triangleList = subdiv.getTriangleList()
            all_triangleList.append(triangleList)
            for t in triangleList :
                pt1 = (int(t[0]), int(t[1]))
                pt2 = (int(t[2]), int(t[3]))
                pt3 = (int(t[4]), int(t[5]))
                if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
                    cv2.line(img, pt1, pt2, delaunay_color, 1) #, cv2.CV_AA, 0)
                    cv2.line(img, pt2, pt3, delaunay_color, 1)#, cv2.CV_AA, 0)
                    cv2.line(img, pt3, pt1, delaunay_color, 1)#, cv2.CV_AA, 0)
            
        return img, all_triangleList

    def plot_image(self,Image,name):
        cv2.imshow(name,Image)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    def getBaryCoord(self,t):
        # The corner points of a triangle
        a = (int(t[0]), int(t[1]))
        b = (int(t[2]), int(t[3]))
        c = (int(t[4]), int(t[5]))

        # The matrix corresponding to the Face
        B_del = [[a[0], b[0], c[0]],
                [a[1], b[1], c[1]],
                [1   , 1   , 1   ]]
        
        # Forming a rectangular boundary with the corner co-ordinates of the triangle lying on its edges/sides.
        x_coord = [a[0],b[0],c[0]]
        y_coord = [a[1],b[1],c[1]]
        # print("X Coord : ",x_coord)
        # print("Y Coord : ",y_coord)
        x_min   = np.min(x_coord)
        x_max   = np.max(x_coord)
        y_min   = np.min(y_coord)
        y_max   = np.max(y_coord)
        x       = [i for i in range(x_min,x_max+1)]
        y       = [j for j in range(y_min,y_max+1)]
        X,Y     = np.meshgrid(x,y)
        X       = X.flatten()
        Y       = Y.flatten()
        #print("Traingle X:",xx)
        BBox    = np.vstack((X,Y))
        one     = np.ones(X.shape)
        BBox    = np.vstack((BBox,one))
        # print("BBox shape before:", BBox.shape)

        BCoord = np.dot(np.linalg.pinv(B_del),BBox) # Calculation of Barycentric Co-ordinates
        # print(BCoord.shape)
        alpha   = BCoord[0]
        beta    = BCoord[1]
        gamma   = BCoord[2]

        # Valid alpha index
        valid_alpha_index = np.where(np.logical_and(-0.0<=alpha, alpha<=1.0))[0]

        # Valid beta index
        valid_beta_index  = np.where(np.logical_and(-0.0<=beta, beta<=1.0))[0]

        # Valid gamma index
        valid_gamma_index = np.where(np.logical_and(-0.0<=alpha+beta+gamma, alpha+beta+gamma<=1.0))[0]

        valid_alpha_beta_index = np.intersect1d(valid_alpha_index,valid_beta_index)
        valid_points_index     = np.intersect1d(valid_alpha_beta_index,valid_gamma_index)

        TBBox       = BBox[:,valid_points_index]
        valid_alpha = alpha[valid_points_index] 
        valid_beta  = beta[valid_points_index]
        valid_gamma = gamma[valid_points_index]
        BaryCoord   = np.vstack((valid_alpha,valid_beta))
        BaryCoord   = np.vstack((BaryCoord,valid_gamma))

        return BaryCoord, TBBox, BBox


    def triangleListSort(self,all_triangleList,facemark_coordinates):
        new_row = np.zeros(6,dtype='float32')
        i = 0
        for row in all_triangleList[0]:
            for index in range(0,len(row),2):
                idx                 = np.where(np.logical_and(facemark_coordinates[0][:,0] == row[index],facemark_coordinates[0][:,1] == row[index+1]))[0][0]
                new_row[index]      = facemark_coordinates[1][idx,0]
                new_row[index+1]    = facemark_coordinates[1][idx,1]
            all_triangleList[1][i]  = new_row
            i=i+1
        return all_triangleList


    def FaceSwap(self,Img, all_triangleList,facemark_coordinates):
        img = Img.copy()
        all_triangleList = self.triangleListSort(all_triangleList,facemark_coordinates)
        for faceA,faceB in zip(all_triangleList[0],all_triangleList[1]):
            # BaryfaceA - BaryCentric Co-ordinates of FaceA
            # TriBBoxA  - Co-ordinates inside the triangle faceA
            # BBoxA     - The entire bounding box of triangle faceA 
            BaryfaceA,TriBBoxA,BBoxA    = self.getBaryCoord(faceA)
            
            # BaryfaceB - BaryCentric Co-ordinates of FaceA
            # TriBBoxB  - Co-ordinates inside the triangle faceB
            # BBoxB     - The entire bounding box of triangle faceB 
            BaryfaceB,TriBBoxB,BBoxB    = self.getBaryCoord(faceB)
            
            # if no points found then move on to next triangle
            if BaryfaceA.shape[1]==0 or BaryfaceB.shape[1]==0:
                continue
        
            # The corner points of a triangle t1
            a1 = (int(faceA[0]), int(faceA[1]))
            b1 = (int(faceA[2]), int(faceA[3]))
            c1 = (int(faceA[4]), int(faceA[5]))
            # The matrix corresponding to the Face 1
            A = [[a1[0], b1[0], c1[0]],
                 [a1[1], b1[1], c1[1]],
                 [1    , 1    , 1    ]]
            
            # The corner points of a triangle t2
            a2 = (int(faceB[0]), int(faceB[1]))
            b2 = (int(faceB[2]), int(faceB[3]))
            c2 = (int(faceB[4]), int(faceB[5]))
            # The matrix corresponding to the Face 2
            B = [[a2[0], b2[0], c2[0]],
                 [a2[1], b2[1], c2[1]],
                 [1    , 1    , 1    ]]

            # # Plotting the triangles
            # cv2.line(Img, a1, b1, (255, 255, 255), 1) #, cv2.CV_AA, 0)
            # cv2.line(Img, b1, c1, (255, 255, 255), 1)#, cv2.CV_AA, 0)
            # cv2.line(Img, c1, a1, (255, 255, 255), 1)#, cv2.CV_AA, 0)
            # cv2.line(Img, a2, b2, (255, 255, 255), 1) #, cv2.CV_AA, 0)
            # cv2.line(Img, b2, c2, (255, 255, 255), 1)#, cv2.CV_AA, 0)
            # cv2.line(Img, c2, a2, (255, 255, 255), 1)#, cv2.CV_AA, 0)
            # self.plot_image(Img,"Triangle Comparison")

            # Calculating A to be copied to face B
            WarpedA = np.dot(A,BaryfaceB)
            # Calculating B to be copied to face A
            WarpedB = np.dot(B,BaryfaceA)
          
            # Co-ordinates in Image A that need to be copied to Image B
            xAw = (WarpedA[0,:]/WarpedA[2,:]).astype(int)
            yAw = (WarpedA[1,:]/WarpedA[2,:]).astype(int)

            # Co-ordinates in Image B that need to be copied to Image A
            xBw = (WarpedB[0,:]/WarpedB[2,:]).astype(int)
            yBw = (WarpedB[1,:]/WarpedB[2,:]).astype(int)
            
            # Actual A co-ordinates inside triangle of Image A
            xA = TriBBoxA[0,:].astype(int)
            yA = TriBBoxA[1,:].astype(int)

            # Actual B co-ordinates inside triangle of Image B
            xB = TriBBoxB[0,:].astype(int)
            yB = TriBBoxB[1,:].astype(int)

            # # Real Co-ordinates
            # print('xB ' ,xB.shape)
            # print('yB ' ,yB.shape)
            # print('xA ' ,xA.shape)
            # print('yA ' ,yA.shape)
            
            # # Warped Co-ordinates
            # print('xAw ',xAw.shape)
            # print('yAw ',yAw.shape)
            # print('xBw ',xBw.shape)
            # print('yBw ',yBw.shape)

            # # Face Swapping
            # img[yA,xA,:] = Img[yBw,xBw,:]
            # img[yB,xB,:] = Img[yAw,xAw,:]      

            # Defining source rectangle for face A, from triangle B
            xSA, ySA, wSA, hSA = cv2.boundingRect(np.array([a1,b1,c1])) 
            # Defining source rectangle for face B, from triangle A
            xSB, ySB, wSB, hSB = cv2.boundingRect(np.array([a2,b2,c2]))

            SourceForA = Img[ySA:ySA+hSA,xSA:xSA+wSA,:]
            SourceForB = Img[ySB:ySB+hSB,xSB:xSB+wSB,:]

            xs = np.linspace(xSA, xSA+wSA, num=wSA, endpoint=False)
            ys = np.linspace(ySA, ySA+hSA, num=hSA, endpoint=False)
            # https://scipython.com/book/chapter-8-scipy/examples/scipyinterpolateinterp2d/
            # Copying from A to B
            interpAblue     = interp2d(xs, ys, SourceForA[:, :, 0], kind='linear')
            interpAgreen    = interp2d(xs, ys, SourceForA[:, :, 1], kind='linear')
            interpAred      = interp2d(xs, ys, SourceForA[:, :, 2], kind='linear')

            xs = np.linspace(xSB, xSB+wSB, num=wSB, endpoint=False)
            ys = np.linspace(ySB, ySB+hSB, num=hSB, endpoint=False)           
            # Copying from B to A
            interpBblue     = interp2d(xs, ys, SourceForB[:, :, 0], kind='linear')
            interpBgreen    = interp2d(xs, ys, SourceForB[:, :, 1], kind='linear')
            interpBred      = interp2d(xs, ys, SourceForB[:, :, 2], kind='linear')

            for x,y,xb,yb in zip(xAw,yAw,xB,yB):
                pxblue          = interpAblue(x,y)[0].astype(int)
                pxgreen         = interpAgreen(x,y)[0].astype(int)
                pxred           = interpAred(x,y)[0].astype(int)
                img[yb,xb,:]    = (pxblue,pxgreen,pxred)
            
            for x,y,xa,ya in zip(xBw,yBw,xA,yA):
                pxblue      = interpBblue(x,y)[0].astype(int)
                pxgreen     = interpBgreen(x,y)[0].astype(int)
                pxred       = interpBred(x,y)[0].astype(int)
                img[ya,xa,:]  = (pxblue,pxgreen,pxred)
            
        cv2.imshow('Warped Image:',img)
        cv2.waitKey(0)


def main():
    swap = FaceSwap()
    Path = "Data/Image.jpg"
    Image = cv2.imread(Path)
    face_marks,facemark_coordinates  = swap.landmarks(Image)
    swap.plot_image(face_marks,"Facial Landmarks")
    tri_img,all_triangleList = swap.delaunayTriangulation(Image,facemark_coordinates)
    swap.plot_image(tri_img,"Delaunay Triangulation")
    swap.FaceSwap(Image, all_triangleList,facemark_coordinates)


    # cap = cv2.VideoCapture(0)
    # while True:
    #     _,frame = cap.read() 
    #     frame = cv2.flip(frame,1)
    #     face_marks,facemark_coordinates  = swap.landmarks(Image)
    #     swap.plot_image(face_marks,"Facial Landmarks")
    #     tri_img,all_triangleList = swap.delaunayTriangulation(Image,facemark_coordinates)
    #     swap.plot_image(tri_img,"Delaunay Triangulation")
    #     swap.FaceSwap(Image, all_triangleList,facemark_coordinates)
    #     key = cv2.waitKey(1)
    #     if key == 27:  break
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




        
        