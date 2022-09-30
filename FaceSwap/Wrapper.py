from tkinter import W
import dlib 
import cv2
import numpy as np 
from imutils import face_utils
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
            size  =Image.shape 
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
        return img,all_triangleList

    def plot_image(self,Image,name):
        cv2.imshow(name,Image)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

def main():
    swap = FaceSwap()
    Path = "/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/FaceSwap/Data/Shiva_img_3.jpeg"
    Image = cv2.imread(Path)
    face_marks,facemark_coordinates = swap.landmarks(Image)
    swap.plot_image(face_marks,"Facial Landmarks")
    tri_img,triangleList = swap.delaunayTriangulation(Image,facemark_coordinates)
    swap.plot_image(tri_img,"Delaunay Triangulation")


    # cap = cv2.VideoCapture(0)
    # while True:
    #     _,frame = cap.read() 
    #     frame = cv2.flip(frame,1)
    #     img,facemark_coordinates = swap.landmarks(frame)
    #     cv2.imshow("Face Landmarks", img)
    #     tri_img = swap.delaunayTriangulation(frame,facemark_coordinates)
    #     cv2.imshow("Delaunay Triangulation", tri_img)
    #     key = cv2.waitKey(1)
    #     if key == 27:  break
    # cap.release()
    # cv2.destroyAllWindows()

    




if __name__ == "__main__":
    main()




        
        