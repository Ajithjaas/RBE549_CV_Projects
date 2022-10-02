from ctypes import resize
import dlib 
import cv2
import numpy as np 

class FaceSwap:
    def __init__(self) -> None:
        self.pred_data = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_detector = dlib.get_frontal_face_detector()
        self._lambda=0.1 

    def landmarks(self,Image):
        """Identifies and draws landmarks on a given face """
        img = Image.copy()
        features = self.face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
        pts = self.pred_data(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), features[0])        
        coordinates = np.zeros((68, 2), dtype="int") # There are 68 features 
        for i in range(0, 68):
            coordinates[i] = (pts.part(i).x, pts.part(i).y)
        return coordinates
    
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


    def TPS(self,face1,face2):
        self.face1 = face1 
        self.face2 = face2 
        self.first_features = self.landmarks(face1)
        self.second_features = self.landmarks(face2)
        def U(r):
            r[r==0] = self._lambda
            return r**2*np.log(r)
        def F(x,y,features,w):
            """TPS function.see: https://en.wikipedia.org/wiki/Thin_plate_spline"""
            point = [x,y] 
            N = features.shape[0]
            points = point*np.ones((N,2))
            norms =  np.linalg.norm(features-points, ord=2,axis=1)
            Z = U(norms)
            return  int(w[-1] + w[-3] * point[0] + w[-2] * point[1] + np.matmul(Z,w[0:-3]))

        def patch(features):
            """Returns all the coordinates in the given face area"""
            xmin = np.min(features[:,0])
            xmax = np.max(features[:,0])
            ymin = np.min(features[:,1])
            ymax = np.max(features[:,1])
            x    = np.arange(xmin, xmax)
            y    = np.arange(ymin, ymax)
            xx,yy = np.mgrid[x[0]: x[-1] + 1, y[0]: y[-1] + 1]
            center = (int((xmin+xmax)/2), int((ymin+ymax)/2))
            return np.vstack((xx.ravel(), yy.ravel())).T ,center

        p = self.second_features.shape[0] 
        Xij = np.subtract.outer(self.first_features[:,0], self.first_features[:,0])  # see: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.outer.html
        Yij = np.subtract.outer(self.first_features[:,1], self.first_features[:,1])
        K = U(np.sqrt(Xij**2 + Yij**2))
        P = np.concatenate((self.first_features, np.ones((len(self.first_features), 1))), axis=1)
        V = np.concatenate((self.second_features, np.array([[0,0], [0,0], [0,0]])), axis=0)
        K1 = np.concatenate((K, P), axis=1)
        K2 = np.concatenate((np.transpose(P), np.zeros((3,3), dtype=np.float32)), axis=1)
        K_mat = np.concatenate((K1, K2), axis=0)
        I = self._lambda*np.identity(p+3)
        weights = np.dot(np.linalg.pinv(K_mat+I), V)  #both wx and wy 

        """Now we need to use these weights and compute the corresponding Fx and Fy 
        values from the source. 
        For this first we need find the convex hull for the given dst features.
        Then need to make the values in this hull to true and rest to false. True equivalent here is white color.
        Once the mask is found , we can do a bitwise and operation with original dst image.
        See : https://learnopencv.com/seamless-cloning-using-opencv-python-cpp/"""
        masked = np.zeros((self.face2.shape[0], self.face2.shape[1]), dtype=np.uint8)
        conv_hull = cv2.convexHull(self.second_features)
        cv2.fillConvexPoly(masked, conv_hull, (255,255,255))
        masked = np.dstack((masked, masked, masked))
        img_blend = cv2.bitwise_and(self.face2, masked)
        self.plot_image(img_blend,"facial region")
        cv2.imwrite("/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/FaceSwap/TPS_Outputs/17.png",img_blend)
        coordinates,center= patch(self.first_features)
        rows, cols = self.face2.shape[:2]

        warped_img = np.zeros((rows, cols, 3), dtype=np.uint8)
        warped_mask = np.zeros((rows,cols, 3), dtype=np.uint8)
        for x,y in coordinates:
            Fx = F(x, y, self.first_features, weights[:,0])
            Fy = F(x, y, self.first_features, weights[:,1])
            if Fx>0 and Fx<cols and Fy>0 and Fy<rows:
                if img_blend[Fy, Fx, 0] != 0:
                    warped_img[y, x, :] = img_blend[Fy, Fx, :]
                    warped_mask[y, x, :] = (255,255,255)
        self.plot_image(warped_img,"Warp")
        cv2.imwrite("/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/FaceSwap/TPS_Outputs/18.png",warped_img)
        blended_img = cv2.seamlessClone(warped_img, self.face1, warped_mask, center, cv2.NORMAL_CLONE)  
        cv2.imwrite("/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/FaceSwap/TPS_Outputs/19.png",blended_img)

        return  blended_img 

    def plot_image(self,Image,name):
        cv2.imshow(name,Image)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

def main():
    face1 = cv2.imread("/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/FaceSwap/Data/Shiva_img_2.jpeg")
    face2 = cv2.imread("/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/FaceSwap/Data/Ajith.JPG")
    faceswap = FaceSwap()
    output_img = faceswap.TPS(face1,face2)
    faceswap.plot_image(face1,"Face1")
    faceswap.plot_image(face2,"Face2")
    faceswap.plot_image(output_img,"Face 2 on Face 1")

    # faceswap = FaceSwap()
    # # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/FaceSwap/Data/Shiva_rec.mp4')
    # face2 = cv2.imread("/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/FaceSwap/Data/leo.webp")
    # out = cv2.VideoWriter("/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /RBE549_CV_Projects/FaceSwap/Outputs/output.avi", -1, 20.0, (640,480))
    # while (cap.isOpened()):
    #     _,face1 = cap.read() 
    #     face1 = cv2.flip(face1,1)
    #     output_img = faceswap.TPS(face1,face2)
    #     out.write(output_img)
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()