import dlib 
import cv2
import numpy as np 
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
        img = Image #.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray_img)
        for face in faces:
            face_marks = self.pred_data(gray_img,face)
            coordinates = []
            for n in range(0,68):
                x,y = face_marks.part(n).x,face_marks.part(n).y 
                coordinates.append((x,y))
                cv2.circle(img,(x,y),3,(0,0,255),-1) # drawing circle at facial marks 
            coordinates = np.asarray(coordinates)
            for shape in self.shapes:
                #Drawing lines at the corresponding shapes 
                indices = np.array(self.shapes[shape])
                shape_connect = coordinates[indices]
                for start,end in zip(shape_connect,shape_connect[1:]):
                    cv2.line(img, start, end, (0, 255, 0), 2)
        return img

    def plot_image(self,Image,name):
        cv2.imshow(name,Image)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

def main():
    swap = FaceSwap()
    # Path = "/Users/ric137k/Desktop/Shiva/WPI/Course Work/RBE:CS 549 - Computer Vision /Projects/RBE549_CV_Projects/ajayamoorthy_p2/Data/Shiva_img.jpeg"
    # Image = cv2.imread(Path)
    # face_marks = swap.landmarks(Image)
    # swap.plot_image(face_marks,"Facial Landmarks")
    cap = cv2.VideoCapture(0)
    while True:
        _,frame = cap.read()
        # Image = cv2.imread(frame)
        img = swap.landmarks(frame)
        cv2.imshow("Face Landmarks", frame)
        key = cv2.waitKey(1)
        if key == 27:  break
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()




        
        