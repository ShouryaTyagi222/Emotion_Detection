import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from src.utils import *

def main(args):
    emotion_model,_,emotion_dict=model_load(args.model_path)
    frame=cv2.imread(args.input_file)
    face_detector=cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    num_faces=face_detector.detectMultiScale(gray_frame,scaleFactor=1.3,minNeighbors=5)
    for (x,y,w,h) in num_faces:
        cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(0,255,0),4)
        roi_gray_frame=gray_frame[y:y+h,x:x+h]
        cropped_img=np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame,(48,48)),-1),0)

    emotion_prediction=emotion_model.predict(cropped_img)
    maxindex=int(np.argmax(emotion_prediction))
    print(emotion_dict[maxindex])
    cv2.putText(frame,emotion_dict[maxindex],(x+20,y+20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_8)

    plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

def parse_args():
    parser = argparse.ArgumentParser(description="Train Generator", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--img_path", type=str, default=None, help="Path to the input image")
    parser.add_argument("-m", "--model_path", type=str, default=None, help="To Resume the Training")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)