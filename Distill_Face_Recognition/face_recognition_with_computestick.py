import cv2
import numpy as np
import os
import random
import sys
import argparse
import time 

#decide which model to inference, teacher or student
parser = argparse.ArgumentParser()
parser.add_argument("--model",help="print 'student' or 'teacher'",default="student")
args = parser.parse_args()



def face_recognise(image):
    blob = cv2.dnn.blobFromImage(image, 
                    1/255, 
                    (64, 64), 
                    [0,0,0], 
                    1, 
                    crop=False)
    net.setInput(blob)
    out = net.forward(args.model + '/output')
    if out[0][0]<out[0][1]:      #same as argmax(index) return the bigger one, [0,1] is my face, [1,0] is other's
        return "Yes,my face"
    else:
        return "No,other face"

if __name__ == '__main__':
    size = 64
    net_detect = cv2.dnn.Net_readFromModelOptimizer('./face-detection-retail-0004.xml', './face-detection-retail-0004.bin')
    net = cv2.dnn.Net_readFromModelOptimizer('./' + args.model + '/inference_graph.xml', './' + args.model + '/inference_graph.bin')
    net_detect.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    prev_frame_time = 0
    new_frame_time = 0
    cap = cv2.VideoCapture(0)                  
    while True:
        _, img = cap.read()                                    #open camera to capture images

        initial_w = cap.get(3)
        initial_h = cap.get(4)
        in_frame = cv2.resize(img, (300, 300))

        in_blob = cv2.dnn.blobFromImage(in_frame, 
                        1, 
                        (300, 300), 
                        [0,0,0], 
                        1, 
                        crop=False)
        net_detect.setInput(in_blob)
        out_detect = net_detect.forward()

        for obj in out_detect[0][0]:
            if obj[2] > 0.5:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                face = img[ymin:ymax, xmin:xmax]
                if face.any():
                    face = cv2.resize(face, (size, size))
                    print('It recognizes my face? %s' % face_recognise(face))
                    if face_recognise(face) == "Yes,my face":
                        cv2.putText(img, 'Yes,my face', (xmax, ymax), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(img, 'No,other face', (xmax, ymax), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)

        cv2.imshow('image', img)
        key = cv2.waitKey(30)
        if key == 27:
            sys.exit(0)
    sess.close()

