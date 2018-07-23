import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
from PIL import Image
import time, shutil, os

def webcamCrop(filename):
    faceCascade = cv2.CascadeClassifier("classifier/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier('classifier/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('classifier/mouth.xml')
    nose_cascade = cv2.CascadeClassifier('classifier/nariz.xml')
    log.basicConfig(filename='webcam.log',level=log.INFO)

    video_capture = cv2.VideoCapture('/Users/artrial/desktop/face_detect/face_clustering/FaceImageClustering/video/' + filename)
    anterior = 0

    if os.path.exists("DataSet"):
        shutil.rmtree("DataSet")

    os.mkdir('DataSet')

    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            break


        # Capture frame-by-frame
        ret, frame = video_capture.read()
        # print(str(frame))
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            print(str(faces) + "faces printed")
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # sub_face = frame[y:y+h, x:x+w]
                # FaceFileName = "unknownfaces/face_" + str(y) + ".jpg"
                # cv2.imwrite(FaceFileName, sub_face)
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                # mouth = mouth_cascade.detectMultiScale(roi_gray)
                # nose = nose_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    # cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(22,255,0),2)
                    sub_face = frame[y:y+h, x:x+w]
                    FaceFileName = "DataSet/face_" + str(y) + ".jpg"
                    cv2.imwrite(FaceFileName, sub_face)

            if anterior != len(faces):
                anterior = len(faces)
                log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


            # Display the resulting frame
            # cv2.imshow('Video', frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Display the resulting frame
            # cv2.imshow('Video', frame)
        else:
            video_capture.release()
            cv2.destroyAllWindows()

    # When everything is done, release the capture
    # video_capture.release()
    # cv2.destroyAllWindows()
