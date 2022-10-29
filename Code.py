import cv2
import numpy as npy
import face_recognition
import os
from datetime import datetime

path = 'CandidatePhotos'
imageArray = []
namesWithoutExt = []
myList = os.listdir(path)
print(myList)
# just to ensure we have the List available

def encodeImg(imageArray):
    tempArray=[]
    for image in imageArray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Converting images to RGB in above step because HOG (Histogram of Oriented Gradients) Method needs RGB image to find shadows and lights.
        encoded = face_recognition.face_encodings(image)[0]
        tempArray.append(encoded)
    return tempArray
# Encoding images in the above function

for name in myList:
    temp= cv2.imread(f'{path}/{name}')
    imageArray.append(temp)
    namesWithoutExt.append(os.path.splitext(name)[0])
print(namesWithoutExt)
# To ensure we have correct names

# In the above array traversal we have removed the Image file extension


def attendance(name):
    with open('Attendance_File.csv','r+') as file:
        # using r+ as an argument for file modification to give Read and Write permission both
        names = []
        dataArray=file.readlines()
        # print(dataArray)
        for individual in dataArray:
            seperate=individual.split(',')
            names.append(seperate[0])
        if name not in names:
            now=datetime.now()
            date = now.strftime('%H:%M:%S')
            # formatting the date as Hours, Minutes, and Second
            file.writelines(f'\n{name},{date}')
# In the above function we are giving read and write permission both:
# Write, because we need to write names of candidates present
# Read, because we need to check if a candidate's name has already been recorded it won't be repeated again.

encodeList=encodeImg(imageArray)
print('Encoding Successful')

cap=cv2.VideoCapture(0)
# Using front camera

while True:
    success, img=cap.read()
    imgSource=cv2.resize(img,(0,0),None,0.25,0.25)
    # Resizing the image because earlier we shrunk the image to reduce the complexity of calculations
    imgSource = cv2.cvtColor(imgSource, cv2.COLOR_BGR2RGB)

    imgFrame = face_recognition.face_locations(imgSource)
    encodeImgFrame = face_recognition.face_encodings(imgSource,imgFrame)

    for encodeFace,faceLoc in zip(encodeImgFrame,imgFrame):
        similar=face_recognition.compare_faces(encodeList,encodeFace)
        faceDis=face_recognition.face_distance(encodeList,encodeFace)
        #print(faceDis)
        matchIndex=npy.argmin(faceDis)

        if similar[matchIndex]:
            name=namesWithoutExt[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            attendance(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)

# faceLoc=face_recognition.face_locations(imgElon)[0]
# encodeElon=face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#
# faceLocTest=face_recognition.face_locations(imgTest)[0]
# encodeTest=face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
#
# results=face_recognition.compare_faces([encodeElon],encodeTest)
# faceDis=face_recognition.face_distance([encodeElon],encodeTest)
