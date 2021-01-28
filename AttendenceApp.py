import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime
path="ImageAttendance"
images=[]
className=[]
mylist=os.listdir(path)
for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])
def markAttendance(name):
    print('atif reyaz khan')
    with open('attendence.csv','r+') as f:
        myDataList=f.readlines()
        namelist=[] 
        for line in myDataList:
            entry=line.split(',')
            print(entry)
            namelist.append(entry[0])
        print(namelist)
        if name not in namelist:
            now=datetime.now()
            print(now)
            dstring=now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dstring}')


def findencoding(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown=findencoding(images)
print(len(encodeListKnown))
cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)
    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex=np.argmin(faceDis)
        print(faceDis,className[matchIndex])
        if matches[matchIndex]:
            name=className[matchIndex].upper()
            
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    cv2.imshow('webcamb',img)
    cv2.waitKey(1)  
 # known_image=face_recognition.load_image_file("sA.jpg")

# biden_endcoding=face_recognition.face_encodings(known_image)[0]
# unkown_endcoding=face_recognition.face_encodings(unkown_image)[0]
# result=face_recognition.compare_faces([biden_endcoding],unkown_endcoding)
# print(result) 
# if(result[0] ==True):
#     print("smae")
# else:
#     print("differendt")
