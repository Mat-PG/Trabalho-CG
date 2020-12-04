import numpy as np
import cv2
import face_recognition
import os
import glob
import datetime

car_cascade = cv2.CascadeClassifier("treinamento/cascade.xml")

face_encodings = []
faces_names = []
currentDir = os.getcwd()
path = os.path.join(currentDir, "faces/")

lista = [f for f in glob.glob(path+"*.jpg")]
tamanhoLista = len(lista)
names = lista.copy()

for i in range(tamanhoLista):
    presets = face_recognition.load_image_file(lista[i])
    encoding = face_recognition.face_encodings(presets)[0]
    face_encodings.append(encoding)
    names[i] = names[i].replace(currentDir, "")
    names[i] = names[i].replace(".jpg", "")
    names[i] = names[i].replace("faces", "")
    faces_names.append(names[i])

face_locations = []
face_encodings = []
face_names = []

camera = cv2.VideoCapture(0)
while True:
    _,frame = camera.read()
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)
    
    HSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    minAmare = np.array([23,255,230])
    maxAmare = np.array([255,255,255])
    maskAmare = cv2.inRange(HSV,minAmare,maxAmare)
    resulAmare = cv2.bitwise_and(frame,frame,mask = maskAmare)
    cinzaAmare = cv2.cvtColor(resulAmare, cv2.COLOR_BGR2GRAY)
    _, threshAma = cv2.threshold(cinzaAmare,3,255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    threshAmaC = cv2.morphologyEx(threshAma, cv2.MORPH_CLOSE, kernel)
    
    contornAma,_ = cv2.findContours(threshAmaC, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    smallframe = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgbSmallframe = smallframe[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgbSmallframe)
    face_encodings = face_recognition.face_encodings(rgbSmallframe, face_locations)
    face_names = []
       
    for face in face_encodings:
        matches = face_recognition.compare_faces(face_encodings, face)
        name = "Desconhecido"
        face_distances = face_recognition.face_distance(face_encodings, face)
        bestMatch = np.argmin(face_distances)
        if matches[bestMatch]:
            name = faces_names[bestMatch]

            for cntA in contornAma:
                (Ax, Ay, Aw, Ah) = cv2.boundingRect(cntA)
                areaAma = cv2.contourArea(cntA)
                if areaAma > 3000:
                    cv2.putText(frame, "Amarelo Detectado", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                    cv2.drawContours(frame, contornAma, -1, (0, 0, 0), 5)

                    hora_atual = datetime.datetime.now()
                    str_index = str(hora_atual.year) + "/" + str(hora_atual.month) + "/" + str(hora_atual.day) + "_" + str(hora_atual.hour) +  ":" + str(hora_atual.minute) + ":" + str(hora_atual.second)
                    cv2.imwrite("logs/chegada "+str_index+".jpg", frame,)
                    print("Verificado")
                else:
                    print("Descohecido")

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, faces_names):
        top = top * 4
        right = right * 4
        bottom = bottom * 4
        left = left * 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0,0,255), cv2.FILLED)
        cv2.putText(frame, name, (left+6, bottom-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)


    cv2.imshow('Binariza', threshAma)
    cv2.imshow('Morph C', threshAmaC)
    cv2.imshow('RGBSmall', rgbSmallframe)
    cv2.imshow('Analise', frame)
    k = cv2.waitKey(60)
    if k==27:
        break

camera.release()
cv2.destroyAllWindows()