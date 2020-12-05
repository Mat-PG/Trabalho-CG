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
    minAmare = np.array([19,203,143])
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
            print(name)

            for cntA in contornAma:
                (Ax, Ay, Aw, Ah) = cv2.boundingRect(cntA)
                areaAma = cv2.contourArea(cntA)
                if areaAma > 3000:
                    cv2.putText(frame, "Amarelo Detectado", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                    cv2.drawContours(frame, contornAma, -1, (0, 0, 0), 5)

                    hora_atual = datetime.datetime.now()
                    str_index = str(hora_atual.year) + "_" + str(hora_atual.month) + "_" + str(hora_atual.day) + "_" + str(hora_atual.hour) +  "_" + str(hora_atual.minute) + "_" + str(hora_atual.second)
                    print(str_index)
                    cv2.imwrite("logs/log "+str_index+".jpg", frame)
                    print("Verificado")

        face_names.append(name)

    cv2.imshow('Binariza', threshAmaC)
    cv2.imshow('RGBSmall', rgbSmallframe)
    cv2.imshow('Analise', frame)
    k = cv2.waitKey(60)
    if k==27:
        break

camera.release()
cv2.destroyAllWindows()