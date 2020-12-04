import cv2

camera = cv2.VideoCapture(0)

contador = 1
while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)

    cv2.imshow("Camera", frame)
    k = cv2.waitKey(30)
    if k == 27:
        break
    elif k == ord("s"):
        cv2.imwrite("positivas/figure"+str(contador)+".jpg", frame,)
        print("positivas/figure"+str(contador)+".jpg salva")
        contador += 1

cv2.destroyAllWindows()
camera.release()