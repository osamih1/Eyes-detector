import os
import matplotlib.pyplot as plt
import cv2

data_dir = "./data"
models_dir = "./models"

face_detector = cv2.CascadeClassifier(os.path.join(models_dir, "haarcascade_frontalface_default.xml"))
eyes_detector = cv2.CascadeClassifier(os.path.join(models_dir, "haarcascade_eye_tree_eyeglasses.xml"))

for img_path in os.listdir(data_dir):
    print(img_path)
    img = cv2.imread(os.path.join(data_dir,img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img_gray, minNeighbors=20)

    plt.figure()
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)

        factor = 0.5
        face_ = img_gray[y:y+h, x:x+w]
        eyes = eyes_detector.detectMultiScale(cv2.resize(face_, (int(factor*w), int(factor*h))))
        for eye in eyes:
            eye = [int(e/factor) for e in eye]
            xe, ye, we, he = eye
            cv2.rectangle(img, (x+xe,y+ye), (x+xe+we,y+ye+he), (0,255,0), 10)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)

plt.show()