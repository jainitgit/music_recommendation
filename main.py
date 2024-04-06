import cv2
from keras.models import model_from_json
import numpy as np
import keras
# with open("emotiondetector.json", "r") as json_file:
#     # Read the JSON content
#     model_json = json_file.read()
# model = model_from_json(model_json)
# model.load("wieght_bias.h5")

model = keras.models.load_model("emotion_Detection_model.keras")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_feature(image):
    feature = np.array(image)

    feature.reshape(1,48,48,1)
    return feature/255.0

webcam = cv2.VideoCapture(0)
# labels = {0:"angry", }

while True:
    i, im  = webcam.read()
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    try:
        for (p,q,r,s) in faces:
            # image = gray[q:q+s,p:p+r]
            image = im[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image, (48,48))
            image = image.reshape(1,48,48,3)
            label = np.argmax(model.predict(image))
            cv2.putText(im, '% s' %(label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
        cv2.imshow("Output",im)
        cv2.waitKey(27)
    except cv2.error:
        pass