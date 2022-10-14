import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import mtcnn
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

detector = mtcnn.MTCNN()

f = open("names.txt","r")
data = f.readlines()
f.close()

model = load_model("models\Resnet50_best.h5")
model.summary()

val_datagen = ImageDataGenerator(rescale= 1.0 / 255.0)

img = cv2.imread("test_images\input\Arsenal.jpg")
dets = detector.detect_faces(img)


for i, d in enumerate(dets):

    x1, y1, w, h = d['box']
    xc = x1 + w//2
    yc = y1 + h//2
    if w >= h : h = w
    else: w = h
    x1 , y1 = xc - w//2 , yc - h//2
    face = img[y1:y1+h,x1:x1+w]
    face = cv2.resize(face,(224,224))
    face = np.expand_dims(face,0)
    x_batch = val_datagen.flow(face)
    prediction = model.predict(x_batch)
    
    name = data[np.argmax(prediction[0])].split(" ")[1].strip()
    print(name)
    cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),6)
    cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,0,0),2)

    cv2.putText(img, name, (x1-w//4,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 6, cv2.LINE_AA)
    cv2.putText(img, name, (x1-w//4,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)

cv2.imwrite("test_images/output/output.jpg",img)
cv2.imshow("",img)
cv2.waitKey(0)