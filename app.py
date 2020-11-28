import cv2
import os
import numpy as np
import tensorflow as tf
from fr_utils import *
import sys
import tensorflow_addons as tfa
from keras.models import load_model
from keras_vggface.vggface import VGGFace
from keras.engine import  Model
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

def roi_e(roi_color,model):
  resized = cv2.resize(roi_color, (224,224))
  img = resized[...,::-1]    
  x_train = np.array([img])
  embedding = model.predict_on_batch(x_train)
  return embedding
  
def imag_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1) # or version=2
    preds = model.predict(x)
    return preds

def who_is_this(roi_color, database, model):
    
    encoding = roi_e(roi_color,model)

    min_dist = 100
      
    for (name, db_enc) in database.items():        
    
        dist = np.linalg.norm(encoding - db_enc)
       
        if dist < min_dist:
            min_dist = dist
            identity = name
       
    return min_dist, identity


FRmodel = VGGFace()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_database = {}

for name in os.listdir('imgs'):
	for image in os.listdir(os.path.join('imgs',name)):
		identity = os.path.splitext(os.path.basename(image))[0]
		face_database[identity] = imag_to_encoding(os.path.join('imgs',name,image), FRmodel)
        
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi = frame[y:y+h, x:x+w]
        min_dist,identity = who_is_this(roi, face_database, FRmodel)


    if min_dist < 0.7:
            cv2.putText(frame, "Face : " + identity[:-1], (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
            cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
    else:
            cv2.putText(frame, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    cv2.imshow('Face Recognition System', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()
