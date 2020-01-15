# Importing model 
import numpy as np
import pickle
with open('model.pkl', 'rb') as file:  
    model = pickle.load(file)
    
# Importing opencv
import cv2 as cv
img = cv.imread('seg_pred/seg_pred/173.jpg')
img = cv.resize(img,(150,150))
img = np.expand_dims(img, axis=0)
a = []
a.append(model.predict_classes(img, batch_size=10))
