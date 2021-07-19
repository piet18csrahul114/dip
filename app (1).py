import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
st.set_option('deprecation.showfileUploaderEncoding', False)
from keras.models import load_model

html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">End Term Digital Image Processing lab</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
         Mathematical Operations on Images
         """
         )
file= st.file_uploader("Please upload image", type=("jpg", "png"))
Direction = st.selectbox("Direction",("X","Y"))
Transformation = st.selectbox("Shearing",("Shearing", "Scaling", "Translation", "Reflection"))
Y_factor =  st.number_input('X_Factor')
X_Factor =  st.number_input('Y_Factor')

import cv2
from  PIL import Image, ImageOps
def import_and_predict(img_T,Direction,Transformation,X_Factor,Y_factor,img1,scaled_img,sheared_img,reflected_img):
  
  image = cv.cvtColor(img_T, cv.COLOR_BGR2RGB)
  
  if Transformation == "Translation": 
    if Direction == "Y":
      M = np.float32([[1, 0, 0], 
                  [0, 1, Y_factor], 
                  [0, 0, 1]])
      img1 = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    else:
      M = np.float32([[1, 0, X_Factor], 
                  [0, 1, 0], 
                  [0, 0, 1]])
      img1 = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]))
  st.image(img1, use_column_width=True)


  if Transformation == "Shearing": 
    rows, cols, dim = image.shape
    if Direction == "Y":
      M1 = np.float32([[1, 0, 0],
                [Y_factor, 1  , 0],
                [0, 0  , 1]])
      sheared_img = cv.warpPerspective(image,M1,(int(cols*1.5),int(rows*1.5)))
    else:
      M1 = np.float32([[1, X_Factor, 0],
                [0, 1  , 0],
                [0, 0  , 1]])
      sheared_img = cv.warpPerspective(image,M1,(int(cols*1.5),int(rows*1.5)))
  st.image(sheared_img, use_column_width=True)

  if Transformation == "Scaling": 
    rows, cols, dim = image.shape
    if Direction == "Y":
      M = np.float32([[1, 0  , 0],
                [0,   Y_factor, 0],
                [0,   0,   1]])
      scaled_img = cv.warpPerspective(image,M,(cols,rows))
    else:
      M = np.float32([[X_Factor, 0  , 0],
                [0,   1, 0],
                [0,   0,   1]])
      scaled_img = cv.warpPerspective(image,M,(cols,rows))
  st.image(scaled_img, use_column_width=True)

  if Transformation == "Reflection":
    rows, cols, dim = image.shape
    if Direction == "Y":
      M = np.float32([[-1,  0, cols],
                  [0, 1, 0],
                  [0,  0, 1   ]])
      reflected_img = cv.warpPerspective(image,M,(int(cols),int(rows)))
    else:
      M = np.float32([[1,  0, 0],
                  [0, -1, rows],
                  [0,  0, 1   ]])
      reflected_img = cv.warpPerspective(image,M,(int(cols),int(rows)))
  st.image(reflected_img, use_column_width=True)
  return 0

if file is None:
  st.text("Please upload an Image file")
else:
  file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  img_T = cv2.imdecode(file_bytes, 1)
  image = cv2.imdecode(file_bytes, 1)
  scaled_img = cv2.imdecode(file_bytes, 1)
  sheared_img = cv2.imdecode(file_bytes, 1)
  reflected_img = cv2.imdecode(file_bytes, 1)
  img1 = cv2.imdecode(file_bytes, 1)
  st.image(file,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Perform"):
  result=import_and_predict(img_T,Direction,Transformation,X_Factor,Y_factor,img1,scaled_img,sheared_img,reflected_img)
  
if st.button("About"):
  st.header("Rahul Chhablani")
  st.subheader("C-Section ,PIET")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Digital Image processing </p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)