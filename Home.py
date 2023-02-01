import streamlit as st
from streamlit_option_menu import option_menu
import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
# define page 
st.set_page_config (
 page_title = 'Breast Cancer  | Home  ',
 layout="wide", 
)
# deine page 
st.title('Breast Cancer Detection Predictor ')
st.set_option('deprecation.showfileUploaderEncoding', False)



def get_Label(number):
    labels = {0:'Density1Benign', 1:'Density1Malignant',2:'Density2Benign',
              3:'Density2Malignant',4:'Density3Benign',5:'Density3Malignant',
              6:'Density4Benign',7:'Density4Malignant'}
    return labels[number]

#load model
@st.cache(allow_output_mutation=True)
def loadModel():
    model = tf.keras.models.load_model('model/CancerClassification.h5')
    return model

def get_prediction(file):
    image= Image.open(file).convert('RGB')
    image = np.array(image)/255
    image = cv2.resize(image,(227,227))
    #image = np.stack((image,)*3, axis=-1)
    image = np.expand_dims(image,axis=0)
    model=loadModel()
    predicttion= model.predict(image)
    predicttion=np.argmax(predicttion)
    return get_Label(predicttion)

content = st.container()
with content :
    box = st.container()
    file = st.file_uploader("Please Upload Image :", type=["jpg", "png",])
    # start validation 
    if file is not   None:
        result= get_prediction(file)
        finalresult=box.success(result)
      
with open('css/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)    