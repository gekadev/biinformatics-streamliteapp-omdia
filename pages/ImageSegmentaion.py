import streamlit as st
from streamlit_option_menu import option_menu
import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
# define page 
st.set_page_config (
 page_title = 'Breast Cancer Segmentaion  |   ',
 layout="wide", 
)
# deine page 
st.title('Breast Cancer Segmentaion  Predictor ')
st.set_option('deprecation.showfileUploaderEncoding', False)


#load model
@st.cache(allow_output_mutation=True)
def loadModel():
    model = tf.keras.models.load_model('model/breast_tumor_segmentation_model.h5')
    return model

def get_prediction(file):
    image= Image.open(file).convert('RGB')
    image = np.array(image)/255
    image = cv2.resize(image,(128,128))
    #image = np.stack((image,)*3, axis=-1)
    image = np.expand_dims(image,axis=0)
    model=loadModel()
    predicttion= model.predict(image)
    return predicttion

content = st.container()
with content :
    box = st.container()
    file = st.file_uploader("Please Upload Image :", type=["jpg", "png",])
    # start validation 
    if file is not   None:
        result= get_prediction(file)
        st.image(result)

      
with open('css/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)    