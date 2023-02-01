import streamlit as st
from streamlit_option_menu import option_menu
import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import joblib
import pandas as pd



# from  datasist.structdata import detect_outliers
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# Pre Processing
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Regressors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
# Error Metrics 

from sklearn.ensemble import RandomForestClassifier
#crossvalidation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

# define page 
st.set_page_config (
 page_title = 'Breast Cancer  | AI Model  ',
 layout="wide", 
)
# deine page 
st.title('Breast Cancer Detection Predictor ')
st.set_option('deprecation.showfileUploaderEncoding', False)

st.markdown('''
This is a dashboard For Predection of Breast Cancer Detection Using AI  Model **Omdina** Local Chapter  

''')

# load model
#model = joblib.load('model/breastcancermodel.pkl')


dfContainer = st.container()
mainContainer = st.container()
with mainContainer :
     boxOne ,BoxTow = st.columns(2)
     radius_mean=boxOne.slider('radius_mean', min_value=0.09, max_value=1000.0, value=0.05,key=1)
     texture_mean=boxOne.slider('texture_mean', min_value=0.09, max_value=1000.0, value=0.05,key=2)
     perimeter_mean=boxOne.slider('perimeter_mean', min_value=0.09, max_value=1000.0, value=0.05,key=3)
     area_mean=boxOne.slider('area_mean', min_value=0.09, max_value=1000.0, value=0.05,key=4)
     smoothness_mean=boxOne.slider('smoothness_mean', min_value=0.09, max_value=1000.0, value=0.05,key=5)
     compactness_mean=boxOne.slider('compactness_mean', min_value=0.09, max_value=1000.0, value=0.05,key=6)
     concavity_mean=boxOne.slider('concavity_mean', min_value=0.09, max_value=1000.0, value=0.05,key=7)
     concavepoints_mean=boxOne.slider('concavepoints_mean', min_value=0.09, max_value=1000.0, value=0.05,key=8)
     symmetry_mean=boxOne.slider('symmetry_mean', min_value=0.09, max_value=1000.0, value=0.5,key=9)
     fractal_dimension_mean=boxOne.slider('fractal_dimension_mean', min_value=0.09, max_value=1000.0, value=0.05,key=10)
     radius_se=boxOne.slider('radius_se', min_value=0.09, max_value=1000.0, value=0.05,key=11)
     texture_se=boxOne.slider('texture_se', min_value=0.09, max_value=1000.0, value=0.5,key=12)
     perimeter_se=boxOne.slider('perimeter_se', min_value=0.09, max_value=1000.0, value=0.05,key=13)
     area_se=boxOne.slider('area_se', min_value=0.09, max_value=1000.0, value=0.05,key=14)
     smoothness_se=boxOne.slider('smoothness_se', min_value=0.09, max_value=1000.0, value=0.05,key=15)
     compactness_se=BoxTow.slider('compactness_se',min_value=0.09, max_value=1000.0, value=0.05,key=16)
     concavity_se=BoxTow.slider('concavity_se',min_value=0.09, max_value=1000.0, value=0.05,key=17)
     concavepoints_se=BoxTow.slider('concavepoints_se',min_value=0.09, max_value=1000.0, value=0.05,key=18)
     symmetry_se=BoxTow.slider('symmetry_se',min_value=0.09, max_value=1000.0, value=0.05,key=19)
     fractal_dimension_se=BoxTow.slider('fractal_dimension_se',min_value=0.09, max_value=1000.0, value=0.05,key=20)
     radius_worst=BoxTow.slider('radius_worst',min_value=0.09, max_value=1000.0, value=0.05,key=21)
     texture_worst=BoxTow.slider('texture_worst',min_value=0.09, max_value=1000.0, value=0.05,key=22)
     perimeter_worst=BoxTow.slider('perimeter_worst',min_value=0.09, max_value=1000.0, value=0.05,key=23)
     area_worst=BoxTow.slider('area_worst',min_value=0.09, max_value=1000.0, value=0.05,key=24)
     smoothness_worst=BoxTow.slider('smoothness_worst',min_value=0.09, max_value=1000.0, value=0.05,key=25)
     compactness_worst=BoxTow.slider('compactness_worst',min_value=0.09, max_value=1000.0, value=0.05,key=26)
     concavity_worst=BoxTow.slider('concavity_worst',min_value=0.09, max_value=1000.0, value=0.05,key=27)
     concavepoints_worst=BoxTow.slider('concavepoints_worst',min_value=0.09, max_value=1000.0, value=0.05,key=28)
     symmetry_worst=BoxTow.slider('symmetry_worst',min_value=0.09, max_value=1000.0, value=0.05,key=29)
     fractal_dimension_worst=BoxTow.slider('fractal_dimension_worst',min_value=0.09, max_value=1000.0, value=0.05,key=30)



# start programing 
#tack inpput and convert it to data frame to display
data = {'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concavepoints_mean': concavepoints_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'radius_se': radius_se,
        'radius_se': radius_se,
        'texture_se':texture_se,
        'perimeter_se':perimeter_se,
        'area_se'  :area_se,
        'smoothness_se':smoothness_se,
        'compactness_se':compactness_se,
        'concavity_se':concavity_se,
        'concavepoints_se':concavepoints_se,
        'symmetry_se' :symmetry_se,
        'fractal_dimension_se':fractal_dimension_se,
        'radius_worst':radius_worst,
        'texture_worst':texture_worst,
        'perimeter_worst':perimeter_worst,
        'area_worst':area_worst,
        'smoothness_worst':smoothness_worst,
        'compactness_worst':compactness_worst,
        'concavity_worst':concavity_worst,
        'concavepoints_worst':concavepoints_worst,
        'symmetry_worst':symmetry_worst,
        'fractal_dimension_worst':fractal_dimension_worst
            }
#get data as data frame 
df = pd.DataFrame(data, index=[0]) 
# Print specified input parameters
dfContainer.header('Input Data to Model')
dfContainer.write(df)  
dfContainer.write('---')
st.session_state['data'] =df
submit = st.button("Predict")
@st.cache(allow_output_mutation=True)
def getModel():
    return joblib.load('model/breastcancermodel.pkl')
def get_Label(number):
    labels = {1:'Malignant', 0:'Benign'}             
    return labels[number]

if submit:
    model = getModel()
    inputdata=np.asarray(st.session_state['data']).reshape(1,-1)
    sc=StandardScaler()
    x_scaled = sc.fit_transform(inputdata)
    prediction = model.predict(x_scaled)
    st.success(get_Label(prediction[0]))
    




    

#d= isinstance(x, float)
