import pickle
import numpy as np
import streamlit as st

pipe = pickle.load(open('/Users/prabalsingh/projects/programming_workspace/python/laptop-price-predictor/pipe.pkl','rb'))
df = pickle.load(open('/Users/prabalsingh/projects/programming_workspace/python/laptop-price-predictor/df.pkl','rb'))

st.title("Laptop Price Predictor")

company = st.selectbox('Brand',df['Company'].unique())

type = st.selectbox('Type',df['TypeName'].unique())

ram = st.selectbox('RAM (in GB) ',[2,4,6,8,12,16,24,32,64])

weight = st.number_input('Weight (in KGs)')

touchscreen = st.selectbox('Touchscreen',['Yes','No'])

ips = st.selectbox('IPS',['Yes','No'])

screensize = st.number_input('Screen Size')

resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

cpu = st.selectbox('CPU',df['CpuProcessor'].unique())

hdd = st.selectbox('HDD (in GB)',[0,128,256,512,768,1024,2048])

ssd = st.selectbox('SSD (in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

os = st.selectbox('OS',df['OpSys'].unique())

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5/screensize

    query = np.array([company,type,ram,os,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu])
    query = query.reshape(1,12)
    st.title("The predicted price for this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

