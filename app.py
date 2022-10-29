from json import load
import streamlit as st
from fastai.vision.all import * 
import pathlib
import platform
import plotly.express as px 

plt=platform.system()
if plt == 'Linux': pathlib.WindowsPath=pathlib.PosixPath


st.title("Mevalarni klassificatsiya qiluvchi model ")

#rasm joylash 
file= st.file_uploader('Rasm yuklash ',type=['png','jpeg','gif','svg'])
 
if file:
    st.image(file)
# Pil convert 

    img=PILImage.create(file)
    #model
    model = load_learner('fruits_model.pkl')

    pred, pred_id, probs =model.predict(img)
    st.success(pred)
    st.info(f'Ehtimollik {probs[pred_id]*100:.2f}')

    fig=px.bar(x=probs*100 ,y=model.dls.vocab)
    st.plotly_chart(fig)

