import streamlit as st
from preprocessing import preprocess
from joblib import dump, load
import numpy as np
from PIL import Image
st.header('CIFAR-10 classification')
st.write('Theofilus Arkhi Susanto')
st.write("Reference: https://medium.com/geekculture/image-classifier-with-streamlit-887fc186f60")

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
im = None
if upload is not None:
    op = Image.open(upload)
    im= preprocess(np.asarray(op))
    c1.header('Input Image')
    c1.image(op,width="stretch")

if im is not None:
    model_loaded = load("xgb.joblib")
    pred = model_loaded.predict(np.array([im]))
    c2.header('Output')
    c2.subheader('Predicted class :')
    c2.write(classes[pred[0]])