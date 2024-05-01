import streamlit as st
from io import StringIO
import pandas as pd
import tensorlow as tf
import librosa
import numpy as np

st.title("Birdbot")
st.subheader("Convolutional neural network model that is trained to classify audio into one of five (wow!) bird categories. ")

uploaded_file = st.file_uploader("Choose a -wav file to predict:")
st.divider()
if uploaded_file is not None:
    # Read as file-like object
    # Load as wav
    y, sr = librosa.load(uploaded_file, duration=3)
    # Calculate spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, srasr)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # mel_spec = tf.keras.utils.normalize(mel_spec)
    mel_spec = tf.expand_dims(mel_spec, axis=0)
    # Load model
    model = tf.keras.models.load_model('./model.pb')
    bird_list = ["Bewick's Wren", "Northern Mockingbird", "American Robin", "Song Sparrow", "Northern Cardinal"]
    p = model.predict(mel_spec) # np array of predictions
    ind = np.argmax(p) # index of bird
    conf = p[0][ind] / 1 # confidence value
    # Print results
    st.markdown("### Result")
    st.write("Bird: ", bird_list[ind])
    st.write("Confidence:", conf*100, '%')