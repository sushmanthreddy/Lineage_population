import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import joblib

import numpy as np
import cv2
import onnxruntime as ort
import imutils
# import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def scale_model_outputs(scaler_path, data):
    scaler= joblib.load(scaler_path)
    scaled=scaler.inverse_transform(data)
    return(scaled)


def onnx_predict_lineage_population(input_image):
    ort_session = ort.InferenceSession('onnx_models/lineage_population_model.onnx')
    img = Image.fromarray(np.uint8(input_image))
    resized = img.resize((256, 256), Image.NEAREST)

    transposed=np.transpose(resized, (2, 1, 0))  
    img_unsqueeze = expand_dims(transposed)

    onnx_outputs = ort_session.run(None, {'input': img_unsqueeze.astype('float32')}) 
    return(onnx_outputs[0])



def expand_dims(arr):
    norm=(arr-np.min(arr))/(np.max(arr)-np.min(arr)) #normalize
    ret = np.expand_dims(norm, axis=0)
    return(ret)



def lineage_population_model():
    selected_box2 = st.sidebar.selectbox(
    'Choose Example Input',
    (['Example_1.png'])
    )

    st.title('Predict Cell Lineage Populations')
    instructions = """
        Predict the population of cells in C. elegans embryo using fluorescence microscopy data. \n
        Either upload your own image or select from the sidebar to get a preconfigured image. 
        The image you select or upload will be fed through the Deep Neural Network in real-time 
        and the output will be displayed to the screen.
        """
    st.text(instructions)
    file = st.file_uploader('Upload an image or choose an example')
    example_image = Image.open('./images/lineage_population_examples/'+selected_box2).convert("RGB")

    col1, col2= st.beta_columns(2)

    if file:
        input = Image.open(file).convert("RGB")
        fig1 = px.imshow(input, binary_string=True, labels=dict(x="Input Image"))
        fig1.update(layout_coloraxis_showscale=False)
        fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        col1.plotly_chart(fig1, use_container_width=True)
    else:
        input = example_image
        fig1 = px.imshow(input, binary_string=True, labels=dict(x="Input Image"))
        fig1.update(layout_coloraxis_showscale=False)
        fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        col1.plotly_chart(fig1, use_container_width=True)

    pressed = st.button('Run')
    if pressed:
        st.empty()
        output = onnx_predict_lineage_population(np.array(input))
        scaled_output = scale_model_outputs(scaler_path="./scaler.gz", data=output)

        for i in range(len(scaled_output[0])):
            scaled_output[0][i]=int(round(scaled_output[0][i]))

        df = pd.DataFrame({"Lineage":["A", "E", "M", "P", "C", "D", "Z"] , "Population": scaled_output[0]})
        col2.table(df)