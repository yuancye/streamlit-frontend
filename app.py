import os
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd

from utils import convert_to_original_coordinates, post_process_bbox, run_inference


def init():
    if "inference_result" not in st.session_state:
        st.session_state["inference_result"] = None

    if "inference_result_with_given_bbox" not in st.session_state:
        st.session_state["inference_result_with_given_bbox"] = None

    if "bboxes" not in st.session_state:
        st.session_state["bboxes"] = None

    if "filename" not in st.session_state:
        st.session_state["filename"] = None

def reset():
    st.session_state["inference_result"] = None
    st.session_state["inference_result_with_given_bbox"] = None
    st.session_state["bboxes"] = []


st.set_page_config(layout="wide")
st.title("Mouse Lab")

init()

uploaded_file = st.file_uploader(label="select an image", type=['jpg', 'jpeg', 'png', 'webp', 'tif', 'tiff','bmp'])
uploaded_image = False

if uploaded_file:
    if st.session_state["filename"] != uploaded_file.name:
        st.session_state["filename"] = uploaded_file.name
        reset()
    try:
        image = Image.open(uploaded_file)
        uploaded_image = True
        # not a valid image file
    except Exception as e:
        st.error("Please select a valid image file.")

if uploaded_image:   
    nms = st.slider(label="NMS", min_value=0.01, max_value=1.0, value=0.5, step=0.1, key="nms")  # Adjusted min_value to 0.01
    min_score = st.slider(label="Min_score_thresh", min_value=0.1, max_value=1.0, value=0.8, step=0.01, key="min_score_thresh")
    
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.subheader("Inference on raw image")
        st.image(image, caption="Uploaded Image")
        inference_btn = st.button("Run inference", key="inference", use_container_width=True, type="primary")
        if inference_btn:
            response = run_inference(uploaded_file, nms, min_score)
            
            inference_result = np.array(response["results"])
            if isinstance(inference_result, np.ndarray):
                st.session_state["inference_result"] = inference_result
            else:
                st.error("Inference result is not valid.")
        # display inference result      
        if st.session_state["inference_result"] is not None:
            st.image(st.session_state["inference_result"])
        else:
            st.write("No inference result available.")
                    
        
    with col2:
        st.subheader("Inference with given bboxes")

        drawing_mode = st.selectbox(
            "bbox annotation:",
            ("rect", "transform"),
            help="rect: Draw bounding boxes\n\ntransform: Double-click an object to remove it"
        )

   
        canvas_width = 800
        canvas_height = 600
        image_width, image_height = image.size

        scale_x = image_width / canvas_width
        scale_y = image_height / canvas_height

        resized_image = image.resize((canvas_width, canvas_height))

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Transparent orange
            stroke_width=2,
            stroke_color="red",
            background_image=resized_image,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
        )


        # Display bounding boxes
        if canvas_result.json_data:
            st.write("Original Bounding boxes x1, y1, x2, y2:")
            bboxes = canvas_result.json_data["objects"]
            bbox_data = [
                {
                    "x": obj["left"],
                    "y": obj["top"],
                    "width": obj["width"],
                    "height": obj["height"],
                }
                for obj in bboxes
                if obj["type"] == "rect" and obj["width"] > 10 and obj["height"] > 10
            ]          
            scaled_bbox= post_process_bbox(bbox_data)
            original_bboxes = convert_to_original_coordinates(scaled_bbox, scale_x, scale_y)
            df_bboxes = pd.DataFrame(original_bboxes, columns=['x1', 'y1', 'x2', 'y2'])
            st.dataframe(df_bboxes)
            st.session_state["bboxes"] = original_bboxes


        # disable button if there is no bbox
        if "bboxes" in st.session_state and len(st.session_state["bboxes"]) > 0:
            disable_button = False
        else:
            disable_button = True


        # inference
        inference_with_bbox_btn = st.button(label="Run inference with given bboxes", 
                                            key="reference_with_bbox", 
                                            use_container_width=True,                                       
                                            type="primary", 
                                            disabled=disable_button)
        if inference_with_bbox_btn:
            response_result =run_inference(uploaded_file, nms, min_score, st.session_state["bboxes"])
            inference_result_with_bbox = np.array(response_result["results"])
            if isinstance(inference_result_with_bbox, np.ndarray):
                st.session_state["inference_result_with_given_bbox"] = inference_result_with_bbox                 
            else:
                st.error("Inference result with given bounding boxes is not valid.")       

        # display inference result
        if st.session_state["inference_result_with_given_bbox"] is not None:
            st.image(st.session_state["inference_result_with_given_bbox"])
        else:
            st.write("No inference result available.")
            

                    

    