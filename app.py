import streamlit as st
from PIL import Image
import base64
import tensorflow as tf
from model.vgg import *
from model.model import *
from utilities.generator import *
import pickle

st.set_page_config(page_title="Image Captioning CNNtoLSTM", page_icon="📷", layout="wide")

#create the model:
vocab_size = 18313
max_length = 74

model = CNNtoRNN(vocab_size, max_length)
model.load_weights("model_epoch_33.h5")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.jpg')

# Set the title and description of your app
st.title("Image Captioning App")
st.markdown("Upload an image and generate a caption.")

# Define the desired size for the image display
image_size = (400, 400)

# Upload image file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image on the left side
    image = Image.open(uploaded_file)
    image.thumbnail(image_size)
    st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption for the uploaded image
    caption = predict_caption(model, uploaded_file, tokenizer, max_length)
    st.write("### Generated Caption:")
    st.write(f"<div style='font-size: 24px;'>{caption}</div>", unsafe_allow_html=True)

