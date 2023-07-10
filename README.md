# Image Captioning Application

This repository contains an Image Captioning application written in Streamlit. The application uses a pretrained VGG network connected to LSTM layers to generate descriptive captions for images. The application is deployed on an AWS EC2 instance. Check this out (if the instance is running) : http://3.141.223.2:8501/

## Features

- Upload an image: The application allows users to upload their own images for caption generation.
- Generate captions: The AI model analyzes the uploaded image and generates a descriptive caption.
- View captions: The generated captions are displayed to the user, providing a textual description of the content in the image.

## Requirements

- Python (version >= 3.6)
- Streamlit (version = 1.24.1)
- Tensorflow

Note: It is recommended to use a virtual environment to manage the dependencies.

## Installation

1. Clone the repository:
   ```bash
   https://github.com/Alirezad126/Image-Captioning-App.git

2. Change to the project directory:
   ```bash
   cd Image-Captioning-App
   
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   
2. Access the application in your browser at `http://localhost:8501`.

3. Upload an image using the provided interface.

4. Wait for the model to generate a caption for the image.

5. View the generated caption displayed on the application.

## Pretrained Model

The application uses a pretrained VGG network connected to LSTM layers for generating captions. The model is trained on the Flickr30k dataset, which consists of 31,783 images with 158,915 crowd-sourced captions. The dataset provides a diverse range of images and captions to train the model for generating accurate and contextually relevant captions.

The pretrained model is included in the repository and will be automatically loaded when running the application.

## Dataset

The model is trained on the [Flickr30k dataset](https://github.com/BryanPlummer/flickr30k_entities), which provides a large collection of images and corresponding captions. The dataset is widely used for image captioning research and evaluation.
