import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Location Image Classifier")
st.text("Provide URL of Location Image for image classification")

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('/app/models/')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes = ['with_mask','without_mask']
#image segmentation function
def segment_image(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output/255
#sharpen the image
def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

# function to get an image
def read_img(content, size):
    img = image.load_img(content, target_size=size)
    #convert image to array
    img = image.img_to_array(img)
    return img

def decode_img(image):
  #read image
    img = read_img(file,(INPUT_SIZE,INPUT_SIZE))
    #masking and segmentation
    image_segmented = segment_image(img)
    #sharpen
    image_sharpen = sharpen_image(image_segmented)
    return np.expand_dims(image_sharpen.copy(), axis=0)

path = st.text_input('Enter Image URL to Classify.. ','https://storage.googleapis.com/image_classification_2021/Glacier-Argentina-South-America-blue-ice.JPEG')
if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
      label =np.argmax(model.predict(decode_img(content)),axis=1)
      st.write(classes[label[0]])    
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Image', use_column_width=True)
