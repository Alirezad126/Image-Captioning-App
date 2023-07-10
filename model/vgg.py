from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

vgg_model = VGG16()
vgg_model = Model(inputs = vgg_model.inputs, outputs = vgg_model.layers[-2].output)

def feature_extraction(uploaded_file):
    #Load image from file
    image = load_img(uploaded_file, target_size=(224,224))
    #convert image to np.array
    image = img_to_array(image)
    #reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #preprocess for vgg
    image = preprocess_input(image)
    #feature extraction
    feature_test = vgg_model.predict(image, verbose=0)
    return feature_test