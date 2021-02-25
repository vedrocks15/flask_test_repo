import requests
import numpy as np 
from PIL import Image
from io import BytesIO
from numpy.linalg import norm
from flask import Flask, request, jsonify
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

# Initializing Flask App
app = Flask(__name__)



# model definition object
model = MobileNet(weights = "imagenet", include_top = False, pooling = "avg", input_shape=(224,224,3))


@app.route('/predict', methods = ['POST'])
def extract_features():
    """
    Model to extract features
    args:
     data: {json} containing url of image whose features are to be extracted & input shape of mode
    returns:
     flattened feature vector obtained through model inference
    """
    try:
        data = request.get_json(force = True)
    except Exception as e:
        return jsonify(["Post Request Error"]),400
    try:
        url = data["img_url"]
        shape = (224,224)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).resize(shape).convert('RGB')
        img_array = np.array(img)
    except Exception as e:
        return jsonify(["Image URL read error"]),400
    try:
        expanded_img_array = np.expand_dims(img_array, axis=0) # just because it is a single image, expand the dimension to get the batch
        preprocessed_img = preprocess_input(expanded_img_array) # pretrained models have their own preprocesss input function
        features = model.predict(preprocessed_img)
    except Exception as e:
        return jsonify(["Tensorflow model error"]),400
    try:
        features = features.flatten()
        features = features / norm(features) # std =1 mean = 0
        return jsonify(features.tolist()),200
    except Exception as e:
        return jsonify(["Feature Flatten Error"]),400


if __name__ == "__main__":
    app.run(debug = True)