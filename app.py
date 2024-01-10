#creating web app for CNN
#set up simple web server
#from flask import Flask, render_template

#app = Flask(__name__)

#@app.route('/')
#ef home():
  #  return render_template('index.html')
#if __name__ == '__main__':
    #app.run(debug = True)

from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization

import cv2
import numpy as np

app = Flask(__name__)
model = load_model('C:/Users/GKamau/PycharmProjects/CNN/cnn_model1.h5')  # Load your trained CNN model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        prediction = predict_image(file_path)
        return render_template('result.html', prediction=prediction, image_path=file_path)

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Add Batch Normalization to the input layer
    model_with_bn = models.Sequential()
    model_with_bn.add(BatchNormalization(input_shape=(64, 64, 3)))
    model_with_bn.add(model)

    prediction = model_with_bn.predict(img)
    if np.argmax(prediction) == 0:
        label = "This is a Cat"
    elif np.argmax(prediction) == 1:
        label = "This is a Dog"
    else:
        label = "This is neither a cat nor a dog"

    print("Prediction Probability:", prediction)
    return label

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create a directory to store uploaded images
    app.run(debug=True)
