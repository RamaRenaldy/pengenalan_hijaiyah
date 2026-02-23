# app.py

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import base64
import re
import os
from datetime import date
import json

today = date.today()
datefull = today.strftime("%d/%m/%Y")
datefull = str(datefull)
date_processed = datefull.split('/')
cur_year = date_processed[2]

app = Flask(__name__)
loaded_model = load_model("models/fixmodel.h5")
print('Model ready!!')

def load_data():
    with open('static/abjad_updated.json') as f:
        return json.load(f)
    
def convertImage(imgData1):
    try:
        imgstr = re.search(r'base64,(.*)', imgData1.decode('utf-8')).group(1)
        with open('output.png', 'wb') as output:
            output.write(base64.b64decode(imgstr))
    except AttributeError as e:
        print(f"Error: {e}")
        raise ValueError("Invalid base64 format")
    except Exception as e:
        print(f"Exception: {e}")
        raise ValueError("Error processing image")
    

def load_image(img_path):
    try:
        img = load_img(img_path, target_size=(150, 150, 3))
        img_tensor = img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0
        return img_tensor
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        raise ValueError(f"File '{img_path}' not found")
    except Exception as e:
        print(f"Exception: {e}")
        raise ValueError("Error loading image")

@app.route('/')
def index():
    return render_template("index.html", year=cur_year)

@app.route('/predict/', methods=['POST'])
def predict():
    class_names = ['ain', 'alif', 'ba', 'dal', 'dhod', 'dzal',
                   'dzho', 'fa', 'ghoin', 'ha', 'ha\'', 'hamzah', 'jim',
                   'kaf', 'kho', 'lam', 'lamalif', 'mim', 'nun', 'qof',
                   'ro', 'shod', 'sin', 'syin', 'ta', 'tho', 'tsa',
                   'wawu', 'ya', 'zain']
    try:
        imgData = request.get_data()
        convertImage(imgData)
        img = load_image('output.png')
        pred = loaded_model.predict(img)
        response = class_names[np.argmax(pred)]
        return jsonify({"prediction": response}), 200
    except ValueError as ve:
        return str(ve), 400
    except Exception as e:
        print(f"Exception: {e}")
        return "Internal Server Error", 500

if __name__ == "__main__":
    app.run(debug=False)
