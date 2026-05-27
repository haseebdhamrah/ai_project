
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

model = load_model("scene_classifier.h5")

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

@app.route("/")
def home():
    return "Scene Classification API is running"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    file_path = "uploaded_image.jpg"
    file.save(file_path)

    img = image.load_img(file_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
