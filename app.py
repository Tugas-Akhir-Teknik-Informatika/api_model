from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import io
import traceback

app = Flask(__name__)
model = load_model('model/model_efficientnet_b0.h5')

# Kelas sesuai urutan pada saat pelatihan
class_names = ['Folliculitis', 'Lichen Planopilaris', 'Normal', 'Psoriasis']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        pred_index = np.argmax(preds)
        label = class_names[pred_index]
        confidence = float(np.max(preds))

        return jsonify({
            'prediction': label,
            'confidence': round(confidence, 3)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
