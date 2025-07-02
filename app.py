from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = load_model('model/model_efficientnet_b0.h5')
class_names = ['Normal', 'Psoriasis', 'Folliculitis', 'Lichen Planopilaris']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    label = class_names[pred_index]
    confidence = float(np.max(preds))

    return jsonify({
        'prediction': label,
        'confidence': round(confidence, 3)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
