from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import os
import requests

app = Flask(__name__)

# Konfigurasi URL model
MODEL_URL = os.getenv("MODEL_URL", "https://storage.googleapis.com/my-bucket/imageclass_model.h5")

# Unduh dan simpan model
model_path = "./imageclass_model.h5"
if not os.path.exists(model_path):
    response = requests.get(MODEL_URL)
    if response.status_code != 200:
        raise ValueError(f"Failed to download model. HTTP status code: {response.status_code}")
    with open(model_path, "wb") as file:
        file.write(response.content)

# Muat model ke memori
model = load_model(model_path)

# Label kategori
labels = ['historical', 'makanan', 'museum', 'nature_adventure', 'park', 'waterpark', 'zoo']

# Batas ukuran file
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Maksimal 2 MB

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Validasi file
    if 'imagefile' not in request.files:
        return jsonify({"error": "No file part"}), 400

    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Simpan file gambar
    image_path = "./images/" + imagefile.filename
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    imagefile.save(image_path)

    # Preproses gambar
    try:
        image = load_img(image_path, target_size=(256, 256))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Prediksi menggunakan model
        yhat = model.predict(image)
        predicted_class = labels[yhat.argmax()]
        confidence = yhat.max()

        # Kembalikan hasil sebagai JSON
        return jsonify({
            "prediction": predicted_class,
            "confidence": f"{confidence * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)
