from flask import Flask, render_template, request, jsonify, redirect, url_for
# from keras.models import load_model
import cv2
import numpy as np
import time
# from flask_ngrok import run_with_ngrok
import tensorflow as tf

app = Flask(__name__)
# run_with_ngrok(app)

face_classifier = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml')
# classifier = load_model(r'.\model.h5')
classifier = tf.keras.models.load_model(
    './model.h5', custom_objects=None, compile=True, options=None
)


emotion_labels = {0: 'You are Angry', 1: 'You are Disgusted', 2: 'You are Feared',
                  3: 'You are Happy', 4: 'You are Neutral', 5: 'You are Sad', 6: 'You are Surprised'}

predicted_emotion = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/save_image', methods=['POST'])
def save_image():
    global predicted_emotion
    img_data = np.frombuffer(request.get_data(), np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray)

    if (len(faces) != 0):

        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            # print(label)
            predicted_emotion = label
            return redirect(url_for('show_emotion', emotion=label))
        else:
            predicted_emotion = "Face Wasn't Detected"
            return redirect(url_for('show_emotion', emotion="No face found"))
    else:
        predicted_emotion = "Face Wasn't Detected"
        return jsonify({'status': 'error'})


@app.route('/emotion')
def show_emotion():
    global predicted_emotion
    time.sleep(2)
    return render_template('emotion.html', emotion=predicted_emotion)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
