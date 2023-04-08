import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import cv2
import openai
import time

MODEL_FILENAME = 'emotion_recognition_model.h5'
openai.api_key = "YOUR_API_KEY_HERE"

def load_data():
    data = pd.read_csv('fer2013.csv')
    pixels = data['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
    x = np.vstack(pixels.values)
    x = x.reshape(-1, 48, 48, 1)
    x = x.astype('float32') / 255.0
    y = pd.get_dummies(data['emotion']).values
    return x, y


def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    start_time = time.time()
    model.fit(
        x_train, y_train, epochs=20, batch_size=64,
        validation_data=(x_test, y_test),
        callbacks=[TqdmCallback(verbose=1)],
        verbose=0
    )
    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f} seconds')
    model.save(MODEL_FILENAME)


def emotion_recognition():
    model = tf.keras.models.load_model(MODEL_FILENAME)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set the frame rate to 30 FPS


    cooldown_time = 5  # Time (in seconds) between GPT-4/GPT-3.5-turbo API calls
    display_duration = 20  # Time (in seconds) to display the AI's response
    last_call_time = 0  # Last time the GPT-4/GPT-3.5-turbo API was called
    response_display_start_time = 0  # Time the current response started being displayed
    chatbot_response = ""

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        emotion_label = None
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, 0)
            face = np.expand_dims(face, -1)
            emotion = model.predict(face)
            emotion_label = emotion_labels[np.argmax(emotion)]

            current_time = time.time()
            if current_time - last_call_time > cooldown_time and current_time - response_display_start_time > display_duration:
                # Get chatbot response
               # Get chatbot response
                # Get chatbot response
                chatbot_response = generate_chatbot_response(f"You see my face and you know I feel {emotion_label}.", "You know my emotion just by seeing my face, say something about my expression.")
                chatbot_response = f'AI: {chatbot_response}'

                # Update last_call_time and response_display_start_time
                last_call_time = current_time
                response_display_start_time = current_time

        # Display chatbot response
        cv2.putText(frame, chatbot_response, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if emotion_label:
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def main():
    if not os.path.exists(MODEL_FILENAME):
        print("Loading dataset...")
        x, y = load_data()

        print("Creating model...")
        model = create_model()

        print("Training model...")
        train_model(model, x, y)
        print("Model trained and saved.")

    print("Running emotion recognition...")
    emotion_recognition()

def generate_chatbot_response(prompt, system_content, retries=30, backoff_factor=2):
    for retry_count in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_content}, {"role": "user", "content": prompt}],
                max_tokens=1200,
                n=1,
                stop=None,
                temperature=1,
            )
            return response.choices[0].message['content'].strip()
        except openai.error.OpenAIError as e:
            if 'Model overloaded' in str(e) and retry_count < retries - 1:
                sleep_time = backoff_factor ** (retry_count + 1)
                time.sleep(sleep_time)
            else:
                raise


if __name__ == '__main__':
    main()

