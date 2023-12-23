import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('keras_model.h5')

gesture_labels = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

camera = cv2.VideoCapture(0)

while True:
    status, frame = camera.read()

    if status:
        frame = cv2.flip(frame, 1)
        frame_resized = cv2.resize(frame, (224, 224))
        frame_expanded = np.expand_dims(frame_resized, axis=0)
        frame_normalized = frame_expanded / 255.0
        predictions = model.predict(frame_normalized)
        predicted_class = np.argmax(predictions)
        gesture_label = gesture_labels.get(predicted_class, 'Unknown')
        confidence_scores = predictions[0] * 100

        cv2.putText(frame, f'Rock: {confidence_scores[0]:.2f}%', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f'Paper: {confidence_scores[1]:.2f}%', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Scissors: {confidence_scores[2]:.2f}%', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Gesture Recognition', frame)
        code = cv2.waitKey(1)

        if code == 32:
            break

camera.release()
cv2.destroyAllWindows()
