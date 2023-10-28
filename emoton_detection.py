import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

def detect_emotion(frame, face_cascade, emotion_model):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame Pre trained classifier, detect object in image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face image to the input size expected by the model
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = img_to_array(face_roi)    # Convert the image to array and normalize
        face_roi = np.expand_dims(face_roi, axis=0) / 255.0
        emotion_scores = emotion_model.predict(face_roi)[0]
        emotion_label = emotion_labels[np.argmax(emotion_scores)]
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)  # Draw the emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)   # Draw a rectangle around the face

    return frame

if __name__ == "__main__":
    emotion_model = load_model('emotion_model.h5')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    #Pre trained model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = detect_emotion(frame, face_cascade, emotion_model)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
