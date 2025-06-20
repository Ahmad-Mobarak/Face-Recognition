import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model

# Load the pre-trained face spoofing detection model
model = load_model('MobileNetFaceSpoof.h5', custom_objects={})
labels = ["Real", "Spoof"]

# Initialize the webcam
cap = cv2.VideoCapture(1)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_roi = frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_AREA)  # Resize to 224x224 for the model
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face_array = img_to_array(face_rgb)  # Convert to array
        face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
        face_array = preprocess_input(face_array)  # Preprocess for MobileNet

        # Predict whether the face is real or spoof
        prediction = model.predict(face_array)[0]
        if prediction < 0.7:  # Threshold for real/spoof classification
            label = "Real"
            color = (0, 255, 0)  # Green for real
        else:
            label = "Spoof"
            color = (0, 0, 255)  # Red for spoof

        # Draw a rectangle around the face and display the label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame with face spoofing detection
    cv2.imshow("Real-Time Face Spoof Detection", frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()