import cv2
import numpy as np
import mysql.connector
import os
import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import face_recognition
from tensorflow.keras.models import model_from_json
import util2
import pickle

# Paths for model and weights
JSON_DIRECTORY = "C:/Users/ahala/OneDrive/Desktop/college/Work-based Professional Project in Cyber security(1)/Final_version/weights/AAOHybrid_model_final.json"
WEIGHTS_DIRECTORY = "C:/Users/ahala/OneDrive/Desktop/college/Work-based Professional Project in Cyber security(1)/Final_version/weights/AAOHybrid_final.weights.h5"

# Load the model
with open(JSON_DIRECTORY, "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights(WEIGHTS_DIRECTORY)

# Constants for frame adjustments and face detection
OFFSET_TOP, OFFSET_BOTTOM, OFFSET_LEFT, OFFSET_RIGHT = 100, 60, 60, 60
FRAME_RESIZE_DIM = (256, 256)

class FaceRecognitionApp:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x600+300+80")
        self.main_window.title("Face Recognition and Authentication App")
        self.main_window.configure(bg="#2C3E50")  # Dark background for modern look

        # Database connection setup
        self.db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="#PpZzalaa09067",
            database="face_recognition"
        )
        self.db_cursor = self.db_connection.cursor()

        # Header label
        header_label = tk.Label(
            self.main_window, text="Face Recognition and Authentication", font=("Helvetica", 24, "bold"),
            bg="#2C3E50", fg="white"
        )
        header_label.place(x=50, y=10)

        # Main webcam frame
        webcam_frame = tk.Frame(self.main_window, width=700, height=500, bg="white", bd=2, relief="sunken")
        webcam_frame.place(x=30, y=60)
        self.webcam_label = util2.get_img_label(webcam_frame)
        self.webcam_label.pack(fill="both", expand=True)

        # Control panel frame
        control_frame = tk.Frame(self.main_window, bg="#34495E", bd=3, relief="ridge", width=300, height=450)
        control_frame.place(x=780, y=60)

        # Title for control panel
        control_label = tk.Label(
            control_frame, text="Control Panel", font=("Helvetica", 18, "bold"), bg="#34495E", fg="white"
        )
        control_label.pack(pady=20)

        # Buttons in control panel
        self.login_button = util2.get_button(control_frame, 'Login', '#27AE60', self.login)
        self.login_button.pack(pady=10, fill="x", padx=20)

        self.logout_button = util2.get_button(control_frame, 'Logout', '#C0392B', self.logout)
        self.logout_button.pack(pady=10, fill="x", padx=20)

        self.register_button = util2.get_button(control_frame, 'Register User', '#7F8C8D', self.register_new_user, fg='black')
        self.register_button.pack(pady=10, fill="x", padx=20)

        # Set up webcam
        self.add_webcam(self.webcam_label)

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Flip frame to create a mirrored effect
            self.current_frame = frame.copy()
            img_ = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img_))
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)
        self._label.after(20, self.process_webcam)

    def detect_real_or_fake(self, frame):
        """Predicts if the face in the given frame is real or fake."""
        faces = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in faces:
            # Expand face bounding box with offsets
            top_expanded = max(top - OFFSET_TOP, 0)
            bottom_expanded = min(bottom + OFFSET_BOTTOM, frame.shape[0])
            left_expanded = max(left - OFFSET_LEFT, 0)
            right_expanded = min(right + OFFSET_RIGHT, frame.shape[1])

            face = frame[top_expanded:bottom_expanded, left_expanded:right_expanded]
            if face.size == 0:
                continue

            # Resize and normalize face for model input
            resized_face = cv2.resize(face, FRAME_RESIZE_DIM)
            resized_face = resized_face.astype("float32") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)

            # Predict real or fake
            preds = model.predict(resized_face)[0]
            label = "fake" if preds[0] > 0.5 else "real"
            return label
        return "no_face"

    def login(self):
        label = self.detect_real_or_fake(self.current_frame)
        if label == "fake":
            util2.msg_box("Error", "Fake face detected. Access denied.")
            return
        elif label == "no_face":
            util2.msg_box("Error", "No face detected. Please try again.")
            return

        name = util2.recognize(self.current_frame, self.db_cursor)
        if name in ['unknown_person', 'no_persons_found']:
            util2.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            if self.is_already_logged_in(name):
                util2.msg_box('Notice', f'{name} is already logged in.')
            else:
                util2.msg_box('Welcome back!', f'Welcome, {name}.')
                self.store_log_in_db(name, 'in')

    def logout(self):
        name = util2.recognize(self.current_frame, self.db_cursor)
        if name in ['unknown_person', 'no_persons_found']:
            util2.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            if not self.is_already_logged_in(name):
                util2.msg_box('Notice', f'{name} is already logged out.')
            else:
                util2.msg_box('Goodbye!', f'Goodbye, {name}.')
                self.store_log_in_db(name, 'out')

    def is_already_logged_in(self, username):
        query = "SELECT action FROM logs WHERE username = %s ORDER BY timestamp DESC LIMIT 1"
        self.db_cursor.execute(query, (username,))
        last_action = self.db_cursor.fetchone()
        return last_action and last_action[0] == 'in'

    def store_log_in_db(self, username, action):
        query = "INSERT INTO logs (username, timestamp, action) VALUES (%s, %s, %s)"
        timestamp = datetime.datetime.now()
        self.db_cursor.execute(query, (username, timestamp, action))
        self.db_connection.commit()

    def register_new_user(self):
        label = self.detect_real_or_fake(self.current_frame)
        if label == "fake":
            util2.msg_box("Error", "Fake face detected. Cannot register.")
            return
        elif label == "no_face":
            util2.msg_box("Error", "No face detected. Please try again.")
            return

        self.register_window = tk.Toplevel(self.main_window)
        self.register_window.geometry("800x400+370+120")
        self.register_window.title("Register New User")
        self.register_window.configure(bg="#34495E")

        self.capture_label = util2.get_img_label(self.register_window)
        self.capture_label.place(x=25, y=40, width=400, height=300)
        self.add_img_to_label(self.capture_label)

        entry_label = tk.Label(self.register_window, text="Enter Username:", font=("Helvetica", 14), bg="#34495E", fg="white")
        entry_label.place(x=450, y=30)

        self.name_entry = tk.Text(self.register_window, height=1, font=("Helvetica", 14))
        self.name_entry.place(x=450, y=70, width=300)

        accept_btn = util2.get_button(self.register_window, 'Accept', '#27AE60', self.accept_register_new_user)
        accept_btn.place(x=450, y=150)

        retry_btn = util2.get_button(self.register_window, 'Retry', '#C0392B', self.register_window.destroy)
        retry_btn.place(x=450, y=250)

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame))
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def accept_register_new_user(self):
        username = self.name_entry.get(1.0, "end-1c").strip()
        if username:
            encodings = face_recognition.face_encodings(self.current_frame)
            if encodings:
                serialized_embeddings = pickle.dumps(encodings[0])
                try:
                    query = "INSERT INTO users (username, embeddings) VALUES (%s, %s)"
                    self.db_cursor.execute(query, (username, serialized_embeddings))
                    self.db_connection.commit()
                    util2.msg_box('Success', 'User registered successfully.')
                    self.register_window.destroy()
                except mysql.connector.Error as err:
                    util2.msg_box('Error', f'Database error: {err}')
            else:
                util2.msg_box('Error', 'No face detected.')
        else:
            util2.msg_box('Error', 'Username cannot be empty.')

    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.start()
