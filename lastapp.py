import os
import datetime
import pickle
import mysql.connector  # Import MySQL connector
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet import preprocess_input
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import face_recognition
import util2

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x600+300+80")
        self.main_window.title("Face Recognition App")
        self.main_window.configure(bg="#2C3E50")

        # Load the spoof detection model
        self.spoof_model = load_model(r'C:\Users\ahala\OneDrive\Desktop\college\levl3 1\Work-based Professional Project in Cyber security(1)\last project\MobileNetFaceSpoof.h5')
        self.spoof_threshold = 0.7  # Adjust threshold based on model's confidence level

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
            self.main_window, text="Face Recognition System", font=("Helvetica", 24, "bold"),
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
        self.login_button_main_window = util2.get_button(control_frame, 'Login', '#27AE60', self.login)
        self.login_button_main_window.pack(pady=10, fill="x", padx=20)

        self.logout_button_main_window = util2.get_button(control_frame, 'Logout', '#C0392B', self.logout)
        self.logout_button_main_window.pack(pady=10, fill="x", padx=20)

        self.register_new_user_button_main_window = util2.get_button(
            control_frame, 'Register New User', '#7F8C8D', self.register_new_user, fg='black'
        )
        self.register_new_user_button_main_window.pack(pady=10, fill="x", padx=20)

        # Set up webcam
        self.add_webcam(self.webcam_label)

        # Directories and log path
        self.db_dir = './db'
        os.makedirs(self.db_dir, exist_ok=True)

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Flip frame to create a mirrored effect
            self.most_recent_capture_arr = frame

            # Detect faces in the frame
            face_locations = face_recognition.face_locations(frame)
            for top, right, bottom, left in face_locations:
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Crop the face image for spoof detection
                face_image = frame[top:bottom, left:right]

                # Detect if the face is real or fake
                result, color = self.detect_real_or_fake(face_image)
                cv2.putText(frame, f"Face: {result}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Convert the frame to RGB and display it in the Tkinter label
            img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)

        # Repeat the process every 20 milliseconds
        self._label.after(20, self.process_webcam)

    def detect_real_or_fake(self, face_image):
        """Detect if a face is real or fake using the spoof detection model."""
        try:
            # Resize the face image to the model's expected input size
            face_image = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_array = img_to_array(face_rgb)
            face_array = np.expand_dims(face_array, axis=0)
            face_array = preprocess_input(face_array)

            # Get the prediction from the model
            prediction = self.spoof_model.predict(face_array)[0][0]  # Assuming the model outputs a single value

            # Debug: Print the prediction value
            print(f"Prediction value: {prediction}")

            if prediction < self.spoof_threshold:
                return "Real", (0, 255, 0)  # Green
            else:
                return "Fake", (0, 0, 255)  # Red
        except Exception as e:
            print(f"Error in detect_real_or_fake: {e}")
            return "Error", (255, 0, 0)  # Red for error

    def add_img_to_label(self, label):
        """Capture an image from the webcam and display it in the specified label."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Flip the frame to create a mirrored effect
            self.register_new_user_capture = frame  # Store the captured frame for registration
            self.most_recent_capture_arr = frame  # Also store it for other uses

            # Convert the frame to RGB and display it in the label
            img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            label.imgtk = imgtk
            label.configure(image=imgtk)

    def login(self):
        # Capture face and detect if it's real
        face_image = self.most_recent_capture_arr
        result, color = self.detect_real_or_fake(face_image)
        if result == "Fake":
            util2.msg_box('Access Denied', 'Fake face detected! Login aborted.')
            return

        # Proceed with existing face recognition logic
        name = util2.recognize(face_image, self.db_cursor)
        if name in ['unknown_person', 'no_persons_found']:
            util2.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            if self.is_already_logged_in(name):
                util2.msg_box('Notice', f'{name} is already logged in.')
            else:
                util2.msg_box('Welcome back!', f'Welcome, {name}.')
                self.store_log_in_db(name, 'in')

    def logout(self):
        name = util2.recognize(self.most_recent_capture_arr, self.db_cursor)
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

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip()
        if name:
            encodings = face_recognition.face_encodings(self.register_new_user_capture)
            if encodings:
                embeddings = encodings[0]  # Use the first detected face encoding
                try:
                    # Serialize the embeddings to binary
                    serialized_embeddings = pickle.dumps(embeddings)

                    # Insert into database
                    query = "INSERT INTO users (username, embeddings) VALUES (%s, %s)"
                    self.db_cursor.execute(query, (name, serialized_embeddings))
                    self.db_connection.commit()

                    util2.msg_box('Success!', 'User was registered successfully!')
                    self.register_new_user_window.destroy()
                except mysql.connector.Error as err:
                    util2.msg_box('Error', f"Database error: {err}")
            else:
                util2.msg_box('Error', 'No face detected. Please try again.')
        else:
            util2.msg_box('Error', 'Username cannot be empty.')

    def try_again_register_new_user(self):
        """Close the registration window and allow the user to try again."""
        self.register_new_user_window.destroy()

    def register_new_user(self):
        """Open a new window for registering a new user."""
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("800x400+370+120")
        self.register_new_user_window.title("Register New User")
        self.register_new_user_window.configure(bg="#34495E")

        # Capture label
        self.capture_label = util2.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=25, y=40, width=400, height=300)
        self.add_img_to_label(self.capture_label)

        # User entry
        self.entry_text_register_new_user = util2.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=450, y=70, width=300, height=50)

        # Text label for entry
        text_label = tk.Label(self.register_new_user_window, text="Please, input username:", font=("Helvetica", 14),
                              bg="#34495E", fg="white")
        text_label.place(x=450, y=30)

        # Accept and Try Again buttons
        self.accept_button = util2.get_button(self.register_new_user_window, 'Accept', '#27AE60', self.accept_register_new_user)
        self.accept_button.place(x=450, y=150)

        self.try_again_button = util2.get_button(self.register_new_user_window, 'Try Again', '#C0392B', self.try_again_register_new_user)
        self.try_again_button.place(x=450, y=250)

    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()