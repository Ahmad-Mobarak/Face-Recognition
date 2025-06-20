import os
import pickle

import tkinter as tk
from tkinter import messagebox
import face_recognition


def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
                        window,
                        text=text,
                        activebackground="black",
                        activeforeground="white",
                        fg=fg,
                        bg=color,
                        command=command,
                        height=2,
                        width=20,
                        font=('Helvetica bold', 20)
                    )

    return button


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Arial", 32))
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)


def recognize(img, db_cursor, threshold=0.4):
    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    else:
        embeddings_unknown = embeddings_unknown[0]

    # Query all user embeddings from the database
    query = "SELECT username, embeddings FROM users"
    db_cursor.execute(query)

    closest_match = None
    closest_distance = float('inf')

    for username, serialized_embeddings in db_cursor.fetchall():
        # Deserialize the embeddings
        try:
            embeddings = pickle.loads(serialized_embeddings)
        except Exception as e:
            print(f"Error deserializing embeddings for {username}: {e}")
            continue

        # Calculate face distance
        distance = face_recognition.face_distance([embeddings], embeddings_unknown)[0]
        if distance < closest_distance:
            closest_distance = distance
            closest_match = username

    # Return the closest match if within threshold
    if closest_distance <= threshold:
        return closest_match
    else:
        return 'unknown_person'



