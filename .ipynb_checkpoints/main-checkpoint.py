import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame, Text, Scrollbar, Entry
from PIL import Image, ImageTk
import threading
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Text-to-Speech setup
engine = pyttsx3.init()

def speak_text(text):
    def tts_thread():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=tts_thread, daemon=True).start()

labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z
labels_dict.update({26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: ' ', 37: '.'})
expected_features = 42

# Initialize variables
last_registered_time = time.time()
current_user = "User 1"
users = {"User 1": {"word": "", "sentence": ""}, "User 2": {"word": "", "sentence": ""}}

# GUI setup
root = tk.Tk()
root.title("Sign Language Chat")
root.geometry("1400x700")
root.configure(bg="#2c2f33")

video_label = Label(root)
video_label.pack()

current_alphabet = StringVar(value="N/A")
current_word = StringVar(value="N/A")
current_sentence = StringVar(value="N/A")

Label(root, text="Current Alphabet:", font=("Arial", 16), fg="#ffffff", bg="#2c2f33").pack()
Label(root, textvariable=current_alphabet, font=("Arial", 18), fg="#1abc9c", bg="#2c2f33").pack()
Label(root, text="Current Word:", font=("Arial", 16), fg="#ffffff", bg="#2c2f33").pack()
word_entry = Entry(root, textvariable=current_word, font=("Arial", 18), fg="#1abc9c", bg="#2c2f33")
word_entry.pack()
Label(root, text="Current Sentence:", font=("Arial", 16), fg="#ffffff", bg="#2c2f33").pack()
Label(root, textvariable=current_sentence, font=("Arial", 18), fg="#1abc9c", bg="#2c2f33").pack()

# Chatroom Setup
chat_frame = Frame(root, bg="#2c2f33")
chat_frame.pack(side="right", padx=20, pady=20)
chat_log = Text(chat_frame, height=20, width=50, font=("Arial", 14), state=tk.DISABLED, bg="#1e2124", fg="white")
chat_log.pack()

# Toggle User Button
def toggle_user():
    global current_user
    current_user = "User 2" if current_user == "User 1" else "User 1"
    user_label.config(text=f"Active User: {current_user}")

def send_message():
    global users
    message = users[current_user]["sentence"]
    if message.strip():
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"{current_user}: {message}\n")
        chat_log.config(state=tk.DISABLED)
        chat_log.see(tk.END)
        speak_text(message)
        users[current_user] = {"word": "", "sentence": ""}
        current_word.set("N/A")
        current_sentence.set("N/A")
        toggle_user()

# Reset button to clear current word and sentence
def reset_detection():
    global users
    users[current_user]["word"] = ""
    users[current_user]["sentence"] = ""
    current_word.set("N/A")
    current_sentence.set("N/A")

reset_button = Button(root, text="Reset Detection", command=reset_detection, font=("Arial", 14), bg="#e74c3c", fg="white")
reset_button.pack()

toggle_button = Button(root, text="Toggle User", command=toggle_user, font=("Arial", 14), bg="#3498db", fg="white")
toggle_button.pack()

send_button = Button(root, text="Send Message", command=send_message, font=("Arial", 14), bg="#2ecc71", fg="white")
send_button.pack()

user_label = Label(root, text=f"Active User: {current_user}", font=("Arial", 16), fg="#ffffff", bg="#2c2f33")
user_label.pack()

# Delay time in seconds
delay_time = 1.5

def process_frame():
    global last_registered_time, users, delay_time
    ret, frame = cap.read()
    if not ret:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()
    
    if results.multi_hand_landmarks and (current_time - last_registered_time > delay_time):  # Delay condition
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            min_x, min_y = min(x_), min(y_)
            data_aux = [(lm.x - min_x, lm.y - min_y) for lm in hand_landmarks.landmark]
            data_aux_flat = [val for pair in data_aux for val in pair]
            if len(data_aux_flat) < expected_features:
                data_aux_flat.extend([0] * (expected_features - len(data_aux_flat)))
            prediction = model.predict([np.asarray(data_aux_flat)])
            # breakpoint()
            predicted_character = str(prediction[0])
            # breakpoint()
            
            current_alphabet.set(str(predicted_character))
            
            if predicted_character == ' ':
                users[current_user]["sentence"] += users[current_user]["word"] + " "
                users[current_user]["word"] = ""
            elif predicted_character == '.':
                users[current_user]["sentence"] += users[current_user]["word"] + "."
                users[current_user]["word"] = ""
            else:
                users[current_user]["word"] += predicted_character
            
            current_word.set(users[current_user]["word"])
            current_sentence.set(users[current_user]["sentence"].strip())
            last_registered_time = current_time  # Update last registered time

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    root.after(10, process_frame)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

process_frame()
root.mainloop()

