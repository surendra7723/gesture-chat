import socket
import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import time
from threading import Thread
from PIL import Image, ImageTk

# Load gesture recognition model
model_path = '/home/surendra/Code/College/gesture/Desktop/client/models/model.p'
with open(model_path, 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict.get('model') or model_dict.get('classifier') or list(model_dict.values())[0]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5000
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_HOST, SERVER_PORT))

class GestureClientApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Client")

        self.label = tk.Label(root, text="Detected Gesture: ", font=("Arial", 16))
        self.label.pack(pady=10)

        self.buffer_word = ""

        self.buffer_label = tk.Label(root, text="Buffer Word: ", font=("Arial", 14))
        self.buffer_label.pack(pady=10)

        self.word_label = tk.Label(root, text="", font=("Arial", 14), bg="#f0f0f0", width=30)
        self.word_label.pack(pady=10)

        self.send_button = tk.Button(root, text="Send Message", command=self.send_buffer, bg="#ccffcc")
        self.send_button.pack(pady=5)

        self.running = False
        self.last_registered_time = time.time()
        self.delay_time = 1.5  # Setting delay to 1.5 seconds like in main.py
        self.cap = None

        self.start_button = tk.Button(root, text="Start Capture", command=self.start_capture, bg="#ccccff")
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(root, text="Stop Capture", command=self.stop_capture, bg="#ffcccc")
        self.stop_button.pack(pady=5)

        # Add a reset button similar to main.py
        self.reset_button = tk.Button(root, text="Reset Buffer", command=self.reset_buffer, bg="#ffcccc")
        self.reset_button.pack(pady=5)

    def start_capture(self):
        if not self.running:
            self.running = True
            Thread(target=self.capture_gesture).start()

    def stop_capture(self):
        if self.running:
            self.running = False
            if self.cap is not None:
                self.cap.release()
    
    def reset_buffer(self):
        self.buffer_word = ""
        self.word_label.config(text="")

    def send_buffer(self):
        if self.buffer_word:
            try:
                client_socket.sendall(self.buffer_word.encode('utf-8'))
                self.buffer_word = ""
                self.word_label.config(text="")
            except Exception as e:
                print("Error sending data:", e)

    def append_to_buffer(self, gesture):
        # Updated to match main.py's way of handling spaces and periods
        if gesture == ' ':
            self.buffer_word += " "
        elif gesture == '.':
            self.buffer_word += "."
        else:
            self.buffer_word += gesture
        self.word_label.config(text=self.buffer_word)

    def capture_gesture(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return

        cv2.namedWindow("Camera Feed")
        expected_features = 42

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            
            current_time = time.time()

            if result.multi_hand_landmarks and (current_time - self.last_registered_time > self.delay_time):
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Extract features using the main.py approach
                    x_ = [lm.x for lm in hand_landmarks.landmark]
                    y_ = [lm.y for lm in hand_landmarks.landmark]
                    min_x, min_y = min(x_), min(y_)
                    
                    data_aux = [(lm.x - min_x, lm.y - min_y) for lm in hand_landmarks.landmark]
                    data_aux_flat = [val for pair in data_aux for val in pair]
                    
                    # Ensure we have the expected number of features
                    if len(data_aux_flat) < expected_features:
                        data_aux_flat.extend([0] * (expected_features - len(data_aux_flat)))
                    
                    try:
                        # Make prediction
                        prediction = model.predict([np.asarray(data_aux_flat)])
                        predicted_character = str(prediction[0])
                        
                        # Update UI and buffer
                        self.label.config(text=f"Detected Gesture: {predicted_character}")
                        self.append_to_buffer(predicted_character)
                        
                        # Update last registered time
                        self.last_registered_time = current_time
                    except Exception as e:
                        print(f"Prediction error: {e}")

            cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyWindow("Camera Feed")

root = tk.Tk()
app = GestureClientApp(root)
root.mainloop()

client_socket.close()
