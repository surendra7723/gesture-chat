# import socket
# import threading
# import tkinter as tk
# from tkinter import scrolledtext
# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import time

# # Load the trained model
# model_data = pickle.load(open("../models/model.p", "rb"))
# model = model_data["model"]  # Extract the actual model

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# gesture_buffer = []

# def connect_to_server():
#     global client
#     try:
#         client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         client.connect((SERVER_HOST, SERVER_PORT))
#         threading.Thread(target=receive_messages, daemon=True).start()
#         print("Connected to server!")
#     except socket.error as e:
#         print(f"Socket error: {e}")
#     except Exception as e:
#         print(f"Error connecting to server: {e}")

# def receive_messages():
#     while True:
#         try:
#             message = client.recv(1024).decode('utf-8')
#             if not message:
#                 print("Disconnected from server.")
#                 break
#             chat_display.config(state=tk.NORMAL)
#             chat_display.insert(tk.END, f"Friend: {message}\n")
#             chat_display.config(state=tk.DISABLED)
#             chat_display.yview(tk.END)
#         except ConnectionResetError:
#             print("Connection closed by server.")
#             break
#         except Exception as e:
#             print(f"Error receiving message: {e}")
#             break

# def send_message():
#     # Check if there's text in the input field
#     message = message_input.get()
    
#     # Check if there's content in the gesture buffer
#     global gesture_buffer
#     gesture_content = ''.join(gesture_buffer)
    
#     # Combine both sources if both exist
#     if message and gesture_content:
#         combined_message = message + " " + gesture_content
#     elif gesture_content:  # Only gesture content
#         combined_message = gesture_content
#     elif message:  # Only message input
#         combined_message = message
#     else:  # Nothing to send
#         return
    
#     # Send the combined message
#     client.send(combined_message.encode('utf-8'))
    
#     # Display in chat
#     chat_display.config(state=tk.NORMAL)
#     chat_display.insert(tk.END, f"You: {combined_message}\n")
#     chat_display.config(state=tk.DISABLED)
#     chat_display.yview(tk.END)
    
#     # Clear both inputs
#     message_input.delete(0, tk.END)
#     gesture_buffer.clear()
#     update_gesture_display()

# def start_gesture_recognition():
#     cap = cv2.VideoCapture(0)
#     last_detected_time = time.time()
#     delay = 1.5  # Delay time between gestures

#     global gesture_buffer

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)

#         current_time = time.time()

#         if results.multi_hand_landmarks and (current_time - last_detected_time > delay):
#             for hand_landmarks in results.multi_hand_landmarks:
#                 x_ = [lm.x for lm in hand_landmarks.landmark]
#                 y_ = [lm.y for lm in hand_landmarks.landmark]
#                 min_x, min_y = min(x_), min(y_)
#                 data_aux = [(lm.x - min_x, lm.y - min_y) for lm in hand_landmarks.landmark]
#                 data_aux_flat = [val for pair in data_aux for val in pair]

#                 if len(data_aux_flat) < 42:
#                     data_aux_flat.extend([0] * (42 - len(data_aux_flat)))

#                 prediction = model.predict([np.asarray(data_aux_flat)])
#                 gesture_text = prediction[0]

#                 if gesture_text:
#                     if gesture_text == "Send":
#                         send_combined_gesture()
#                     elif gesture_text == "Space":
#                         gesture_buffer.append(" ")
#                         update_gesture_display()
#                     elif gesture_text == "Delete" or gesture_text == "Backspace":
#                         if gesture_buffer:
#                             gesture_buffer.pop()
#                             update_gesture_display()
#                     else:
#                         # Add the gesture to buffer instead of sending immediately
#                         gesture_buffer.append(gesture_text)
#                         update_gesture_display()

#                 last_detected_time = current_time

#         cv2.imshow("Gesture Recognition", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def send_combined_gesture():
#     global gesture_buffer
#     if gesture_buffer:
#         combined_message = ''.join(gesture_buffer)
#         client.send(combined_message.encode('utf-8'))
#         chat_display.config(state=tk.NORMAL)
#         chat_display.insert(tk.END, f"You: {combined_message}\n")
#         chat_display.config(state=tk.DISABLED)
#         chat_display.yview(tk.END)

#         gesture_buffer = []  # Clear the buffer
#         update_gesture_display()

# def update_gesture_display():
#     gesture_label.config(text="Current Input: " + ''.join(gesture_buffer))

# def setup_gui():
#     global chat_display, message_input, gesture_label
#     root = tk.Tk()
#     root.title("Chat Client")

#     chat_display = scrolledtext.ScrolledText(root, state=tk.DISABLED, wrap=tk.WORD, width=50, height=20)
#     chat_display.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

#     message_input = tk.Entry(root, width=40)
#     message_input.grid(row=1, column=0, padx=10, pady=10)

#     send_button = tk.Button(root, text="Send", command=send_message)
#     send_button.grid(row=1, column=1, padx=10, pady=10)

#     gesture_button = tk.Button(root, text="Start Gesture Input", command=lambda: threading.Thread(target=start_gesture_recognition, daemon=True).start())
#     gesture_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

#     gesture_label = tk.Label(root, text="Current Input: ", anchor="w")
#     gesture_label.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

#     root.mainloop()

# SERVER_HOST = '127.0.0.1'
# SERVER_PORT = 5000

# if __name__ == "__main__":
#     connect_to_server()
#     setup_gui()


import socket
import threading
import tkinter as tk
from tkinter import scrolledtext
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image, ImageTk

# Load the trained model
model_data = pickle.load(open("../models/model.p", "rb"))
model = model_data["model"]  # Extract the actual model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Global variables
gesture_buffer = []
cap = None
is_camera_on = False

def connect_to_server():
    global client
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((SERVER_HOST, SERVER_PORT))
        threading.Thread(target=receive_messages, daemon=True).start()
        print("Connected to server!")
    except socket.error as e:
        print(f"Socket error: {e}")
    except Exception as e:
        print(f"Error connecting to server: {e}")

def receive_messages():
    while True:
        try:
            message = client.recv(1024).decode('utf-8')
            if not message:
                print("Disconnected from server.")
                break
            chat_display.config(state=tk.NORMAL)
            chat_display.insert(tk.END, f"Friend: {message}\n")
            chat_display.config(state=tk.DISABLED)
            chat_display.yview(tk.END)
        except ConnectionResetError:
            print("Connection closed by server.")
            break
        except Exception as e:
            print(f"Error receiving message: {e}")
            break

def send_message():
    # Check if there's text in the input field
    message = message_input.get()
    
    # Check if there's content in the gesture buffer
    global gesture_buffer
    gesture_content = ''.join(gesture_buffer)
    
    # Combine both sources if both exist
    if message and gesture_content:
        combined_message = message + " " + gesture_content
    elif gesture_content:  # Only gesture content
        combined_message = gesture_content
    elif message:  # Only message input
        combined_message = message
    else:  # Nothing to send
        return
    
    # Send the combined message
    client.send(combined_message.encode('utf-8'))
    
    # Display in chat
    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, f"You: {combined_message}\n")
    chat_display.config(state=tk.DISABLED)
    chat_display.yview(tk.END)
    
    # Clear both inputs
    message_input.delete(0, tk.END)
    gesture_buffer.clear()
    update_gesture_display()

def start_gesture_recognition():
    global is_camera_on, cap
    
    if is_camera_on:
        return
    
    is_camera_on = True
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        is_camera_on = False
        return
    
    # Update the button text
    gesture_button.config(text="Stop Gesture Input")
    
    # Start updating the camera feed
    update_camera_feed()

def stop_gesture_recognition():
    global is_camera_on, cap
    
    is_camera_on = False
    
    if cap is not None:
        cap.release()
    
    # Reset the video display
    video_label.config(image="")
    
    # Update the button text
    gesture_button.config(text="Start Gesture Input")

def toggle_gesture_recognition():
    global is_camera_on
    
    if is_camera_on:
        stop_gesture_recognition()
    else:
        start_gesture_recognition()

def update_camera_feed():
    global cap, is_camera_on
    
    if not is_camera_on:
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        stop_gesture_recognition()
        return
    
    # Process frame for hand detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # If hands are detected, draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
    
    # Process gestures less frequently
    current_time = time.time()
    last_detection_time = getattr(update_camera_feed, 'last_detection_time', 0)
    detection_delay = 1.5  # seconds
    
    if results.multi_hand_landmarks and (current_time - last_detection_time > detection_delay):
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            min_x, min_y = min(x_), min(y_)
            data_aux = [(lm.x - min_x, lm.y - min_y) for lm in hand_landmarks.landmark]
            data_aux_flat = [val for pair in data_aux for val in pair]
            
            if len(data_aux_flat) < 42:
                data_aux_flat.extend([0] * (42 - len(data_aux_flat)))
            
            prediction = model.predict([np.asarray(data_aux_flat)])
            gesture_text = prediction[0]
            
            if gesture_text:
                # Display detected gesture on frame
                cv2.putText(frame_rgb, f"Gesture: {gesture_text}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                if gesture_text == "Send":
                    send_combined_gesture()
                elif gesture_text == "Space":
                    gesture_buffer.append(" ")
                    update_gesture_display()
                elif gesture_text == "Delete" or gesture_text == "Backspace":
                    if gesture_buffer:
                        gesture_buffer.pop()
                        update_gesture_display()
                else:
                    # Add the gesture to buffer
                    gesture_buffer.append(gesture_text)
                    update_gesture_display()
            
            update_camera_feed.last_detection_time = current_time
    
    # Convert to ImageTk format
    img = Image.fromarray(frame_rgb)
    img = img.resize((320, 240))  # Resize for better fit in the UI
    imgtk = ImageTk.PhotoImage(image=img)
    
    # Update label with new image
    video_label.imgtk = imgtk  # Keep a reference to prevent garbage collection
    video_label.config(image=imgtk)
    
    # Schedule the next frame update
    video_label.after(10, update_camera_feed)

def send_combined_gesture():
    global gesture_buffer
    if gesture_buffer:
        combined_message = ''.join(gesture_buffer)
        client.send(combined_message.encode('utf-8'))
        chat_display.config(state=tk.NORMAL)
        chat_display.insert(tk.END, f"You: {combined_message}\n")
        chat_display.config(state=tk.DISABLED)
        chat_display.yview(tk.END)

        gesture_buffer = []  # Clear the buffer
        update_gesture_display()

def update_gesture_display():
    gesture_label.config(text="Current Input: " + ''.join(gesture_buffer))

def setup_gui():
    global chat_display, message_input, gesture_label, video_label, gesture_button
    root = tk.Tk()
    root.title("Gesture Chat Client")
    root.geometry("800x600")  # Set initial window size
    
    # Create main frames
    left_frame = tk.Frame(root, width=400)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    right_frame = tk.Frame(root, width=400)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Left frame - Chat area
    chat_display = scrolledtext.ScrolledText(left_frame, state=tk.DISABLED, wrap=tk.WORD, 
                                            width=40, height=20)
    chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    
    # Message input area
    input_frame = tk.Frame(left_frame)
    input_frame.pack(fill=tk.X, pady=5)
    
    message_input = tk.Entry(input_frame, width=30)
    message_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    message_input.bind("<Return>", lambda event: send_message())
    
    send_button = tk.Button(input_frame, text="Send", command=send_message)
    send_button.pack(side=tk.RIGHT)
    
    # Gesture buffer display
    gesture_label = tk.Label(left_frame, text="Current Input: ", anchor="w", 
                            font=("Arial", 12), relief=tk.SUNKEN, padx=5, pady=5)
    gesture_label.pack(fill=tk.X, pady=10)
    
    # Right frame - Video feed and controls
    video_label = tk.Label(right_frame, bg="black", width=320, height=240)
    video_label.pack(pady=10)
    
    # Control buttons
    controls_frame = tk.Frame(right_frame)
    controls_frame.pack(fill=tk.X, pady=10)
    
    gesture_button = tk.Button(controls_frame, text="Start Gesture Input", 
                              command=toggle_gesture_recognition)
    gesture_button.pack(side=tk.LEFT, padx=5)
    
    send_gesture_button = tk.Button(controls_frame, text="Send Gesture Input", 
                                   command=send_combined_gesture)
    send_gesture_button.pack(side=tk.RIGHT, padx=5)
    
    # Cleanup on window close
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_gesture_recognition(), root.destroy()))
    
    root.mainloop()

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5000

if __name__ == "__main__":
    connect_to_server()
    setup_gui()