import socket
import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
import time
from threading import Thread
from PIL import Image, ImageTk
import hashlib
from tkinter import messagebox
import pyttsx3  # Add text-to-speech library

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
# You can adjust these settings as needed
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 0.9)  # Volume (0 to 1)

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

# User credentials (username: password)
# In a real application, store these securely, not in the source code
user_credentials = {
    "user1": hashlib.sha256("password1".encode()).hexdigest(),
    "user2": hashlib.sha256("password2".encode()).hexdigest()
}

class LoginWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Login")
        self.root.geometry("400x300")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(False, False)
        
        # Variables to store authentication result
        self.authenticated = False
        self.username = None
        
        # Create login form
        self.frame = tk.Frame(root, bg="#f0f0f0", padx=40, pady=40)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        self.title_label = tk.Label(
            self.frame, 
            text="Gesture Recognition Login", 
            font=("Arial", 16, "bold"), 
            bg="#f0f0f0"
        )
        self.title_label.pack(pady=(0, 20))
        
        # Username
        self.username_label = tk.Label(
            self.frame, 
            text="Username:", 
            font=("Arial", 12), 
            bg="#f0f0f0", 
            anchor="w"
        )
        self.username_label.pack(fill=tk.X)
        
        self.username_entry = tk.Entry(self.frame, font=("Arial", 12))
        self.username_entry.pack(fill=tk.X, pady=(0, 15))
        
        # Password
        self.password_label = tk.Label(
            self.frame, 
            text="Password:", 
            font=("Arial", 12), 
            bg="#f0f0f0", 
            anchor="w"
        )
        self.password_label.pack(fill=tk.X)
        
        self.password_entry = tk.Entry(self.frame, font=("Arial", 12), show="*")
        self.password_entry.pack(fill=tk.X, pady=(0, 20))
        
        # Login button
        self.login_button = tk.Button(
            self.frame, 
            text="Login", 
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=5,
            command=self.authenticate
        )
        self.login_button.pack()
        
        # Bind Enter key to login
        self.root.bind("<Return>", lambda event: self.authenticate())
        
        # Center the username field cursor
        self.username_entry.focus_set()

    def authenticate(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Login Error", "Please enter both username and password")
            return
            
        # Check credentials
        if username in user_credentials and hashlib.sha256(password.encode()).hexdigest() == user_credentials[username]:
            self.authenticated = True
            self.username = username
            self.root.destroy()  # Close login window
        else:
            messagebox.showerror("Login Error", "Invalid username or password")
            self.password_entry.delete(0, tk.END)  # Clear password field

class GestureClientApp:
    def __init__(self, root, username="Guest"):
        self.root = root
        self.root.title(f"Gesture Recognition Client - {username}")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # TTS control variable
        self.tts_enabled = tk.BooleanVar(value=True)
        
        # Create a header frame for user info and logout
        self.header_frame = tk.Frame(root, bg="#f0f0f0")
        self.header_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Add logged-in user display
        self.user_info_label = tk.Label(
            self.header_frame, 
            text=f"Logged in as: {username}", 
            font=("Arial", 10, "italic"),
            fg="#555555",
            bg="#f0f0f0",
        )
        self.user_info_label.pack(side=tk.LEFT, padx=5)
        
        # Add TTS checkbox
        self.tts_check = tk.Checkbutton(
            self.header_frame,
            text="Enable Speech",
            variable=self.tts_enabled,
            bg="#f0f0f0",
            font=("Arial", 10)
        )
        self.tts_check.pack(side=tk.LEFT, padx=20)
        
        # Add logout button
        self.logout_button = tk.Button(
            self.header_frame,
            text="Logout",
            font=("Arial", 10),
            bg="#FF5252",
            fg="white",
            padx=10,
            pady=2,
            command=self.logout
        )
        self.logout_button.pack(side=tk.RIGHT, padx=5)
        
        # User data with logged-in username
        self.users = {
            "User 1": {"buffer": "", "name": username},
            "User 2": {"buffer": "", "name": "Guest"}
        }
        self.current_user = "User 1"
        
        # Create main frames
        self.left_frame = tk.Frame(root, bg="#f0f0f0")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.right_frame = tk.Frame(root, bg="#f0f0f0")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        
        # Left side - Camera and gesture display
        self.video_frame = tk.Label(self.left_frame, bg="black")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.label = tk.Label(self.left_frame, text="Detected Gesture: ", font=("Arial", 16), bg="#f0f0f0")
        self.label.pack(pady=10)
        
        # Right side - User selection and status
        self.user_frame = tk.LabelFrame(self.right_frame, text="Active User", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.user_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.user_label = tk.Label(self.user_frame, text=f"Current: {self.current_user}", font=("Arial", 12), bg="#f0f0f0")
        self.user_label.pack(side=tk.LEFT, padx=5, pady=10)
        
        self.toggle_user_button = ttk.Button(self.user_frame, text="Switch User", command=self.toggle_user)
        self.toggle_user_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Status frame
        self.status_frame = tk.LabelFrame(self.right_frame, text="Status", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_indicator = tk.Canvas(self.status_frame, width=20, height=20, bg="#f0f0f0", highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=10, pady=10)
        self.status_light = self.status_indicator.create_oval(2, 2, 18, 18, fill="red")
        
        self.status_label = tk.Label(self.status_frame, text="Ready to detect", font=("Arial", 12), bg="#f0f0f0")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Buffer frames - one for each user
        self.buffer_notebook = ttk.Notebook(self.right_frame)
        self.buffer_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a tab for each user
        for user in self.users:
            frame = tk.Frame(self.buffer_notebook, bg="#f0f0f0")
            self.buffer_notebook.add(frame, text=user)
            
            buffer_frame = tk.Frame(frame, bg="#f0f0f0")
            buffer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            label = tk.Label(buffer_frame, text="", font=("Arial", 14), 
                          bg="white", relief=tk.SUNKEN, width=20, height=4, wraplength=200)
            label.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
            self.users[user]["label"] = label
            
            # Add speak buffer button
            speak_button = ttk.Button(
                buffer_frame, 
                text="Speak Buffer", 
                command=lambda u=user: self.speak_buffer(u)
            )
            speak_button.pack(fill=tk.X, pady=5)
        
        # Control buttons
        self.button_frame = tk.Frame(self.right_frame, bg="#f0f0f0")
        self.button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(self.button_frame, text="Start Capture", command=self.start_capture)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        self.stop_button = ttk.Button(self.button_frame, text="Stop Capture", command=self.stop_capture)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        self.reset_button = ttk.Button(self.button_frame, text="Reset Buffer", command=self.reset_buffer)
        self.reset_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        self.send_button = ttk.Button(self.right_frame, text="Send Message", command=self.send_buffer)
        self.send_button.pack(fill=tk.X, padx=5, pady=10)
        
        # Chat history
        self.chat_frame = tk.LabelFrame(self.right_frame, text="Chat History", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chat_text = tk.Text(self.chat_frame, height=5, width=25, font=("Arial", 10), wrap=tk.WORD)
        self.chat_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_text.config(state=tk.DISABLED)
        
        # Add chat controls
        self.chat_controls = tk.Frame(self.chat_frame, bg="#f0f0f0")
        self.chat_controls.pack(fill=tk.X, padx=5, pady=2)
        
        self.speak_last_button = ttk.Button(
            self.chat_controls, 
            text="Speak Last Message", 
            command=self.speak_latest_message
        )
        self.speak_last_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.speak_all_button = ttk.Button(
            self.chat_controls, 
            text="Speak All Messages", 
            command=self.speak_chat_history
        )
        self.speak_all_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Initialize variables
        self.running = False
        self.last_registered_time = time.time()
        self.delay_time = 1.5
        self.cap = None
        self.photo = None  # Store the photo reference
        self.status_light_state = "red"  # Track the current state of the status light
        
        # Start message receiving thread
        Thread(target=self.receive_messages, daemon=True).start()

    def toggle_user(self):
        """Switch between users"""
        self.current_user = "User 2" if self.current_user == "User 1" else "User 1"
        self.user_label.config(text=f"Current: {self.current_user}")
        
        # Switch to the appropriate tab
        user_index = 0 if self.current_user == "User 1" else 1
        self.buffer_notebook.select(user_index)

    def start_capture(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return
                
            self.status_indicator.itemconfig(self.status_light, fill="green")
            self.status_label.config(text="Capture started")
            Thread(target=self.capture_gesture, daemon=True).start()

    def stop_capture(self):
        if self.running:
            self.running = False
            self.status_indicator.itemconfig(self.status_light, fill="red")
            self.status_label.config(text="Capture stopped")
    
    def reset_buffer(self):
        """Reset the current user's buffer"""
        self.users[self.current_user]["buffer"] = ""
        self.users[self.current_user]["label"].config(text="")

    def send_buffer(self):
        """Send the current user's buffer content"""
        buffer = self.users[self.current_user]["buffer"]
        if buffer:
            try:
                # Format the message with the user identifier
                message = f"{self.current_user}: {buffer}"
                client_socket.sendall(message.encode('utf-8'))
                
                # Add to the chat history
                self.chat_text.config(state=tk.NORMAL)
                self.chat_text.insert(tk.END, message + "\n")
                self.chat_text.see(tk.END)
                self.chat_text.config(state=tk.DISABLED)
                
                # Don't speak your own message when sending
                
                # Clear the buffer
                self.users[self.current_user]["buffer"] = ""
                self.users[self.current_user]["label"].config(text="")
                
                # Switch users automatically
                self.toggle_user()
            except Exception as e:
                print("Error sending data:", e)

    def append_to_buffer(self, gesture):
        """Add detected gesture to current user's buffer and speak the latest character"""
        if gesture == ' ':
            self.users[self.current_user]["buffer"] += " "
        elif gesture == '.':
            self.users[self.current_user]["buffer"] += "."
        else:
            self.users[self.current_user]["buffer"] += gesture
            # Optionally speak the detected character
            if self.tts_enabled.get():
                Thread(target=self.speak_message, args=(f"Detected {gesture}",), daemon=True).start()
            
        self.users[self.current_user]["label"].config(text=self.users[self.current_user]["buffer"])

    def capture_gesture(self):
        expected_features = 42
        
        # Create a separate thread for frame processing
        self.frame_ready = False
        self.processed_frame = None
        
        # Use a dedicated function for UI updates
        def update_ui():
            if not self.running:
                return
                
            if self.frame_ready and self.processed_frame is not None:
                # Convert the processed frame to PhotoImage
                img = Image.fromarray(self.processed_frame)
                img = img.resize((640, 480), Image.LANCZOS)
                
                # Keep a reference to prevent garbage collection
                self.photo = ImageTk.PhotoImage(image=img)
                self.video_frame.config(image=self.photo)
                
                # Reset flag
                self.frame_ready = False
                
            # Schedule the next update
            self.root.after(33, update_ui)  # ~30 FPS
        
        # Start the UI update cycle
        update_ui()
        
        try:
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Add user indicator to the frame
                cv2.rectangle(frame, (0, 0), (200, 40), (0, 0, 0), -1)
                cv2.putText(frame, f"Active: {self.current_user}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Process frame for hand detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(frame_rgb)
                
                current_time = time.time()
                cooldown_active = current_time - self.last_registered_time <= self.delay_time
                
                # Update status display only when necessary
                if cooldown_active:
                    if self.status_light_state != "orange":
                        self.root.after_idle(lambda: self.status_indicator.itemconfig(self.status_light, fill="orange"))
                        self.root.after_idle(lambda: self.status_label.config(text="Cooldown active"))
                        self.status_light_state = "orange"
                else:
                    if self.status_light_state != "green":
                        self.root.after_idle(lambda: self.status_indicator.itemconfig(self.status_light, fill="green"))
                        self.root.after_idle(lambda: self.status_label.config(text="Ready to detect"))
                        self.status_light_state = "green"
                
                # Process hand landmarks if detected
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Draw landmarks on the frame
                        mp_drawing.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Only process gesture if cooldown is complete
                        if not cooldown_active:
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
                                
                                # Draw the prediction on the frame
                                cv2.putText(frame, f"Gesture: {predicted_character}", 
                                          (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                
                                # Update UI using after_idle to ensure thread safety
                                self.root.after_idle(lambda p=predicted_character: self.label.config(text=f"Detected Gesture: {p}"))
                                self.root.after_idle(lambda p=predicted_character: self.append_to_buffer(p))
                                
                                # Update last registered time
                                self.last_registered_time = current_time
                            except Exception as e:
                                print(f"Prediction error: {e}")
                
                # Convert to RGB for Tkinter
                self.processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready = True
                
                # Small sleep to reduce CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error in capture_gesture: {e}")
        finally:
            if self.cap is not None:
                self.cap.release()
            self.running = False
            self.root.after_idle(lambda: self.status_indicator.itemconfig(self.status_light, fill="red"))
            self.root.after_idle(lambda: self.status_label.config(text="Capture ended"))

    def logout(self):
        """Handle user logout and return to login screen"""
        # Stop any ongoing capture
        if self.running:
            self.stop_capture()
        
        # Ask for confirmation
        if messagebox.askyesno("Logout", "Are you sure you want to logout?"):
            # Destroy the current window
            self.root.destroy()
            
            # Start a new login process
            start_application()

    # Add new TTS methods
    def speak_message(self, message):
        """Use TTS to speak a message"""
        if not self.tts_enabled.get():
            return
            
        try:
            # Run in a thread to avoid blocking UI
            def speak():
                tts_engine.say(message)
                tts_engine.runAndWait()
                
                # Reset status message when done speaking
                self.root.after_idle(lambda: self.status_label.config(
                    text="Ready to detect" if not self.running else "Capture started"))
                
            # Update status to show speaking
            self.status_label.config(text="Speaking message...")
            
            # Start speaking thread
            Thread(target=speak, daemon=True).start()
            
        except Exception as e:
            print(f"TTS error: {e}")
    
    def speak_buffer(self, user=None):
        """Speak the buffer content for a specific user or current user"""
        user = user or self.current_user
        buffer = self.users[user]["buffer"]
        
        if buffer:
            message = f"{user} buffer says: {buffer}"
            self.speak_message(message)
        else:
            self.speak_message(f"{user} buffer is empty")
            
    def speak_latest_message(self):
        """Speak the latest message in the chat history"""
        if not self.chat_text.get("1.0", tk.END).strip():
            self.speak_message("No messages in chat history")
            return
            
        # Get the last line of the chat
        chat_content = self.chat_text.get("1.0", tk.END)
        lines = [line for line in chat_content.splitlines() if line.strip()]
        
        if not lines:
            return
            
        last_message = lines[-1]
        self.speak_message(last_message)
    
    def speak_chat_history(self):
        """Speak all messages in the chat history"""
        chat_content = self.chat_text.get("1.0", tk.END).strip()
        if not chat_content:
            self.speak_message("No messages in chat history")
            return
            
        self.speak_message("Chat history: " + chat_content)
    
    def receive_messages(self):
        """Listen for incoming messages from the server"""
        while True:
            try:
                message = client_socket.recv(1024).decode('utf-8')
                if message:
                    # Add to chat history
                    self.root.after_idle(self.update_chat, message)
                    
                    # Speak the message if it's not from the current user
                    if not message.startswith(f"{self.current_user}:"):
                        self.speak_message(message)
            except Exception as e:
                print(f"Error receiving messages: {e}")
                break
    
    def update_chat(self, message):
        """Thread-safe update of chat history"""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, message + "\n")
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

def start_application():
    """Start the application with login flow"""
    # First, show the login window
    login_root = tk.Tk()
    login_app = LoginWindow(login_root)
    login_root.mainloop()
    
    # Only proceed if authentication was successful
    if login_app.authenticated:
        # Create main application window
        main_root = tk.Tk()
        app = GestureClientApp(main_root, username=login_app.username)
        
        # Clean up when window is closed
        def on_closing():
            app.stop_capture()
            if app.cap is not None:
                app.cap.release()
            main_root.destroy()
        
        main_root.protocol("WM_DELETE_WINDOW", on_closing)
        main_root.mainloop()

def main():
    start_application()

if __name__ == "__main__":
    main()
