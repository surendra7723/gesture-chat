
import cv2

import pickle
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Message
from django.contrib.auth import logout
from django.http import JsonResponse

@login_required
def chat_view(request):
    print(request)
    messages = Message.objects.order_by("-timestamp")[:50]  # Load last 50 messages
    return render(request, "chat/chat.html", {"messages": messages})

def logout_view(request):
    if request.method == "POST":
        logout(request)
        return JsonResponse({"message": "Logged out successfully!"})
    else:
        return JsonResponse({"error": "Method Not Allowed"}, status=405)
    

# Load the trained model
MODEL_PATH = "./model.p"

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)  # Corrected line


# OpenCV: Initialize webcam
cap = cv2.VideoCapture(0)

@csrf_exempt
# Replace the entire capture_gesture method

def capture_gesture(self):
    """Process video frames and detect hand gestures with improved stability"""
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
                
            # Create ROI (Region of Interest) for better hand detection
            h, w, _ = frame.shape
            roi_size = min(h, w) * 0.7  # Use 70% of the smaller dimension
            center_x, center_y = w // 2, h // 2
            roi_left = max(0, int(center_x - roi_size // 2))
            roi_top = max(0, int(center_y - roi_size // 2))
            roi_right = min(w, int(center_x + roi_size // 2))
            roi_bottom = min(h, int(center_y + roi_size // 2))
            
            # Draw ROI guide on frame
            cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), 
                         (0, 255, 255), 2)
            cv2.putText(frame, "Hand Detection Area", (roi_left + 10, roi_top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
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
                for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    # Validate hand detection confidence
                    if hasattr(result, 'multi_handedness') and i < len(result.multi_handedness):
                        hand_confidence = result.multi_handedness[i].classification[0].score
                        if hand_confidence < 0.85:  # Skip low-confidence detections
                            continue
                    
                    # Basic hand validation - check if landmarks form a reasonable hand shape
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    
                    # Check if hand is in a reasonable shape
                    dist_thumb_wrist = ((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)**0.5
                    if dist_thumb_wrist > 0.7:  # Hand is probably not valid
                        continue
                    
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
                        # Extract and normalize features
                        x_ = [lm.x for lm in hand_landmarks.landmark]
                        y_ = [lm.y for lm in hand_landmarks.landmark]
                        
                        # Calculate bounding box with padding
                        min_x, max_x = min(x_), max(x_)
                        min_y, max_y = min(y_), max(y_)
                        
                        # Calculate range with protection against zero division
                        x_range = max(max_x - min_x, 0.001)  # Avoid division by zero
                        y_range = max(max_y - min_y, 0.001)
                        
                        # Normalize coordinates to 0-1 range for consistent scaling
                        data_aux = []
                        for lm in hand_landmarks.landmark:
                            # Scale to 0-1 range
                            norm_x = (lm.x - min_x) / x_range
                            norm_y = (lm.y - min_y) / y_range
                            data_aux.append(norm_x)
                            data_aux.append(norm_y)
                        
                        # Flatten array for model input
                        data_aux_flat = data_aux
                        
                        # Ensure we have the expected number of features
                        if len(data_aux_flat) < expected_features:
                            data_aux_flat.extend([0] * (expected_features - len(data_aux_flat)))
                        elif len(data_aux_flat) > expected_features:
                            data_aux_flat = data_aux_flat[:expected_features]
                        
                        try:
                            # Make prediction
                            prediction = model.predict([np.asarray(data_aux_flat)])
                            predicted_character = str(prediction[0])
                            
                            # Add to recent predictions list for smoothing
                            self.last_predictions.append(predicted_character)
                            if len(self.last_predictions) > 5:  # Keep only most recent predictions
                                self.last_predictions.pop(0)
                            
                            # Check if we have enough consistent predictions
                            if len(self.last_predictions) >= self.prediction_threshold:
                                # Find most common prediction in the recent buffer
                                from collections import Counter
                                prediction_counts = Counter(self.last_predictions)
                                most_common = prediction_counts.most_common(1)[0]
                                most_common_char, count = most_common
                                
                                # Only register if the prediction is stable
                                if count >= self.prediction_threshold:
                                    # Draw the prediction on frame with confidence indicator
                                    cv2.putText(frame, f"Gesture: {most_common_char} (Confirmed)", 
                                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    
                                    # Only update UI if the prediction is reliable
                                    self.root.after_idle(lambda p=most_common_char: self.label.config(
                                        text=f"Detected Gesture: {p}"))
                                    self.root.after_idle(lambda p=most_common_char: self.append_to_buffer(p))
                                    
                                    # Update last registered time and reset predictions
                                    self.last_registered_time = current_time
                                    self.last_predictions = []  # Reset after confirming a gesture
                                else:
                                    # Show the current prediction but don't register it
                                    cv2.putText(frame, f"Gesture: {predicted_character} (Detecting...)", 
                                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                            else:
                                # Not enough consistent predictions yet
                                cv2.putText(frame, f"Gesture: {predicted_character}", 
                                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                            
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



    