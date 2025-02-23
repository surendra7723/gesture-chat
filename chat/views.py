
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
def capture_gesture(request):
    if request.method == "POST":
        try:
            # Capture a frame from the webcam
            ret, frame = cap.read()
            # breakpoint()
            if not ret:
                return JsonResponse({"error": "Failed to capture image"}, status=500)

            # Preprocess frame (convert to grayscale, resize, flatten, etc.)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))  # Adjust based on your model input size
            features = resized.flatten().reshape(1, -1)  # Flatten to a 1D array

            # Predict the gesture
            prediction = model.predict(features)
            gesture_name = str(prediction[0])  # Convert numpy result to string
            
            return JsonResponse({"gesture": gesture_name})
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"message": "Send a POST request to capture a gesture."})



    