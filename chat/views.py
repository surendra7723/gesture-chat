
import cv2

import pickle
import numpy as np
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from .models import Message


class ChatView(LoginRequiredMixin, TemplateView):
    template_name = "chat/chat.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['messages'] = Message.objects.order_by("-timestamp")[:50]
        return context


# Load the trained model
MODEL_PATH = "./model.p"

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)  # Corrected line


# OpenCV: Initialize webcam
cap = cv2.VideoCapture(0)

@method_decorator(csrf_exempt, name='dispatch')
class CaptureGestureView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            # Capture a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                return Response({"error": "Failed to capture image"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Preprocess frame (convert to grayscale, resize, flatten, etc.)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))  # Adjust based on your model input size
            features = resized.flatten().reshape(1, -1)  # Flatten to a 1D array

            # Predict the gesture
            prediction = model.predict(features)
            gesture_name = str(prediction[0])  # Convert numpy result to string

            return Response({"gesture": gesture_name})

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
