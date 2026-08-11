import base64
import io
import pickle
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from datetime import datetime

from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.shortcuts import get_object_or_404

from chat.models import Message
from .models import GestureHistory, ChatRoom, RoomMembership, UserProfile
from .serializers import (
    UserSerializer, UserRegistrationSerializer, UserProfileSerializer,
    GestureHistorySerializer, MessageSerializer, ChatRoomSerializer,
    ChatRoomDetailSerializer, RoomMembershipSerializer,
    GesturePredictionSerializer, GesturePredictionResponseSerializer
)


# Load ML Model
MODEL_PATH = "./model.p"
try:
    with open(MODEL_PATH, "rb") as model_file:
        model_dict = pickle.load(model_file)
        ML_MODEL = model_dict.get('model') or model_dict.get('classifier') or list(model_dict.values())[0]
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")
    ML_MODEL = None

# MediaPipe setup
mp_hands = mp.solutions.hands
HANDS = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)


# ============= Authentication Views =============

@api_view(['POST'])
@permission_classes([AllowAny])
def register_user(request):
    """Register a new user"""
    serializer = UserRegistrationSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        return Response({
            'user': UserSerializer(user).data,
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def login_user(request):
    """Login user and return JWT tokens"""
    username = request.data.get('username')
    password = request.data.get('password')
    
    user = authenticate(username=username, password=password)
    if user:
        refresh = RefreshToken.for_user(user)
        return Response({
            'user': UserSerializer(user).data,
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        })
    return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_user(request):
    """Logout user (blacklist token if using token blacklist)"""
    try:
        refresh_token = request.data.get('refresh')
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
        return Response({'message': 'Successfully logged out'})
    except Exception:
        return Response({'message': 'Logged out'})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def current_user(request):
    """Get current user details"""
    serializer = UserSerializer(request.user)
    return Response(serializer.data)


# ============= User Profile Views =============

class UserProfileViewSet(viewsets.ModelViewSet):
    """ViewSet for user profiles"""
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return UserProfile.objects.filter(user=self.request.user)
    
    @action(detail=False, methods=['get', 'put', 'patch'])
    def me(self, request):
        """Get or update current user's profile"""
        profile, created = UserProfile.objects.get_or_create(user=request.user)
        
        if request.method == 'GET':
            serializer = self.get_serializer(profile)
            return Response(serializer.data)
        
        serializer = self.get_serializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ============= Gesture Recognition Views =============

def process_gesture_image(image_data):
    """Process image and predict gesture using MediaPipe and ML model"""
    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        img_array = np.array(image)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = HANDS.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            return None, "No hand detected"
        
        # Extract features
        hand_landmarks = results.multi_hand_landmarks[0]
        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]
        min_x, min_y = min(x_), min(y_)
        
        data_aux = [(lm.x - min_x, lm.y - min_y) for lm in hand_landmarks.landmark]
        data_aux_flat = [val for pair in data_aux for val in pair]
        
        # Ensure 42 features
        expected_features = 42
        if len(data_aux_flat) < expected_features:
            data_aux_flat.extend([0] * (expected_features - len(data_aux_flat)))
        
        # Predict
        if ML_MODEL:
            prediction = ML_MODEL.predict([np.asarray(data_aux_flat[:expected_features])])
            predicted_gesture = str(prediction[0])
            
            # Get confidence if available
            try:
                confidence = float(ML_MODEL.predict_proba([np.asarray(data_aux_flat[:expected_features])]).max())
            except:
                confidence = None
            
            return predicted_gesture, confidence
        else:
            return None, "Model not loaded"
            
    except Exception as e:
        return None, str(e)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def predict_gesture(request):
    """Predict gesture from uploaded image"""
    serializer = GesturePredictionSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    image_data = serializer.validated_data['image']
    save_history = serializer.validated_data.get('save_history', True)
    
    gesture, confidence = process_gesture_image(image_data)
    
    if gesture is None:
        return Response({'error': confidence}, status=status.HTTP_400_BAD_REQUEST)
    
    # Save to history if requested
    if save_history:
        GestureHistory.objects.create(
            user=request.user,
            gesture=gesture,
            confidence=confidence
        )
    
    response_data = {
        'gesture': gesture,
        'confidence': confidence,
        'timestamp': datetime.now()
    }
    
    response_serializer = GesturePredictionResponseSerializer(response_data)
    return Response(response_serializer.data)


class GestureHistoryViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for gesture history"""
    serializer_class = GestureHistorySerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return GestureHistory.objects.filter(user=self.request.user)

# ============= Chat Room Views =============

class ChatRoomViewSet(viewsets.ModelViewSet):
    """ViewSet for chat rooms"""
    permission_classes = [IsAuthenticated]
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return ChatRoomDetailSerializer
        return ChatRoomSerializer
    
    def get_queryset(self):
        if self.action in ['join', 'leave', 'messages']:
            return ChatRoom.objects.all()
        return ChatRoom.objects.filter(members=self.request.user)
    
    def perform_create(self, serializer):
        room = serializer.save(created_by=self.request.user)
        # Add creator as admin member
        RoomMembership.objects.create(room=room, user=self.request.user, is_admin=True)
    
    @action(detail=True, methods=['post'])
    def join(self, request, pk=None):
        """Join a chat room"""
        room = self.get_object()
        membership, created = RoomMembership.objects.get_or_create(
            room=room, user=request.user
        )
        if created:
            return Response({'message': f'Joined room {room.name}'})
        return Response({'message': 'Already a member'})
    
    @action(detail=True, methods=['post'])
    def leave(self, request, pk=None):
        """Leave a chat room"""
        room = self.get_object()
        try:
            membership = RoomMembership.objects.get(room=room, user=request.user)
            membership.delete()
            return Response({'message': f'Left room {room.name}'})
        except RoomMembership.DoesNotExist:
            return Response({'error': 'Not a member'}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """Get messages in a room"""
        room = self.get_object()
        messages = Message.objects.filter(room=room).order_by('-timestamp')[:50]
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)


# ============= Message Views =============

class MessageViewSet(viewsets.ModelViewSet):
    """ViewSet for messages"""
    serializer_class = MessageSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = Message.objects.all().order_by('-timestamp')
        room_id = self.request.query_params.get('room_id')
        if room_id:
            queryset = queryset.filter(room_id=room_id)
        return queryset
    
    def perform_create(self, serializer):
        room_id = self.request.data.get('room')
        if room_id:
            serializer.save(sender=self.request.user, room_id=room_id)
        else:
            serializer.save(sender=self.request.user)

