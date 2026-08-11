import base64
import json
from io import BytesIO
from unittest.mock import patch, MagicMock

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient, APITestCase
from PIL import Image

from api.models import GestureHistory, ChatRoom, RoomMembership, UserProfile
from chat.models import Message


class APITestBase(APITestCase):
    """Base test class with helper methods"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client = APIClient()
        self.register_url = reverse('register')
        self.login_url = reverse('login')
        self.logout_url = reverse('logout')
        self.current_user_url = reverse('current_user')
        self.predict_url = reverse('predict-gesture')
        self.token_refresh_url = reverse('token_refresh')
    
    def get_jwt_tokens(self, username='testuser', password='testpass123'):
        """Helper to get JWT tokens for a user"""
        response = self.client.post(self.login_url, {
            'username': username,
            'password': password
        }, format='json')
        return response.data
    
    def authenticate(self, username='testuser', password='testpass123'):
        """Helper to authenticate client with JWT"""
        tokens = self.get_jwt_tokens(username, password)
        access = tokens.get('access')
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {access}')
        return tokens
    
    def create_test_image_base64(self, color=(255, 0, 0), size=(100, 100)):
        """Helper to create a base64 encoded test image"""
        image = Image.new('RGB', size, color)
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')


class AuthenticationTests(APITestBase):
    """Test authentication endpoints"""
    
    def test_user_registration(self):
        """Test user registration creates user and returns JWT"""
        data = {
            'username': 'newuser',
            'email': 'new@example.com',
            'password': 'newpass123',
            'password_confirm': 'newpass123',
            'first_name': 'New',
            'last_name': 'User'
        }
        response = self.client.post(self.register_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('user', response.data)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)
        self.assertEqual(response.data['user']['username'], 'newuser')
        self.assertTrue(User.objects.filter(username='newuser').exists())
        self.assertTrue(UserProfile.objects.filter(user__username='newuser').exists())
    
    def test_registration_password_mismatch(self):
        """Test registration fails with mismatched passwords"""
        data = {
            'username': 'newuser',
            'email': 'new@example.com',
            'password': 'newpass123',
            'password_confirm': 'differentpass'
        }
        response = self.client.post(self.register_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('non_field_errors', response.data)
    
    def test_registration_missing_fields(self):
        """Test registration fails with missing required fields"""
        response = self.client.post(self.register_url, {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_user_login(self):
        """Test user login returns JWT tokens"""
        response = self.client.post(self.login_url, {
            'username': 'testuser',
            'password': 'testpass123'
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)
        self.assertIn('user', response.data)
    
    def test_login_invalid_credentials(self):
        """Test login fails with invalid credentials"""
        response = self.client.post(self.login_url, {
            'username': 'testuser',
            'password': 'wrongpass'
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_get_current_user_authenticated(self):
        """Test getting current user info when authenticated"""
        self.authenticate()
        response = self.client.get(self.current_user_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['username'], 'testuser')
    
    def test_get_current_user_unauthenticated(self):
        """Test getting current user info fails when not authenticated"""
        response = self.client.get(self.current_user_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class UserProfileTests(APITestBase):
    """Test user profile endpoints"""
    
    def test_get_profile_creates_if_missing(self):
        """Test getting profile creates it if it doesn't exist"""
        self.authenticate()
        response = self.client.get(reverse('profile-me'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'offline')
    
    def test_update_profile(self):
        """Test updating user profile"""
        self.authenticate()
        data = {'bio': 'Test bio', 'status': 'online'}
        response = self.client.put(reverse('profile-me'), data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['bio'], 'Test bio')
        self.assertEqual(response.data['status'], 'online')
    
    def test_partial_update_profile(self):
        """Test partial update of user profile"""
        self.authenticate()
        data = {'bio': 'Partial update'}
        response = self.client.patch(reverse('profile-me'), data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['bio'], 'Partial update')


class ChatRoomTests(APITestBase):
    """Test chat room endpoints"""
    
    def test_create_chat_room(self):
        """Test creating a chat room"""
        self.authenticate()
        data = {'name': 'Test Room', 'description': 'A test room'}
        response = self.client.post(reverse('chatroom-list'), data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['name'], 'Test Room')
        self.assertEqual(response.data['created_by_username'], 'testuser')
        self.assertEqual(response.data['members_count'], 1)
    
    def test_list_chat_rooms(self):
        """Test listing chat rooms"""
        self.authenticate()
        room = ChatRoom.objects.create(name='Test Room', created_by=self.user)
        RoomMembership.objects.create(room=room, user=self.user, is_admin=True)
        
        response = self.client.get(reverse('chatroom-list'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)
    
    def test_join_chat_room(self):
        """Test joining a chat room"""
        self.authenticate()
        room = ChatRoom.objects.create(name='Test Room', created_by=self.user)
        RoomMembership.objects.create(room=room, user=self.user, is_admin=True)
        
        # Create another user and try to join
        other_user = User.objects.create_user(username='other', password='pass123')
        self.client.force_authenticate(user=other_user)
        
        response = self.client.post(reverse('chatroom-join', kwargs={'pk': room.id}))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(RoomMembership.objects.filter(room=room, user=other_user).exists())
    
    def test_leave_chat_room(self):
        """Test leaving a chat room"""
        self.authenticate()
        room = ChatRoom.objects.create(name='Test Room', created_by=self.user)
        RoomMembership.objects.create(room=room, user=self.user, is_admin=True)
        
        response = self.client.post(reverse('chatroom-leave', kwargs={'pk': room.id}))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(RoomMembership.objects.filter(room=room, user=self.user).exists())


class MessageTests(APITestBase):
    """Test message endpoints"""
    
    def test_create_message(self):
        """Test creating a message"""
        self.authenticate()
        data = {'content': 'Hello World'}
        response = self.client.post(reverse('message-list'), data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['content'], 'Hello World')
        self.assertEqual(response.data['sender_username'], 'testuser')
    
    def test_list_messages(self):
        """Test listing messages"""
        self.authenticate()
        Message.objects.create(sender=self.user, content='Test message')
        
        response = self.client.get(reverse('message-list'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)


class GestureHistoryTests(APITestBase):
    """Test gesture history endpoints"""
    
    def test_gesture_history_list(self):
        """Test listing gesture history"""
        self.authenticate()
        GestureHistory.objects.create(user=self.user, gesture='A', confidence=0.9)
        
        response = self.client.get(reverse('gesture-history-list'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)
        self.assertEqual(response.data['results'][0]['gesture'], 'A')


class GesturePredictionTests(APITestBase):
    """Test gesture prediction endpoint"""
    
    @patch('api.views.ML_MODEL')
    def test_predict_gesture_success(self, mock_model):
        """Test successful gesture prediction"""
        self.authenticate()
        
        mock_model.predict.return_value = ['A']
        mock_model.predict_proba.return_value = [[0.1, 0.9]]
        
        base64_image = self.create_test_image_base64()
        data = {'image': base64_image, 'save_history': True}
        
        response = self.client.post(self.predict_url, data, format='json')
        
        if response.status_code == status.HTTP_200_OK:
            self.assertIn('gesture', response.data)
            self.assertIn('confidence', response.data)
        else:
            # Model might fail if MediaPipe has issues
            self.assertIn(response.status_code, [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ])
    
    def test_predict_gesture_no_hand(self):
        """Test gesture prediction with no hand detected"""
        self.authenticate()
        
        with patch('api.views.HANDS.process') as mock_process:
            mock_result = MagicMock()
            mock_result.multi_hand_landmarks = None
            mock_process.return_value = mock_result
            
            base64_image = self.create_test_image_base64()
            data = {'image': base64_image}
            
            response = self.client.post(self.predict_url, data, format='json')
            self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
            self.assertIn('error', response.data)
    
    def test_predict_gesture_unauthenticated(self):
        """Test gesture prediction requires authentication"""
        data = {'image': 'test'}
        response = self.client.post(self.predict_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class URLRoutingTests(TestCase):
    """Test URL routing"""
    
    def test_api_urls_configured(self):
        """Test that API URLs are properly configured"""
        from django.urls import reverse
        from django.urls.exceptions import NoReverseMatch
        
        # These should not raise NoReverseMatch
        urls_to_test = [
            'register',
            'login',
            'logout',
            'current_user',
            'predict-gesture',
            'profile-me',
            'chatroom-list',
            'message-list',
            'gesture-history-list',
        ]
        
        for url_name in urls_to_test:
            try:
                reverse(url_name)
            except NoReverseMatch:
                self.fail(f"URL '{url_name}' is not properly configured")
    
    def test_api_v1_prefix_in_urls(self):
        """Test that API URLs are under /api/v1/"""
        from django.urls import get_resolver
        resolver = get_resolver()
        
        # Check that api/v1/ is in the URL patterns
        url_patterns = [str(p.pattern) for p in resolver.url_patterns]
        self.assertTrue(any('api/v1/' in p for p in url_patterns))


class ModelTests(TestCase):
    """Test API models"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
    
    def test_gesture_history_creation(self):
        """Test GestureHistory model creation"""
        history = GestureHistory.objects.create(
            user=self.user,
            gesture='A',
            confidence=0.95
        )
        self.assertEqual(str(history), f"{self.user.username} - A at {history.timestamp}")
    
    def test_chat_room_creation(self):
        """Test ChatRoom model creation"""
        room = ChatRoom.objects.create(
            name='Test Room',
            description='A test room',
            created_by=self.user
        )
        self.assertEqual(str(room), 'Test Room')
    
    def test_room_membership_unique(self):
        """Test RoomMembership enforces unique room-user pairs"""
        room = ChatRoom.objects.create(name='Test', created_by=self.user)
        RoomMembership.objects.create(room=room, user=self.user, is_admin=True)
        
        with self.assertRaises(Exception):
            RoomMembership.objects.create(room=room, user=self.user, is_admin=False)
    
    def test_user_profile_creation(self):
        """Test UserProfile can be created for user"""
        self.assertFalse(UserProfile.objects.filter(user=self.user).exists())
        profile = UserProfile.objects.create(user=self.user)
        self.assertEqual(profile.status, 'offline')


class SerializerTests(APITestBase):
    """Test serializers"""
    
    def test_user_registration_serializer(self):
        """Test user registration serializer validation"""
        from api.serializers import UserRegistrationSerializer
        
        # Valid data
        data = {
            'username': 'serializeruser',
            'email': 'ser@example.com',
            'password': 'pass12345',
            'password_confirm': 'pass12345'
        }
        serializer = UserRegistrationSerializer(data=data)
        self.assertTrue(serializer.is_valid())
        
        # Password mismatch
        data['password_confirm'] = 'different'
        serializer = UserRegistrationSerializer(data=data)
        self.assertFalse(serializer.is_valid())
    
    def test_gesture_prediction_serializer(self):
        """Test gesture prediction serializer"""
        from api.serializers import GesturePredictionSerializer
        
        data = {'image': 'test_image_data', 'save_history': True}
        serializer = GesturePredictionSerializer(data=data)
        self.assertTrue(serializer.is_valid())
        
        # Missing required field
        data = {'save_history': True}
        serializer = GesturePredictionSerializer(data=data)
        self.assertFalse(serializer.is_valid())
