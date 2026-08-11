from rest_framework import serializers
from django.contrib.auth.models import User
from django.db import transaction

from chat.models import Message
from .models import GestureHistory, ChatRoom, RoomMembership, UserProfile


class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model"""
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'date_joined']
        read_only_fields = ['id', 'date_joined']


class UserRegistrationSerializer(serializers.ModelSerializer):
    """Serializer for user registration"""
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password_confirm', 'first_name', 'last_name']
    
    def validate(self, data):
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError("Passwords do not match")
        return data
    @transaction.atomic
    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(**validated_data)
        # Create user profile
        UserProfile.objects.create(user=user)
        return user


class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for UserProfile model"""
    username = serializers.CharField(source='user.username', read_only=True)
    email = serializers.CharField(source='user.email', read_only=True)
    
    class Meta:
        model = UserProfile
        fields = ['id', 'username', 'email', 'avatar', 'bio', 'status', 'last_seen']
        read_only_fields = ['id', 'last_seen']


class GestureHistorySerializer(serializers.ModelSerializer):
    """Serializer for GestureHistory model"""
    username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = GestureHistory
        fields = ['id', 'username', 'gesture', 'confidence', 'timestamp', 'image_path']
        read_only_fields = ['id', 'timestamp']


class MessageSerializer(serializers.ModelSerializer):
    """Serializer for Message model"""
    sender_username = serializers.CharField(source='sender.username', read_only=True)
    room_id = serializers.IntegerField(source='room.id', read_only=True)
    
    class Meta:
        model = Message
        fields = ['id', 'room', 'room_id', 'sender', 'sender_username', 'content', 'timestamp']
        read_only_fields = ['id', 'sender', 'timestamp']


class RoomMembershipSerializer(serializers.ModelSerializer):
    """Serializer for RoomMembership model"""
    username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = RoomMembership
        fields = ['id', 'user', 'username', 'joined_at', 'is_admin']
        read_only_fields = ['id', 'joined_at']


class ChatRoomSerializer(serializers.ModelSerializer):
    """Serializer for ChatRoom model"""
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    members_count = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatRoom
        fields = ['id', 'name', 'description', 'created_by', 'created_by_username', 
                  'created_at', 'is_private', 'members_count']
        read_only_fields = ['id', 'created_by', 'created_at']
    
    def get_members_count(self, obj):
        return obj.members.count()


class ChatRoomDetailSerializer(ChatRoomSerializer):
    """Detailed serializer for ChatRoom with members list"""
    members = RoomMembershipSerializer(source='roommembership_set', many=True, read_only=True)
    recent_messages = serializers.SerializerMethodField()
    
    class Meta(ChatRoomSerializer.Meta):
        fields = ChatRoomSerializer.Meta.fields + ['members', 'recent_messages']
    
    def get_recent_messages(self, obj):
        # Get recent messages for this room (placeholder - needs room field in Message model)
        return []


class GesturePredictionSerializer(serializers.Serializer):
    """Serializer for gesture prediction input"""
    image = serializers.CharField(help_text="Base64 encoded image or image URL")
    save_history = serializers.BooleanField(default=True)


class GesturePredictionResponseSerializer(serializers.Serializer):
    """Serializer for gesture prediction response"""
    gesture = serializers.CharField()
    confidence = serializers.FloatField(required=False)
    timestamp = serializers.DateTimeField()
