from django.db import models
from django.contrib.auth.models import User
from chat.models import Message


class GestureHistory(models.Model):
    """Track gesture recognition history for users"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='gesture_history')
    gesture = models.CharField(max_length=10)
    confidence = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    image_path = models.CharField(max_length=255, null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'Gesture Histories'
    
    def __str__(self):
        return f"{self.user.username} - {self.gesture} at {self.timestamp}"


class ChatRoom(models.Model):
    """Chat rooms for group conversations"""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_rooms')
    created_at = models.DateTimeField(auto_now_add=True)
    is_private = models.BooleanField(default=False)
    members = models.ManyToManyField(User, through='RoomMembership', related_name='chat_rooms')
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.name


class RoomMembership(models.Model):
    """Membership relationship between users and chat rooms"""
    room = models.ForeignKey(ChatRoom, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    joined_at = models.DateTimeField(auto_now_add=True)
    is_admin = models.BooleanField(default=False)
    
    class Meta:
        unique_together = ['room', 'user']
        ordering = ['-joined_at']
    
    def __str__(self):
        return f"{self.user.username} in {self.room.name}"


class UserProfile(models.Model):
    """Extended user profile information"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    avatar = models.CharField(max_length=255, blank=True)
    bio = models.TextField(blank=True)
    status = models.CharField(
        max_length=20, 
        choices=[
            ('online', 'Online'),
            ('offline', 'Offline'),
            ('away', 'Away'),
        ],
        default='offline'
    )
    last_seen = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s profile"
