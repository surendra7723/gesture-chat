from django.contrib import admin
from .models import GestureHistory, ChatRoom, RoomMembership, UserProfile


@admin.register(GestureHistory)
class GestureHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'gesture', 'confidence', 'timestamp']
    list_filter = ['gesture', 'timestamp']
    search_fields = ['user__username', 'gesture']


@admin.register(ChatRoom)
class ChatRoomAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_by', 'is_private', 'created_at']
    list_filter = ['is_private', 'created_at']
    search_fields = ['name', 'description']


@admin.register(RoomMembership)
class RoomMembershipAdmin(admin.ModelAdmin):
    list_display = ['room', 'user', 'is_admin', 'joined_at']
    list_filter = ['is_admin', 'joined_at']
    search_fields = ['room__name', 'user__username']


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'status', 'last_seen']
    list_filter = ['status']
    search_fields = ['user__username']
