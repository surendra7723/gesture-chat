# consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import AnonymousUser

from api.models import RoomMembership, ChatRoom

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope.get("user", AnonymousUser())
        if self.user.is_anonymous:
            await self.close()
            return

        path = self.scope.get("path", "")
        self.room_id = None
        self.room_group_name = "chat_general"

        if "/room/" in path:
            try:
                self.room_id = int(path.split("/room/")[-1].rstrip("/"))
                room = await self.get_room(self.room_id)
                if not room:
                    await self.close()
                    return
                is_member = await self.is_member(self.room_id, self.user.id)
                if not is_member:
                    await self.close()
                    return
                self.room_group_name = f"chat_room_{self.room_id}"
            except (ValueError, TypeError):
                await self.close()
                return

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json.get('message', '')

        if not message.strip():
            return

        if self.room_id:
            saved = await self.save_message(self.room_id, self.user.id, message)
            payload = {
                'type': 'chat_message',
                'message': message,
                'username': self.user.username,
                'room_id': self.room_id,
                'message_id': saved.id,
                'timestamp': saved.timestamp.isoformat(),
            }
        else:
            payload = {
                'type': 'chat_message',
                'message': message,
                'username': self.user.username,
            }

        await self.channel_layer.group_send(
            self.room_group_name,
            payload
        )

    async def chat_message(self, event):
        await self.send(text_data=json.dumps({
            'message': event['message'],
            'username': event.get('username', ''),
            'room_id': event.get('room_id'),
            'message_id': event.get('message_id'),
            'timestamp': event.get('timestamp'),
        }))

    @database_sync_to_async
    def get_room(self, room_id):
        try:
            return ChatRoom.objects.get(id=room_id)
        except ChatRoom.DoesNotExist:
            return None

    @database_sync_to_async
    def is_member(self, room_id, user_id):
        return RoomMembership.objects.filter(room_id=room_id, user_id=user_id).exists()

    @database_sync_to_async
    def save_message(self, room_id, user_id, content):
        return Message.objects.create(
            room_id=room_id,
            sender_id=user_id,
            content=content,
        )
