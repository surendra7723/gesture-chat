from django.urls import path
from django.contrib.auth.views import LoginView, LogoutView
from .views import chat_view,capture_gesture

urlpatterns = [
    path("chat/", chat_view, name="chat"),
    path('gesture/',capture_gesture,name='gesture'),
    path("", LoginView.as_view(template_name="chat/login.html"), name="login"),
    path("logout/", LogoutView.as_view(next_page="login"), name="logout"),
]
