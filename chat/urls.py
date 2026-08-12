from django.urls import path
from django.contrib.auth.views import LoginView, LogoutView
from .views import ChatView, CaptureGestureView

urlpatterns = [
    path("chat/", ChatView.as_view(), name="chat"),
    path('gesture/',CaptureGestureView.as_view(),name='gesture'),
    path("", LoginView.as_view(template_name="chat/login.html"), name="login"),
    path("logout/", LogoutView.as_view(next_page="login"), name="logout"),
]
