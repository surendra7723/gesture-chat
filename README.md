# Gesture Chat

An AI-powered communication application that translates American Sign Language (ASL) hand gestures into text and speech, enabling real-time chat-based interaction. Built on Django Channels for low-latency WebSocket communication and MediaPipe for client-side hand landmark detection, it couples a browser-based chat interface with a scikit-learn RandomForest classifier trained on 42 geometric features extracted per frame. The project ships both as a web app (ASGI/Daphne) and a standalone desktop client (Tkinter + TCP sockets), with gesture predictions routed either through a REST endpoint or directly into chat messages.

**Tech Stack:** Python 3.x | Django 5.1.5 | Channels 4.2.0 | Daphne 4.1.2 | OpenCV 4.10 | MediaPipe 0.10.14 | scikit-learn 1.5.2 | SQLite | Twisted 24.11.0

**Unit Testing:** `python manage.py test`

## Core Features

- **Real-time WebSocket Chat** — Low-latency messaging via Django Channels 4.2.0 and Daphne 4.1.2 (ASGI).
- **Gesture Capture API** — REST endpoint that accepts webcam frames, processes them with OpenCV + MediaPipe, and returns a predicted gesture via a trained RandomForest model.
- **ML-Powered Sign Recognition** — 42-feature extraction (21 hand landmarks × x, y) fed into scikit-learn 1.5.2 for A–Z, 0–9, space, and period classification.
- **Dual Deployment Modes** — Web interface for browser-based chat; standalone desktop client (Tkinter + TCP sockets) for peer-to-peer gesture chat with text-to-speech.
- **Authentication** — Django session-based login/logout with redirect flow.
- **Message History** — Last 50 messages loaded on chat render via Django ORM.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend Framework | Django 5.1.5 |
| Real-time Layer | Django Channels 4.2.0 + Daphne 4.1.2 |
| WebSocket Protocol | Autobahn / Twisted 24.11.0 |
| Database | SQLite (development) |
| Computer Vision | OpenCV 4.10.0.84 |
| Hand Tracking | MediaPipe 0.10.14 |
| Machine Learning | scikit-learn 1.5.2 (RandomForestClassifier) |
| Numerical Computing | NumPy 2.1.3, SciPy 1.15.1 |
| Desktop GUI | Tkinter |
| Text-to-Speech | pyttsx3 2.98 |
| Task Queue | InMemoryChannelLayer (dev); Redis 5.2.1 supported |

## Prerequisites

- Python 3.10+
- `uv` package manager
- SQLite (bundled with Python)
- Redis (optional, for production channel layer)
- Webcam (for gesture capture)

## Setup Guide

### Quick Install

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python manage.py migrate && python manage.py runserver
```

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/gesture-chat.git && cd gesture-chat
   ```

2. **Create virtual environment**
   ```bash
   uv venv
   source .venv/bin/activate   # Linux/macOS
   # .venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Apply database migrations**
   ```bash
   python manage.py migrate
   ```

5. **Create superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

6. **Run development server**
   ```bash
   python manage.py runserver
   ```

7. **Access the application**
   - Web chat: `http://localhost:8000/`
   - Admin panel: `http://localhost:8000/admin/`

## Desktop App

The desktop app is a standalone peer-to-peer gesture chat client built with Tkinter. It uses a raw TCP socket server for message broadcasting and runs gesture recognition entirely on the client side using OpenCV, MediaPipe, and a trained scikit-learn RandomForest model. Text-to-speech (pyttsx3) is integrated for accessibility.

### Prerequisites

- Python 3.10+
- Webcam
- Trained model file (`model.p`) — see note below

### Model Path Configuration

`Desktop/client/client.py` line 22 contains a hardcoded absolute path to the trained model:

```python
model_path = '/home/surendra/Code/College/gesture/Desktop/client/models/model.p'
```

Before running, update this to a relative path or set an environment variable so the client can locate `model.p` on any machine.

### Step 1 — Start the TCP Server

```bash
python Desktop/server/server.py
```

Expected output:
```
[*] Server started on 0.0.0.0:5000
```

### Step 2 — Start Desktop Clients

In separate terminal windows:

```bash
python Desktop/client/client.py
```

### Step 3 — Authenticate

Use the hardcoded demo credentials:

| Username | Password |
|----------|----------|
| `user1`  | `password1` |
| `user2`  | `password2` |

### Step 4 — Operate the Client

1. Click **Start Capture** to activate webcam gesture detection
2. Show ASL gestures (A–Z, 0–9, space, period) to the camera
3. Detected characters accumulate in the active user's buffer
4. Click **Send Message** to broadcast the buffer to all connected clients via TCP
5. Click **Switch User** to toggle between User 1 and User 2
6. Enable the **Speech** checkbox to activate TTS for detected characters and incoming messages

## API Overview

**Base URL:** `http://localhost:8000/`

### HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Login page |
| POST | `/logout/` | Logout user |
| GET | `/chat/` | Chat room (requires authentication) |
| POST | `/gesture/` | Capture webcam frame and return predicted gesture (JSON `{"gesture": "..."}`) |

### WebSocket Endpoints

| Path | Description |
|------|-------------|
| `ws/chat/` | Real-time chat messaging via `ChatConsumer` |

## Project Structure

```
gesture-chat/
├── chatapp/                          # Django project configuration
│   ├── __init__.py
│   ├── asgi.py                       # ASGI entry (Daphne + Channels)
│   ├── settings.py                   # Django settings
│   ├── urls.py                       # Root URL router
│   └── wsgi.py
├── chat/                             # Core Django app
│   ├── __init__.py
│   ├── admin.py                      # Admin registrations
│   ├── apps.py
│   ├── consumers.py                  # ChatConsumer (WebSocket)
│   ├── migrations/
│   │   ├── 0001_initial.py
│   │   └── __init__.py
│   ├── models.py                     # Message model
│   ├── routing.py                    # WebSocket URL routing
│   ├── static/
│   │   └── chat/
│   │       ├── chat.css
│   │       └── login.css
│   ├── templates/
│   │   └── chat/
│   │       ├── chat.html
│   │       └── login.html
│   ├── tests.py
│   ├── urls.py                       # App URL patterns
│   └── views.py                      # HTTP + gesture views
├── Desktop/                          # Standalone desktop client/server
│   ├── client/
│   │   ├── client.py                 # Tkinter client with TTS
│   │   └── models/
│   │       └── model.p               # Trained ML model
│   ├── server/
│   │   └── server.py                 # TCP socket server
│   └── requirements.txt
├── manage.py                         # Django management script
├── requirements.txt                  # Python dependencies
├── model.p                           # Trained RandomForest model
├── data.pickle                       # Extracted features dataset
├── collectImgs.py                    # Webcam image collection utility
├── createDataset.py                  # MediaPipe feature extraction
├── trainClassifier.py                # Model training & evaluation
├── main.py                           # Standalone Tkinter app entry
└── run.ps1                           # Windows run script
```
