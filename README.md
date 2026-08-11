# Gesture Chat

AI-powered communication translating ASL gestures to text/speech via Django + MediaPipe + scikit-learn. Includes real-time WebSocket chat and a JWT-authenticated REST API.

## Quick Start

 ```bash
 python -m venv .venv && source .venv/bin/activate
 uv pip install -r requirements.txt -r requirements-api.txt
 python manage.py migrate
 python manage.py runserver
 ```

- Web chat: `http://localhost:8000/`
- Admin: `http://localhost:8000/admin/`
- API docs: `http://localhost:8000/api/v1/docs/`

## Stack

Django 5.1.5 | DRF 3.15.2 | SimpleJWT 5.3.1 | Channels 4.2.0 | Daphne 4.1.2 | MediaPipe 0.10.14 | scikit-learn 1.5.2 | SQLite

## Tests

```bash
python manage.py test
```
