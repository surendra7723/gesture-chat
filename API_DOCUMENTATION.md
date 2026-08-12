# Gesture Chat REST API

## Base URL
http://localhost:8000/api/v1/

## Authentication
Use JWT tokens. Include in header: `Authorization: Bearer <token>`

## Key Endpoints

### Authentication
- POST /auth/register/ - Register new user
- POST /auth/login/ - Login and get tokens
- POST /auth/token/refresh/ - Refresh access token
- GET /auth/user/ - Get current user

### Gestures
- POST /gestures/predict/ - Predict gesture from base64 image
- GET /gestures/history/ - Get user's gesture history

### Chat Rooms
- GET /rooms/ - List user's rooms
- POST /rooms/ - Create new room
- GET /rooms/{id}/ - Get room details
- POST /rooms/{id}/join/ - Join room
- POST /rooms/{id}/leave/ - Leave room

### Messages
- GET /messages/ - List messages
- POST /messages/ - Create message

### Profiles
- GET /profiles/me/ - Get/update current user profile

## Interactive Docs
- Swagger UI: http://localhost:8000/swagger/
- ReDoc: http://localhost:8000/redoc/

## Example
```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use token
curl -X GET http://localhost:8000/api/v1/auth/user/ \
  -H "Authorization: Bearer <token>"
```
