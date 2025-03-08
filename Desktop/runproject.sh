#!/bin/bash

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate || { echo "Failed to activate virtual environment."; exit 1; }
else
    echo "Virtual environment is already activated."
fi

# Run server
echo "Starting server..."
python server/server.py &

# Wait for the server to initialize (adjust time if necessary)
sleep 1.6
# Run client
echo "Starting client..."
python client/client.py
