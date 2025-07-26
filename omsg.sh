#!/bin/bash

# One-shot Ollama message script
# Usage: omsg "your message here"

if [ $# -eq 0 ]; then
    echo "Usage: omsg \"your message here\""
    exit 1
fi

# Combine all arguments into a single message
MESSAGE="$*"

# Escape double quotes in the message for JSON
MESSAGE_ESCAPED=$(echo "$MESSAGE" | sed 's/"/\\"/g')

# Send curl request to Ollama API inside the Docker container and extract just the response text
docker exec open-webui curl -s http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mistral:7b",
        "prompt": "'"$MESSAGE_ESCAPED"'",
        "stream": false
    }' | jq -r '.response' 2>/dev/null || echo "Error: Failed to get response from Ollama"