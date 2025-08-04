# ElevenLabs Voice Assistant

This project demonstrates how to use the ElevenLabs Conversation API to build a voice assistant with Python.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables:
   - Rename `env.example` to `.env`
   - Add your ElevenLabs API key to the `.env` file

## Usage

Run the voice assistant:
```bash
python voice_assistant.py
```

This will:
- Load your API credentials from environment variables
- Connect to your ElevenLabs agent
- Start a voice conversation session
- Handle user interactions and responses

## Features

- Voice conversation with ElevenLabs AI agent
- Customizable user schedule and prompts
- Real-time audio interface
- Response callbacks and transcript logging
- Interruption handling

## Configuration

You can customize:
- **User name**: What the assistant calls the user
- **Schedule**: User's schedule for task management
- **System prompt**: Context for the assistant
- **First message**: Initial greeting

## API Key Security

⚠️ **Important**: The API key is loaded from environment variables for security. Never commit API keys to version control.

## Project Structure

- `voice_assistant.py` - Main voice assistant implementation
- `config.py` - Environment variable loading
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (create from env.example)

## Next Steps

You can extend this project by:
- Adding more sophisticated conversation flows
- Implementing voice cloning features
- Creating a web interface
- Adding multi-language support
