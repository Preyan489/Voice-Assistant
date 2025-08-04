# ElevenLabs API Project

This project demonstrates how to use the ElevenLabs text-to-speech API with Python.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Your API key is already configured in `config.py`

## Usage

Run the demo script to test the API:
```bash
python elevenlabs_demo.py
```

This will:
- Generate speech from text and save it as `output.mp3`
- List available voices

## Features

- Text-to-speech generation
- Voice selection
- Audio file export
- Error handling

## API Key Security

⚠️ **Important**: The API key is stored in `config.py` for demonstration purposes. In a production environment, you should:
- Use environment variables
- Never commit API keys to version control
- Use a `.env` file (added to `.gitignore`)

## Available Voices

The demo will show you available voices. You can change the voice in the script by modifying the `voice` parameter in the `generate()` function.

## Models

- `eleven_monolingual_v1`: Best for English text
- `eleven_multilingual_v1`: Supports multiple languages

## Next Steps

You can extend this project by:
- Creating a web interface
- Building a voice cloning application
- Implementing real-time speech synthesis
- Adding voice customization features 