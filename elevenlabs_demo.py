import os
from elevenlabs import generate, save, set_api_key
from config import ELEVENLABS_API_KEY

def main():
    # Set your API key
    set_api_key(ELEVENLABS_API_KEY)
    
    print("ElevenLabs API Demo")
    print("=" * 50)
    
    # Example 1: Generate speech from text
    print("\n1. Generating speech from text...")
    text = "Hello! This is a test of the ElevenLabs text-to-speech API. It's working great!"
    
    try:
        audio = generate(
            text=text,
            voice="Rachel",  # You can change this to any available voice
            model="eleven_monolingual_v1"
        )
        
        # Save the audio file
        save(audio, "output.mp3")
        print("✅ Audio generated and saved as 'output.mp3'")
        
    except Exception as e:
        print(f"❌ Error generating audio: {e}")
    
    # Example 2: List available voices
    print("\n2. Available voices:")
    try:
        from elevenlabs import voices
        available_voices = voices()
        
        for voice in available_voices[:5]:  # Show first 5 voices
            print(f"  - {voice.name} ({voice.category})")
        
        if len(available_voices) > 5:
            print(f"  ... and {len(available_voices) - 5} more voices")
            
    except Exception as e:
        print(f"❌ Error fetching voices: {e}")

if __name__ == "__main__":
    main() 