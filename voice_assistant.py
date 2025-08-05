import os
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key, voices

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("ELEVENLABS_API_KEY")

# We are going to inform the assistant that the user has a schedule and prompt it to help the user.
# In this part you can customize:
# - The user's name: what the assistant will call the user.
# - The schedule: the user's schedule that the assistant will use to provide help.
# - The prompt: the message that the assistant will receive when the conversation starts to understand the context of the conversation.
# - The first message: the first message the assistant will say to the user.

# Prompts are used to provide context to the assistant and help it understand the user's needs.

# Here's my example:
user_name = "Alex"
schedule = "Sales Meeting with Taipy at 10:00; Gym with Sophie at 17:00"
prompt = f"You are a helpful assistant. Your interlocutor has the following schedule: {schedule}. Help them manage their time and tasks effectively."
first_message = f"Hello {user_name}, how can I help you today?"

def test_elevenlabs_connection():
    """
    Test the ElevenLabs API connection and basic functionality
    """
    try:
        # Set API key
        set_api_key(API_KEY)
        print("✅ API key set successfully")
        
        # Get available voices first
        print("📋 Getting available voices...")
        available_voices = voices()
        
        if not available_voices:
            print("❌ No voices available")
            return
        
        # Use the first available voice
        first_voice = available_voices[0]
        print(f"🎤 Using voice: {first_voice.name}")
        
        # Test voice generation
        print("🎤 Testing voice generation...")
        audio = generate(
            text=first_message,
            voice=first_voice.name,
            model="eleven_monolingual_v1"
        )
        
        # Save the audio file
        save(audio, "test_output.mp3")
        print("✅ Audio generated and saved as 'test_output.mp3'")
        
        # List available voices
        print("\n📋 Available voices:")
        for voice in available_voices[:5]:
            print(f"  - {voice.name} ({voice.category})")
        
        if len(available_voices) > 5:
            print(f"  ... and {len(available_voices) - 5} more voices")
        
        print(f"\n🎉 ElevenLabs API is working! Your assistant will say: '{first_message}'")
        print(f"📅 Schedule: {schedule}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    print("Setting up ElevenLabs API...")
    test_elevenlabs_connection() 