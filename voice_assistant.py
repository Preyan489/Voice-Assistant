import os
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key, voices

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Enhanced ChatGPT-like configuration
user_name = "Alex"
schedule = "Sales Meeting with Taipy at 10:00; Gym with Sophie at 17:00"

# Enhanced prompt for ChatGPT-like behavior
prompt = f"""You are a helpful AI assistant similar to ChatGPT. Your user {user_name} has the following schedule: {schedule}. 

You can:
- Help manage schedules and tasks
- Answer general questions thoughtfully
- Provide intelligent and helpful responses
- Engage in natural conversation
- Help with organization and productivity
- Respond to any type of user input appropriately
- Remember context from previous interactions
- Provide relevant suggestions and follow-ups

Be conversational, friendly, helpful, and intelligent. Understand context and provide relevant, useful responses. You're designed to be like ChatGPT - understanding, helpful, and engaging."""

# Enhanced first message
first_message = f"Hello {user_name}! I'm your AI assistant, designed to help you stay organized and productive. I can help with your schedule, answer questions, or just chat with you. How can I help you today?"

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
        
        # Test voice generation with enhanced message
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
        
        print(f"\n🎉 ElevenLabs API is working! Your ChatGPT-like assistant will say: '{first_message}'")
        print(f"📅 Schedule: {schedule}")
        print(f"🤖 Assistant Type: Advanced ChatGPT-like AI")
        print(f"💬 Capabilities: Natural conversation, schedule management, task organization, general assistance")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    print("Setting up ChatGPT-like ElevenLabs Voice Assistant...")
    test_elevenlabs_connection() 