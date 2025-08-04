import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from elevenlabs.types import ConversationConfig

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

def setup_conversation():
    """
    Set up the ElevenLabs client and configure a conversation instance.
    We will then configure the conversation with the agent's first message and system prompt.
    """
    # Initialize the ElevenLabs client
    client = ElevenLabs(api_key=API_KEY)
    
    # Underneath in the same file, we are then going to set this configuration to our ElevenLabs agent:
    conversation_override = {
        "agent": {
            "prompt": {"prompt": prompt,},
            "first_message": first_message
        }
    }
    
    config = ConversationConfig(
        conversation_config_override=conversation_override,
        extra_body={},
        dynamic_variables={}
    )
    
    # Create the conversation instance
    conversation = Conversation(
        client=client,
        AGENT_ID="agent_7501k1v287vsetq9b32khpph8er3",
        config=config,
        requires_auth=True,
        audio_interface=DefaultAudioInterface()
    )
    
    return conversation

# 3. Implement Callbacks for Responses
# We'll also need to handle assistant responses by printing the assistant's responses and user transcripts, 
# as well as handling the situation where the user interrupts the assistant. 
# We can do so by implementing a few callback functions underneath our configuration.

def print_agent_response(response):
    print(f"Agent: {response}")

def print_interrupted_response(original, corrected):
    print(f"Agent interrupted, truncated response: {corrected}")

def print_user_transcript(transcript):
    print(f"User: {transcript}")

if __name__ == "__main__":
    print("Setting up ElevenLabs Conversation API...")
    
    # Initialize the ElevenLabs client
    client = ElevenLabs(api_key=API_KEY)
    
    # Underneath in the same file, we are then going to set this configuration to our ElevenLabs agent:
    conversation_override = {
        "agent": {
            "prompt": {"prompt": prompt,},
            "first_message": first_message
        }
    }
    
    config = ConversationConfig(
        conversation_config_override=conversation_override,
        extra_body={},
        dynamic_variables={}
    )
    
    # 4. Start the Voice Assistant Session
    # Finally, initiate the conversation session in the same file:
    conversation = Conversation(
        client,
        "agent_7501k1v287vsetq9b32khpph8er3",
        config=config,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        callback_agent_response=print_agent_response,
        callback_agent_response_correction=print_interrupted_response,
        callback_user_transcript=print_user_transcript,
    )

    print("✅ Conversation setup complete!")
    print("🎤 Starting voice assistant session...")
    conversation.start_session() 