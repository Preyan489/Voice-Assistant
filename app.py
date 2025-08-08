from flask import Flask, render_template, request, jsonify, send_file
import os
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key, voices
import tempfile
import uuid
import json
from datetime import datetime

# Import LLM conversation handler
try:
    from llm_conversation import LLMConversationHandler
    llm_available = True
except ImportError as e:
    print(f"⚠️ LLM conversation handler not available: {e}")
    llm_available = False
    
    # Fallback to basic NLU
    try:
        from nlu_engine import VoiceAssistantNLU
        nlu_available = True
    except ImportError as e:
        print(f"⚠️ Basic NLU not available: {e}")
        nlu_available = False

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Get API key from environment
API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize ElevenLabs
if API_KEY:
    set_api_key(API_KEY)

# Initialize conversation handler
if llm_available:
    try:
        conversation_handler = LLMConversationHandler()
        print("✅ LLM conversation handler initialized successfully")
        conversation_type = "llm"
    except Exception as e:
        print(f"❌ Error initializing LLM conversation handler: {e}")
        conversation_handler = None
        conversation_type = "none"
        
        # Try fallback to basic NLU
        if nlu_available:
            try:
                conversation_handler = VoiceAssistantNLU()
                print("✅ Basic NLU engine initialized as fallback")
                conversation_type = "basic"
            except Exception as e:
                print(f"❌ Error initializing fallback NLU: {e}")
else:
    conversation_handler = None
    conversation_type = "none"
    
    # Try fallback to basic NLU
    if nlu_available:
        try:
            conversation_handler = VoiceAssistantNLU()
            print("✅ Basic NLU engine initialized as fallback")
            conversation_type = "basic"
        except Exception as e:
            print(f"❌ Error initializing fallback NLU: {e}")

# Configuration - now more dynamic and ChatGPT-like
user_name = "Alex"
schedule = "Sales Meeting with Taipy at 10:00; Gym with Sophie at 17:00"
base_prompt = f"""You are a helpful AI assistant similar to ChatGPT. Your user {user_name} has the following schedule: {schedule}. 

You can:
- Help manage schedules and tasks
- Answer general questions
- Provide thoughtful responses
- Engage in natural conversation
- Help with organization and productivity
- Respond to any type of user input appropriately

Be conversational, friendly, helpful, and intelligent. Understand context and provide relevant, useful responses."""

@app.route('/')
def index():
    """Main page with voice assistant interface"""
    return render_template('index.html', 
                         user_name=user_name, 
                         schedule=schedule,
                         base_prompt=base_prompt,
                         conversation_available=conversation_handler is not None,
                         conversation_type=conversation_type)

@app.route('/api/voices')
def get_voices():
    """Get available voices"""
    try:
        available_voices = voices()
        voice_list = [{"name": voice.name, "category": voice.category} for voice in available_voices]
        return jsonify({"success": True, "voices": voice_list})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/generate', methods=['POST'])
def generate_speech():
    """Generate speech from user input with dynamic response using advanced NLU"""
    try:
        data = request.get_json()
        user_input = data.get('text', '').strip()
        voice_name = data.get('voice', 'Aria')
        
        if not user_input:
            return jsonify({"success": False, "error": "Please provide some text to generate speech"})
        
        # Use NLU engine to understand user intent and generate response
        if nlu_engine:
            # Parse intent using advanced NLU
            intent_data = nlu_engine.parse_intent(user_input)
            response_data = nlu_engine.get_response(intent_data, user_name)
            assistant_response = response_data['response']
            intent = response_data['intent']
            confidence = response_data['confidence']
            method = intent_data.get('method', 'unknown')
        else:
            # Fallback to simple response generation
            assistant_response = generate_assistant_response(user_input)
            intent = "fallback"
            confidence = 0.5
            method = "fallback"
        
        # Generate audio for the assistant's response
        audio = generate(
            text=assistant_response,
            voice=voice_name,
            model="eleven_monolingual_v1"
        )
        
        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        filename = f"voice_assistant_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(temp_dir, filename)
        save(audio, filepath)
        
        return jsonify({
            "success": True, 
            "filename": filename,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "intent": intent,
            "confidence": confidence,
            "method": method,
            "nlu_type": nlu_type,
            "nlu_info": {
                "intent": intent,
                "confidence": round(confidence, 2),
                "method": method,
                "nlu_type": nlu_type
            },
            "message": f"Generated response: {assistant_response[:100]}..."
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def generate_assistant_response(user_input):
    """
    Enhanced ChatGPT-like response generation when NLU is not available
    """
    user_input_lower = user_input.lower()
    
    # Time-related queries
    if any(word in user_input_lower for word in ['time', 'schedule', 'meeting', 'appointment']):
        if '10:00' in user_input_lower or 'sales' in user_input_lower:
            return f"Hi {user_name}! I see you have a Sales Meeting with Taipy at 10:00 today. Would you like me to help you prepare for it or check if there are any conflicts? I can also help you set up a reminder if you'd like."
        elif '17:00' in user_input_lower or 'gym' in user_input_lower:
            return f"Your gym session with Sophie is scheduled for 17:00. That's a great way to stay active! Would you like me to remind you about it or help you plan around it? I can also help you prepare for your workout."
        else:
            return f"Looking at your schedule, you have a Sales Meeting with Taipy at 10:00 and a gym session with Sophie at 17:00. How can I help you manage your time today? I can assist with preparation, reminders, or just help you stay organized."
    
    # Greeting responses
    elif any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        return f"Hello {user_name}! How can I help you today? I can assist with your schedule, answer questions, help you stay organized, or just chat with you. What's on your mind?"
    
    # Help requests
    elif any(word in user_input_lower for word in ['help', 'assist', 'support']):
        return f"I'm here to help you, {user_name}! I can help with your schedule, answer questions, generate ideas, help you stay organized, or just have a conversation. What would you like to focus on today?"
    
    # Weather or general chat
    elif any(word in user_input_lower for word in ['weather', 'day', 'nice', 'good']):
        return f"I hope you're having a great day, {user_name}! I'm here to help make it even better. Whether you need help with your schedule, want to chat, or have questions, I'm ready to assist. What's on your mind?"
    
    # Task or to-do related
    elif any(word in user_input_lower for word in ['task', 'todo', 'remind', 'remember']):
        return f"I can help you manage tasks and reminders, {user_name}! Would you like me to help you organize your day, set up some reminders, or just chat about what you need to get done? I'm here to help you stay productive."
    
    # Questions about the assistant
    elif any(word in user_input_lower for word in ['who are you', 'what can you do', 'your name']):
        return f"I'm your AI assistant, {user_name}! I'm designed to help you manage your schedule, answer questions, and make your day more productive. I can help with organization, time management, general conversation, and much more. I'm here to be your helpful companion!"
    
    # General questions
    elif any(word in user_input_lower for word in ['what is', 'how does', 'why', 'when', 'where', 'who']):
        return f"That's an interesting question, {user_name}! I'm designed to help with schedule management, task organization, and general assistance. While I have some limitations, I'm great at helping you stay organized and productive. What specific area would you like to focus on?"
    
    # Opinion requests
    elif any(word in user_input_lower for word in ['what do you think', 'your opinion', 'how do you feel']):
        return f"That's a thoughtful question, {user_name}! As an AI assistant, I'm designed to help you stay organized and productive. I think good time management and organization are key to success. What's your perspective on this?"
    
    # Personal questions about the assistant
    elif any(word in user_input_lower for word in ['how are you', 'are you ok', 'how do you feel']):
        return f"I'm doing well, {user_name}! I'm here and ready to help you with whatever you need. I'm excited to help you stay organized and productive. How are you doing today?"
    
    # Joke requests
    elif any(word in user_input_lower for word in ['joke', 'funny', 'humor', 'laugh']):
        return f"I'd love to share a joke with you, {user_name}! Here's one: Why don't scientists trust atoms? Because they make up everything! 😄 Now, how can I help you with your schedule or tasks today?"
    
    # Gratitude
    elif any(word in user_input_lower for word in ['thank you', 'thanks', 'appreciate']):
        return f"You're very welcome, {user_name}! I'm here to help make your day better. Is there anything else I can assist you with? I'm always ready to help you stay organized and productive."
    
    # Farewell
    elif any(word in user_input_lower for word in ['goodbye', 'bye', 'see you', 'farewell']):
        return f"Goodbye, {user_name}! Have a wonderful day ahead. I'll be here when you need me! Take care and stay organized!"
    
    # Default conversational response - more ChatGPT-like
    else:
        return f"That's interesting, {user_name}! I'm here to help you with your schedule, answer questions, help you stay organized, or just chat. I can assist with time management, task organization, reminders, and general conversation. What would you like to focus on today?"

@app.route('/api/audio/<filename>')
def get_audio(filename):
    """Serve generated audio files"""
    try:
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='audio/mpeg')
        else:
            return jsonify({"error": "Audio file not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def status():
    """Check API status"""
    try:
        available_voices = voices()
        return jsonify({
            "success": True,
            "api_key": "configured" if API_KEY else "missing",
            "voices_count": len(available_voices),
            "user_name": user_name,
            "schedule": schedule,
            "assistant_type": "advanced_conversational",
            "nlu_available": nlu_available,
            "nlu_type": nlu_type,
            "nlu_engine": nlu_type
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/conversation', methods=['POST'])
def conversation():
    """Handle conversational interactions with LLM or fallback NLU"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        conversation_history = data.get('history', [])
        
        if not user_message:
            return jsonify({"success": False, "error": "Please provide a message"})
        
        # Use conversation handler to get response
        if conversation_handler:
            if conversation_type == "llm":
                # Use LLM for conversation
                response_data = conversation_handler.get_response(user_message)
                assistant_response = response_data['response']
                intent = response_data['intent']
                confidence = response_data['confidence']
                method = "llm"
                suggestions = response_data.get('suggestions', [])
                conversation_history = response_data.get('conversation_history', [])
            else:
                # Use basic NLU
                intent_data = conversation_handler.parse_intent(user_message)
                response_data = conversation_handler.get_response(intent_data, user_name)
                assistant_response = response_data['response']
                intent = response_data['intent']
                confidence = response_data['confidence']
                method = "nlu"
                suggestions = get_conversation_suggestions(intent, user_message)
            
            # Add contextual information
            contextual_info = {
                'conversation_length': len(conversation_history) + 1,
                'recent_topics': [msg.get('intent', 'general_chat') for msg in conversation_history[-3:]] if conversation_history else [],
                'user_engagement': 'high' if len(conversation_history) > 2 else 'medium',
                'conversation_type': conversation_type
            }
        else:
            # Fallback to basic response generation
            assistant_response = generate_assistant_response(user_message)
            intent = "fallback"
            confidence = 0.5
            method = "fallback"
            contextual_info = {}
            suggestions = get_conversation_suggestions("general_chat", user_message)
        
        # Add to conversation history
        conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "intent": intent,
            "confidence": confidence,
            "method": method,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "contextual_info": contextual_info
        })
        
        return jsonify({
            "success": True,
            "assistant_response": assistant_response,
            "intent": intent,
            "confidence": confidence,
            "method": method,
            "nlu_info": {
                "intent": intent,
                "confidence": round(confidence, 2),
                "method": method,
                "nlu_type": nlu_type,
                "contextual_info": contextual_info
            },
            "conversation_history": conversation_history,
            "suggestions": get_conversation_suggestions(intent, user_message)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def get_conversation_suggestions(intent, user_message):
    """Generate conversation suggestions based on intent and context"""
    suggestions = []
    
    if intent == 'schedule_query':
        suggestions = [
            "Tell me more about my meetings",
            "Help me prepare for my appointments",
            "Set up a reminder for me",
            "What's my next task?"
        ]
    elif intent == 'task_help':
        suggestions = [
            "Help me organize my day",
            "Set up some reminders",
            "What should I prioritize?",
            "Help me plan my schedule"
        ]
    elif intent == 'general_chat':
        suggestions = [
            "Tell me a joke",
            "How can you help me?",
            "What's on my schedule?",
            "Help me stay organized"
        ]
    elif intent == 'greeting':
        suggestions = [
            "What's on my schedule today?",
            "Help me with my tasks",
            "Tell me about yourself",
            "How can you help me?"
        ]
    else:
        suggestions = [
            "What's on my schedule?",
            "Help me stay organized",
            "Tell me a joke",
            "How can you help me?"
        ]
    
    return suggestions

@app.route('/api/nlu/parse', methods=['POST'])
def parse_intent():
    """Parse user input using advanced NLU engine"""
    try:
        data = request.get_json()
        user_input = data.get('text', '').strip()
        
        if not user_input:
            return jsonify({"success": False, "error": "Please provide text to parse"})
        
        if nlu_engine:
            intent_data = nlu_engine.parse_intent(user_input)
            response_data = nlu_engine.get_response(intent_data, user_name)
            
            return jsonify({
                "success": True,
                "intent": intent_data['intent'],
                "confidence": intent_data['confidence'],
                "method": intent_data.get('method', 'unknown'),
                "slots": intent_data.get('slots', []),
                "response": response_data['response']
            })
        else:
            return jsonify({
                "success": False,
                "error": "NLU engine not available"
            })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/nlu/train', methods=['POST'])
def train_nlu():
    """Train the NLU model on new examples"""
    try:
        data = request.get_json()
        user_input = data.get('text', '').strip()
        expected_intent = data.get('intent', '').strip()
        
        if not user_input or not expected_intent:
            return jsonify({"success": False, "error": "Please provide both text and expected intent"})
        
        if nlu_engine and hasattr(nlu_engine, 'train_on_example'):
            success = nlu_engine.train_on_example(user_input, expected_intent)
            return jsonify({
                "success": success,
                "message": f"Trained on: '{user_input}' -> {expected_intent}" if success else "Training failed"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Training not supported by current NLU engine"
            })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 