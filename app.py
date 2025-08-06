from flask import Flask, render_template, request, jsonify, send_file
import os
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key, voices
import tempfile
import uuid
import json

# Import our advanced NLU engine
try:
    from advanced_nlu_engine import AdvancedVoiceAssistantNLU
    nlu_available = True
except ImportError as e:
    print(f"⚠️ Advanced NLU not available: {e}")
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

# Initialize NLU engine
if nlu_available:
    try:
        # Try advanced NLU first
        try:
            nlu_engine = AdvancedVoiceAssistantNLU()
            print("✅ Advanced NLU engine initialized successfully")
            nlu_type = "advanced_ml"
        except:
            # Fallback to basic NLU
            nlu_engine = VoiceAssistantNLU()
            print("✅ Basic NLU engine initialized successfully")
            nlu_type = "basic"
    except Exception as e:
        print(f"❌ Error initializing NLU engine: {e}")
        nlu_engine = None
        nlu_type = "none"
else:
    nlu_engine = None
    nlu_type = "none"

# Configuration - now more dynamic
user_name = "Alex"
schedule = "Sales Meeting with Taipy at 10:00; Gym with Sophie at 17:00"
base_prompt = f"You are a helpful AI assistant. Your user {user_name} has the following schedule: {schedule}. Help them manage their time and tasks effectively. Be conversational, friendly, and helpful."

@app.route('/')
def index():
    """Main page with voice assistant interface"""
    return render_template('index.html', 
                         user_name=user_name, 
                         schedule=schedule,
                         base_prompt=base_prompt,
                         nlu_available=nlu_available,
                         nlu_type=nlu_type)

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
            "message": f"Generated response: {assistant_response[:100]}..."
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def generate_assistant_response(user_input):
    """
    Fallback response generation when NLU is not available
    """
    user_input_lower = user_input.lower()
    
    # Time-related queries
    if any(word in user_input_lower for word in ['time', 'schedule', 'meeting', 'appointment']):
        if '10:00' in user_input_lower or 'sales' in user_input_lower:
            return f"Hi {user_name}! I see you have a Sales Meeting with Taipy at 10:00 today. Would you like me to help you prepare for it or check if there are any conflicts?"
        elif '17:00' in user_input_lower or 'gym' in user_input_lower:
            return f"Your gym session with Sophie is scheduled for 17:00. That's a great way to stay active! Would you like me to remind you about it or help you plan around it?"
        else:
            return f"Looking at your schedule, you have a Sales Meeting with Taipy at 10:00 and a gym session with Sophie at 17:00. How can I help you manage your time today?"
    
    # Greeting responses
    elif any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        return f"Hello {user_name}! How can I help you today? I can assist with your schedule, answer questions, or just chat with you."
    
    # Help requests
    elif any(word in user_input_lower for word in ['help', 'assist', 'support']):
        return f"I'm here to help you, {user_name}! I can help with your schedule, answer questions, generate ideas, or just have a conversation. What would you like to do?"
    
    # Weather or general chat
    elif any(word in user_input_lower for word in ['weather', 'day', 'nice', 'good']):
        return f"I hope you're having a great day, {user_name}! I'm here to help make it even better. What's on your mind?"
    
    # Task or to-do related
    elif any(word in user_input_lower for word in ['task', 'todo', 'remind', 'remember']):
        return f"I can help you manage tasks and reminders, {user_name}! Would you like me to help you organize your day or set up some reminders?"
    
    # Questions about the assistant
    elif any(word in user_input_lower for word in ['who are you', 'what can you do', 'your name']):
        return f"I'm your AI assistant, {user_name}! I'm here to help you manage your schedule, answer questions, and make your day more productive. I can also just chat with you!"
    
    # Default conversational response
    else:
        return f"That's interesting, {user_name}! I'm here to help you with your schedule, answer questions, or just chat. What would you like to focus on today?"

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
    """Handle conversational interactions with advanced NLU"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        conversation_history = data.get('history', [])
        
        if not user_message:
            return jsonify({"success": False, "error": "Please provide a message"})
        
        # Use NLU engine to understand and respond
        if nlu_engine:
            intent_data = nlu_engine.parse_intent(user_message)
            response_data = nlu_engine.get_response(intent_data, user_name)
            assistant_response = response_data['response']
            intent = response_data['intent']
            confidence = response_data['confidence']
            method = intent_data.get('method', 'unknown')
        else:
            assistant_response = generate_assistant_response(user_message)
            intent = "fallback"
            confidence = 0.5
            method = "fallback"
        
        # Add to conversation history
        conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "intent": intent,
            "confidence": confidence,
            "method": method,
            "timestamp": "now"
        })
        
        return jsonify({
            "success": True,
            "assistant_response": assistant_response,
            "intent": intent,
            "confidence": confidence,
            "method": method,
            "conversation_history": conversation_history
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

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