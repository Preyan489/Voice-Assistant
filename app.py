from flask import Flask, render_template, request, jsonify, send_file
import os
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key, voices
import tempfile
import uuid
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Get API key from environment
API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize ElevenLabs
if API_KEY:
    set_api_key(API_KEY)

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
                         base_prompt=base_prompt)

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
    """Generate speech from user input with dynamic response"""
    try:
        data = request.get_json()
        user_input = data.get('text', '').strip()
        voice_name = data.get('voice', 'Aria')
        
        if not user_input:
            return jsonify({"success": False, "error": "Please provide some text to generate speech"})
        
        # Generate dynamic response based on user input
        assistant_response = generate_assistant_response(user_input)
        
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
            "message": f"Generated response: {assistant_response[:100]}..."
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def generate_assistant_response(user_input):
    """
    Generate a contextual response based on user input
    This simulates an AI assistant that understands context and responds appropriately
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
            "assistant_type": "dynamic_conversational"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/conversation', methods=['POST'])
def conversation():
    """Handle conversational interactions"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        conversation_history = data.get('history', [])
        
        if not user_message:
            return jsonify({"success": False, "error": "Please provide a message"})
        
        # Generate contextual response
        assistant_response = generate_assistant_response(user_message)
        
        # Add to conversation history
        conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": "now"
        })
        
        return jsonify({
            "success": True,
            "assistant_response": assistant_response,
            "conversation_history": conversation_history
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 