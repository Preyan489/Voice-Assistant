from flask import Flask, render_template, request, jsonify, send_file
import os
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key, voices
import tempfile
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Get API key from environment
API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize ElevenLabs
if API_KEY:
    set_api_key(API_KEY)

# Configuration
user_name = "Alex"
schedule = "Sales Meeting with Taipy at 10:00; Gym with Sophie at 17:00"
prompt = f"You are a helpful assistant. Your interlocutor has the following schedule: {schedule}. Help them manage their time and tasks effectively."
first_message = f"Hello {user_name}, how can I help you today?"

@app.route('/')
def index():
    """Main page with voice assistant interface"""
    return render_template('index.html', 
                         user_name=user_name, 
                         schedule=schedule,
                         first_message=first_message)

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
    """Generate speech from text"""
    try:
        data = request.get_json()
        text = data.get('text', first_message)
        voice_name = data.get('voice', 'Aria')
        
        # Generate audio
        audio = generate(
            text=text,
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
            "message": f"Generated speech: {text}"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

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
            "schedule": schedule
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 