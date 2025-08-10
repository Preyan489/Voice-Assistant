from flask import Flask, render_template, request, jsonify, send_file
import os
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key, voices
import tempfile
import uuid
import json
from datetime import datetime
from typing import Dict, Any, List
import random

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
    """Generate speech from user input with enhanced conversational logic"""
    try:
        data = request.get_json()
        user_input = data.get('text', '').strip()
        voice_name = data.get('voice', 'Aria')
        context = data.get('context', {})
        
        if not user_input:
            return jsonify({"success": False, "error": "Please provide some text to generate speech"})
        
        # Use enhanced conversation handler to understand user intent and generate response
        if conversation_handler:
            if conversation_type == "llm":
                # Use enhanced LLM for conversation with context
                response_data = conversation_handler.get_response(user_input, context)
                assistant_response = response_data['response']
                intent = response_data['intent']
                confidence = response_data['confidence']
                method = "llm"
                suggestions = response_data.get('suggestions', [])
                conversation_stats = response_data.get('conversation_stats', {})
                analysis = response_data.get('analysis', {})
            else:
                # Use enhanced basic NLU
                intent_data = conversation_handler.parse_intent(user_input)
                response_data = conversation_handler.get_response(intent_data, user_name)
                assistant_response = response_data['response']
                intent = response_data['intent']
                confidence = response_data['confidence']
                method = "nlu"
                suggestions = response_data.get('suggestions', [])
                conversation_stats = response_data.get('conversation_context', {})
                analysis = response_data.get('context', {})
        else:
            # Fallback to enhanced response generation
            assistant_response = generate_enhanced_assistant_response(user_input, context)
            intent = "fallback"
            confidence = 0.5
            method = "fallback"
            suggestions = get_conversation_suggestions("general_chat", user_input)
            conversation_stats = {}
            analysis = {}
        
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
            "conversation_type": conversation_type,
            "suggestions": suggestions,
            "conversation_stats": conversation_stats,
            "analysis": analysis,
            "conversation_info": {
                "intent": intent,
                "confidence": round(confidence, 2),
                "method": method,
                "conversation_type": conversation_type,
                "suggestions": suggestions,
                "context": context
            },
            "message": f"Generated response: {assistant_response[:100]}..."
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def generate_enhanced_assistant_response(user_input: str, context: Dict[str, Any] = None) -> str:
    """
    Enhanced ChatGPT-like response generation when NLU is not available
    """
    user_input_lower = user_input.lower()
    context = context or {}
    
    # Enhanced time-related queries with context awareness
    if any(word in user_input_lower for word in ['time', 'schedule', 'meeting', 'appointment']):
        if '10:00' in user_input_lower or 'sales' in user_input_lower:
            return f"Hi {user_name}! I see you have a Sales Meeting with Taipy at 10:00 today. That's coming up soon! Would you like me to help you prepare for it, check for any conflicts, or set up a reminder? I'm here to make sure you're ready and on time."
        elif '17:00' in user_input_lower or 'gym' in user_input_lower:
            return f"Your gym session with Sophie is scheduled for 17:00. That's a great way to stay active and healthy! Would you like me to remind you about it, help you plan around it, or assist with workout preparation? I can help you stay on track with your fitness goals."
        else:
            return f"Looking at your schedule, you have a Sales Meeting with Taipy at 10:00 and a gym session with Sophie at 17:00. How can I help you manage your time effectively today? I can assist with preparation, reminders, scheduling optimization, or just help you stay organized and productive."
    
    # Enhanced greeting responses with context
    elif any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        time_greeting = "Good morning" if datetime.now().hour < 12 else "Good afternoon" if datetime.now().hour < 17 else "Good evening"
        return f"{time_greeting}, {user_name}! How can I help you today? I can assist with your schedule, answer questions, help you stay organized, or just chat with you. What's on your mind?"
    
    # Enhanced help requests with specific offerings
    elif any(word in user_input_lower for word in ['help', 'assist', 'support']):
        return f"I'm here to help you, {user_name}! I can help with your schedule management, task organization, time optimization, answering questions, generating ideas, or just having a conversation. What would you like to focus on today? I'm excited to assist you!"
    
    # Enhanced weather or general chat with personality
    elif any(word in user_input_lower for word in ['weather', 'day', 'nice', 'good']):
        return f"I hope you're having a wonderful day, {user_name}! I'm here to help make it even better. Whether you need help with your schedule, want to chat, have questions, or just need some assistance, I'm ready to help. What would you like to work on or discuss?"
    
    # Enhanced task or to-do related with actionable suggestions
    elif any(word in user_input_lower for word in ['task', 'todo', 'remind', 'remember']):
        return f"I can definitely help you manage tasks and reminders, {user_name}! Would you like me to help you organize your day, set up some reminders, create a to-do list, or prioritize your activities? I'm here to help you stay productive and organized. What specific area would you like to focus on?"
    
    # Enhanced questions about the assistant with capabilities
    elif any(word in user_input_lower for word in ['who are you', 'what can you do', 'your name']):
        return f"I'm your AI voice assistant, {user_name}! I'm designed to help you manage your schedule, answer questions, and make your day more productive. I can help with organization, time management, task prioritization, reminders, general conversation, and much more. I'm here to be your helpful companion and make your day run smoothly!"
    
    # Enhanced general questions with helpful context
    elif any(word in user_input_lower for word in ['what is', 'how does', 'why', 'when', 'where', 'who']):
        return f"That's an interesting question, {user_name}! I'm designed to help with schedule management, task organization, and general assistance. While I have some limitations, I'm great at helping you stay organized, productive, and on track. What specific area would you like to focus on? I'm here to help!"
    
    # Enhanced opinion requests with thoughtful responses
    elif any(word in user_input_lower for word in ['what do you think', 'your opinion', 'how do you feel']):
        return f"That's a thoughtful question, {user_name}! As an AI assistant, I'm designed to help you stay organized and productive. I think good time management, organization, and having a supportive system are key to success. What's your perspective on this? I'd love to hear your thoughts and help you implement strategies that work for you."
    
    # Enhanced personal questions about the assistant with personality
    elif any(word in user_input_lower for word in ['how are you', 'are you ok', 'how do you feel']):
        return f"I'm doing wonderfully, {user_name}! I'm here and ready to help you with whatever you need. I'm excited to help you stay organized and productive, and I'm always learning and improving to better serve you. How are you doing today? I'm here to support you!"
    
    # Enhanced joke requests with variety
    elif any(word in user_input_lower for word in ['joke', 'funny', 'humor', 'laugh']):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything! 😄",
            "What do you call a fake noodle? An impasta! 😄",
            "Why did the scarecrow win an award? Because he was outstanding in his field! 😄",
            "What do you call a bear with no teeth? A gummy bear! 😄"
        ]
        selected_joke = random.choice(jokes)
        return f"I'd love to share a joke with you, {user_name}! Here's one: {selected_joke} Now, how can I help you with your schedule or tasks today? I'm here to assist and keep you smiling!"
    
    # Enhanced gratitude with appreciation
    elif any(word in user_input_lower for word in ['thank you', 'thanks', 'appreciate']):
        return f"You're very welcome, {user_name}! I'm here to help make your day better and more productive. Is there anything else I can assist you with? I'm always ready to help you stay organized, answer questions, or just chat. I appreciate you taking the time to thank me!"
    
    # Enhanced farewell with warmth
    elif any(word in user_input_lower for word in ['goodbye', 'bye', 'see you', 'farewell']):
        return f"Goodbye, {user_name}! Have a wonderful and productive day ahead. I'll be here when you need me again! Take care, stay organized, and remember I'm always ready to help. See you soon!"
    
    # Enhanced default conversational response with more personality
    else:
        return f"That's interesting, {user_name}! I'm here to help you with your schedule, answer questions, help you stay organized, or just chat. I can assist with time management, task organization, reminders, general conversation, and much more. What would you like to focus on today? I'm excited to help you make the most of your day!"

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
            "conversation_available": conversation_handler is not None,
            "conversation_type": conversation_type,
            "llm_available": llm_available
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/conversation', methods=['POST'])
def conversation():
    """Handle enhanced conversational interactions with LLM or fallback NLU"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        conversation_history = data.get('history', [])
        context = data.get('context', {})
        
        if not user_message:
            return jsonify({"success": False, "error": "Please provide a message"})
        
        # Use enhanced conversation handler to get response
        if conversation_handler:
            if conversation_type == "llm":
                # Use enhanced LLM for conversation with context
                response_data = conversation_handler.get_response(user_message, context)
                assistant_response = response_data['response']
                intent = response_data['intent']
                confidence = response_data['confidence']
                method = "llm"
                suggestions = response_data.get('suggestions', [])
                conversation_history = response_data.get('conversation_history', [])
                conversation_stats = response_data.get('conversation_stats', {})
                analysis = response_data.get('analysis', {})
            else:
                # Use enhanced basic NLU
                intent_data = conversation_handler.parse_intent(user_message)
                response_data = conversation_handler.get_response(intent_data, user_name)
                assistant_response = response_data['response']
                intent = response_data['intent']
                confidence = response_data['confidence']
                method = "nlu"
                suggestions = response_data.get('suggestions', [])
                conversation_stats = response_data.get('conversation_context', {})
                analysis = response_data.get('context', {})
            
            # Add enhanced contextual information
            contextual_info = {
                'conversation_length': len(conversation_history) + 1,
                'recent_topics': [msg.get('intent', 'general_chat') for msg in conversation_history[-3:]] if conversation_history else [],
                'user_engagement': 'high' if len(conversation_history) > 2 else 'medium',
                'conversation_type': conversation_type,
                'context': context,
                'analysis': analysis,
                'conversation_stats': conversation_stats
            }
        else:
            # Fallback to enhanced response generation
            assistant_response = generate_enhanced_assistant_response(user_message, context)
            intent = "fallback"
            confidence = 0.5
            method = "fallback"
            contextual_info = {'conversation_type': 'fallback'}
            suggestions = get_conversation_suggestions("general_chat", user_message)
            conversation_stats = {}
            analysis = {}
        
        # Add to conversation history with enhanced metadata
        conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "intent": intent,
            "confidence": confidence,
            "method": method,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "contextual_info": contextual_info,
            "suggestions": suggestions,
            "analysis": analysis
        })
        
        return jsonify({
            "success": True,
            "assistant_response": assistant_response,
            "intent": intent,
            "confidence": confidence,
            "method": method,
            "conversation_info": {
                "intent": intent,
                "confidence": round(confidence, 2),
                "method": method,
                "conversation_type": conversation_type,
                "contextual_info": contextual_info,
                "suggestions": suggestions,
                "analysis": analysis
            },
            "conversation_history": conversation_history,
            "suggestions": suggestions,
            "conversation_stats": conversation_stats
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def get_conversation_suggestions(intent: str, user_input: str = "") -> List[str]:
    """Generate contextual suggestions for conversation continuation"""
    suggestions = []
    
    # Schedule-related suggestions
    if intent in ['schedule_query', 'meeting_query', 'gym_query']:
        suggestions.extend([
            "What's my next meeting?",
            "When is my gym session?",
            "Help me prepare for the meeting",
            "Set a reminder for my appointments"
        ])
    
    # Task-related suggestions
    elif intent in ['task_help', 'reminder_request']:
        suggestions.extend([
            "What should I prioritize today?",
            "Set a reminder for me",
            "Help me organize my day",
            "Create a to-do list"
        ])
    
    # Entertainment suggestions
    elif intent == 'entertainment':
        suggestions.extend([
            "Tell me another joke",
            "Share a motivational quote",
            "Tell me a short story",
            "Play a word game"
        ])
    
    # Emotional support suggestions
    elif intent == 'emotional_support':
        suggestions.extend([
            "How can I help you feel better?",
            "Would you like to talk about it?",
            "I'm here to listen and support you",
            "Let's focus on something positive"
        ])
    
    # General suggestions
    else:
        suggestions.extend([
            "What's on my schedule today?",
            "Help me stay organized",
            "Tell me something interesting",
            "How can you help me?"
        ])
    
    return suggestions[:4]  # Return top 4 suggestions

@app.route('/api/nlu/parse', methods=['POST'])
def parse_intent():
    """Parse user input using enhanced conversation handler"""
    try:
        data = request.get_json()
        user_input = data.get('text', '').strip()
        context = data.get('context', {})
        
        if not user_input:
            return jsonify({"success": False, "error": "Please provide text to parse"})
        
        if conversation_handler:
            if conversation_type == "llm":
                # Use enhanced LLM for parsing with context
                response_data = conversation_handler.get_response(user_input, context)
                intent = response_data['intent']
                confidence = response_data['confidence']
                method = "llm"
                response = response_data['response']
                analysis = response_data.get('analysis', {})
                suggestions = response_data.get('suggestions', [])
            else:
                # Use enhanced basic NLU
                intent_data = conversation_handler.parse_intent(user_input)
                response_data = conversation_handler.get_response(intent_data, user_name)
                intent = response_data['intent']
                confidence = response_data['confidence']
                method = "nlu"
                response = response_data['response']
                analysis = response_data.get('context', {})
                suggestions = response_data.get('suggestions', [])
            
            return jsonify({
                "success": True,
                "intent": intent,
                "confidence": confidence,
                "method": method,
                "slots": [],
                "response": response,
                "analysis": analysis,
                "suggestions": suggestions
            })
        else:
            return jsonify({
                "success": False,
                "error": "Conversation handler not available"
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
        
        if conversation_handler and hasattr(conversation_handler, 'train_on_example'):
            success = conversation_handler.train_on_example(user_input, expected_intent)
            return jsonify({
                "success": success,
                "message": f"Trained on: '{user_input}' -> {expected_intent}" if success else "Training failed"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Training not supported by current conversation handler"
            })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/conversation/insights', methods=['GET'])
def get_conversation_insights():
    """Get insights about conversation patterns and performance"""
    try:
        if conversation_handler:
            if conversation_type == "llm":
                insights = conversation_handler.get_conversation_summary()
            else:
                insights = conversation_handler.get_conversation_insights()
            
            return jsonify({
                "success": True,
                "insights": insights,
                "conversation_type": conversation_type
            })
        else:
            return jsonify({
                "success": False,
                "error": "Conversation handler not available"
            })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/conversation/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation history and context"""
    try:
        if conversation_handler:
            if hasattr(conversation_handler, 'clear_history'):
                conversation_handler.clear_history()
            if hasattr(conversation_handler, 'reset_conversation_context'):
                conversation_handler.reset_conversation_context()
            
            return jsonify({
                "success": True,
                "message": "Conversation history and context reset successfully"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Conversation handler not available"
            })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/conversation/preferences', methods=['GET', 'POST'])
def manage_preferences():
    """Get or set user conversation preferences"""
    try:
        if not conversation_handler:
            return jsonify({
                "success": False,
                "error": "Conversation handler not available"
            })
        
        if request.method == 'GET':
            # Get current preferences
            if hasattr(conversation_handler, 'get_user_preferences'):
                preferences = conversation_handler.get_user_preferences()
            else:
                preferences = {}
            
            return jsonify({
                "success": True,
                "preferences": preferences
            })
        
        elif request.method == 'POST':
            # Set new preferences
            data = request.get_json()
            key = data.get('key')
            value = data.get('value')
            
            if not key or value is None:
                return jsonify({
                    "success": False,
                    "error": "Please provide both key and value"
                })
            
            if hasattr(conversation_handler, 'set_user_preference'):
                conversation_handler.set_user_preference(key, value)
                return jsonify({
                    "success": True,
                    "message": f"Preference '{key}' set successfully"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Preference setting not supported"
                })
                
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 