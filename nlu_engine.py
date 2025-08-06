import json
import re
import random
from datetime import datetime

class VoiceAssistantNLU:
    def __init__(self):
        self.initialize_patterns()
        print("✅ Custom NLU engine initialized successfully")
    
    def initialize_patterns(self):
        """Initialize intent patterns and training data"""
        
        # Define intent patterns with regex and keywords
        self.intent_patterns = {
            'greeting': {
                'patterns': [
                    r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                    r'\b(how are you|what\'s up)\b'
                ],
                'keywords': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you', "what's up"]
            },
            
            'schedule_query': {
                'patterns': [
                    r'\b(what\'s on my schedule|what\'s my schedule|what do I have today)\b',
                    r'\b(what meetings|what appointments|what\'s planned)\b',
                    r'\b(show me my schedule|tell me about my schedule)\b'
                ],
                'keywords': ['schedule', 'meeting', 'appointment', 'planned', 'today', 'have']
            },
            
            'meeting_query': {
                'patterns': [
                    r'\b(sales meeting|meeting with taipy|meeting at 10)\b',
                    r'\b(when is my sales meeting|what time is the meeting)\b'
                ],
                'keywords': ['sales', 'meeting', 'taipy', '10:00', '10 am']
            },
            
            'gym_query': {
                'patterns': [
                    r'\b(gym session|gym with sophie|workout)\b',
                    r'\b(when is my gym|what time is gym)\b'
                ],
                'keywords': ['gym', 'workout', 'sophie', '17:00', '5 pm', 'exercise']
            },
            
            'task_help': {
                'patterns': [
                    r'\b(help me with tasks|need help with tasks|help me organize)\b',
                    r'\b(help me manage|task assistance|to-do list)\b'
                ],
                'keywords': ['help', 'tasks', 'organize', 'manage', 'assistance', 'to-do']
            },
            
            'reminder_request': {
                'patterns': [
                    r'\b(remind me|set a reminder|need a reminder)\b',
                    r'\b(can you remind me|set up a reminder)\b'
                ],
                'keywords': ['remind', 'reminder', 'set up', 'remember']
            },
            
            'assistant_info': {
                'patterns': [
                    r'\b(who are you|what can you do|what\'s your name)\b',
                    r'\b(tell me about yourself|what are your capabilities)\b'
                ],
                'keywords': ['who are you', 'what can you do', 'your name', 'capabilities', 'yourself']
            },
            
            'general_chat': {
                'patterns': [
                    r'\b(how\'s your day|how are you doing|what\'s the weather)\b',
                    r'\b(nice day|good day|how\'s it going)\b'
                ],
                'keywords': ['weather', 'day', 'nice', 'good', 'going']
            },
            
            'time_query': {
                'patterns': [
                    r'\b(what time is it|what\'s the time|current time)\b',
                    r'\b(time now|what time do I have|next appointment)\b'
                ],
                'keywords': ['time', 'when', 'current', 'now', 'appointment']
            }
        }
        
        # Define response templates
        self.response_templates = {
            'greeting': [
                "Hello {user_name}! How can I help you today?",
                "Hi {user_name}! Great to see you. What can I assist you with?",
                "Hey {user_name}! I'm here to help. What's on your mind?"
            ],
            
            'schedule_query': [
                "Looking at your schedule, {user_name}, you have a Sales Meeting with Taipy at 10:00 and a gym session with Sophie at 17:00. How can I help you manage your time today?",
                "Your schedule today includes a Sales Meeting with Taipy at 10:00 and gym with Sophie at 17:00. Would you like me to help you prepare for any of these?",
                "Today you have: Sales Meeting with Taipy at 10:00, and gym session with Sophie at 17:00. What would you like to know more about?"
            ],
            
            'meeting_query': [
                "You have a Sales Meeting with Taipy at 10:00 today, {user_name}. Would you like me to help you prepare for it or check if there are any conflicts?",
                "Your Sales Meeting with Taipy is scheduled for 10:00. Should I set up a reminder for you?",
                "The Sales Meeting with Taipy is at 10:00. Is there anything specific you'd like to know about it?"
            ],
            
            'gym_query': [
                "Your gym session with Sophie is scheduled for 17:00, {user_name}. That's a great way to stay active! Would you like me to remind you about it?",
                "You have gym with Sophie at 17:00. Would you like me to help you plan around it or set a reminder?",
                "Your workout session with Sophie is at 17:00. Should I help you prepare for it?"
            ],
            
            'task_help': [
                "I can help you manage tasks and reminders, {user_name}! Would you like me to help you organize your day or set up some reminders?",
                "I'm here to help with your tasks, {user_name}! I can assist with scheduling, reminders, and organization. What would you like to focus on?",
                "Let me help you with your tasks, {user_name}! I can help organize your schedule, set reminders, or just chat about what you need to get done."
            ],
            
            'reminder_request': [
                "I can help you set up reminders, {user_name}! What would you like me to remind you about and when?",
                "Sure, {user_name}! I'd be happy to set a reminder for you. What should I remind you about?",
                "Reminders are one of my specialties, {user_name}! What do you need to be reminded about?"
            ],
            
            'assistant_info': [
                "I'm your AI assistant, {user_name}! I'm here to help you manage your schedule, answer questions, and make your day more productive. I can also just chat with you!",
                "I'm your personal AI assistant, {user_name}! I can help with your schedule, tasks, reminders, and general conversation. What would you like to know about my capabilities?",
                "I'm here to help you, {user_name}! I can assist with schedule management, task organization, reminders, and friendly conversation. How can I help you today?"
            ],
            
            'general_chat': [
                "I hope you're having a great day, {user_name}! I'm here to help make it even better. What's on your mind?",
                "That's interesting, {user_name}! I'm here to help you with your schedule, answer questions, or just chat. What would you like to focus on today?",
                "I'm glad you're here, {user_name}! How can I help you today? I can assist with your schedule, tasks, or just have a friendly conversation."
            ],
            
            'time_query': [
                "I can help you with time-related questions, {user_name}! You have a Sales Meeting at 10:00 and gym at 17:00. Is there a specific time you're asking about?",
                "Looking at your schedule, {user_name}, you have appointments at 10:00 and 17:00. What time information do you need?",
                "I can help you with timing, {user_name}! Your schedule shows 10:00 for the Sales Meeting and 17:00 for gym. What would you like to know?"
            ]
        }
    
    def parse_intent(self, user_input):
        """Parse user input and return intent with confidence score"""
        user_input_lower = user_input.lower()
        
        best_intent = 'general_chat'
        best_confidence = 0.0
        matched_patterns = []
        
        # Check each intent pattern
        for intent_name, intent_data in self.intent_patterns.items():
            confidence = 0.0
            
            # Check regex patterns
            for pattern in intent_data['patterns']:
                if re.search(pattern, user_input_lower, re.IGNORECASE):
                    confidence += 0.4
                    matched_patterns.append(pattern)
            
            # Check keywords
            keyword_matches = 0
            for keyword in intent_data['keywords']:
                if keyword.lower() in user_input_lower:
                    keyword_matches += 1
            
            # Calculate keyword confidence
            if keyword_matches > 0:
                confidence += min(0.3, keyword_matches * 0.1)
            
            # Check for exact matches
            if any(keyword.lower() == user_input_lower.strip() for keyword in intent_data['keywords']):
                confidence += 0.2
            
            # Update best intent if confidence is higher
            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent_name
        
        # Ensure minimum confidence for general_chat
        if best_confidence < 0.1:
            best_intent = 'general_chat'
            best_confidence = 0.3
        
        return {
            'intent': best_intent,
            'confidence': min(1.0, best_confidence),
            'slots': self.extract_slots(user_input_lower),
            'raw_input': user_input,
            'matched_patterns': matched_patterns
        }
    
    def extract_slots(self, user_input):
        """Extract slots/entities from user input"""
        slots = []
        
        # Extract time slots
        time_patterns = [
            r'\b(10:00|10 am|10:00 am)\b',
            r'\b(17:00|5 pm|5:00 pm)\b',
            r'\b(morning|afternoon|evening)\b',
            r'\b(today|tomorrow|this week)\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            for match in matches:
                slots.append({'entity': 'time', 'value': match})
        
        # Extract person slots
        person_patterns = [
            r'\b(alex|taipy|sophie)\b'
        ]
        
        for pattern in person_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            for match in matches:
                slots.append({'entity': 'person', 'value': match})
        
        # Extract activity slots
        activity_patterns = [
            r'\b(meeting|gym|workout|sales|appointment|task|reminder)\b'
        ]
        
        for pattern in activity_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            for match in matches:
                slots.append({'entity': 'activity', 'value': match})
        
        return slots
    
    def get_response(self, intent_data, user_name="Alex"):
        """Generate response based on parsed intent"""
        intent = intent_data.get('intent', 'general_chat')
        confidence = intent_data.get('confidence', 0.0)
        
        # Get response templates for the intent
        templates = self.response_templates.get(intent, self.response_templates['general_chat'])
        
        # Select a random template
        selected_template = random.choice(templates)
        
        # Format the response with user name
        response = selected_template.format(user_name=user_name)
        
        return {
            'response': response,
            'intent': intent,
            'confidence': confidence
        }
    
    def train_on_example(self, user_input, expected_intent):
        """Train the model on a new example (for future enhancement)"""
        # This could be used to improve the model over time
        print(f"Training example: '{user_input}' -> {expected_intent}")
        return True 