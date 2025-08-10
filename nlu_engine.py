import json
import re
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

class VoiceAssistantNLU:
    def __init__(self):
        self.initialize_patterns()
        self.conversation_context = {
            'last_intent': None,
            'conversation_flow': [],
            'user_mood': 'neutral',
            'interaction_count': 0,
            'preferred_topics': []
        }
        self.response_memory = {}
        print("✅ Enhanced Custom NLU engine initialized successfully")
    
    def initialize_patterns(self):
        """Initialize enhanced intent patterns with regex and keywords"""
        
        # Define enhanced intent patterns with regex and keywords
        self.intent_patterns = {
            'greeting': {
                'patterns': [
                    r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                    r'\b(how are you|what\'s up|how\'s it going)\b',
                    r'\b(nice to see you|pleasure to meet you)\b'
                ],
                'keywords': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you', "what's up", 'going', 'nice', 'pleasure'],
                'priority': 1.0
            },
            
            'schedule_query': {
                'patterns': [
                    r'\b(what\'s on my schedule|what\'s my schedule|what do I have today)\b',
                    r'\b(what meetings|what appointments|what\'s planned)\b',
                    r'\b(show me my schedule|tell me about my schedule)\b',
                    r'\b(what\'s happening today|what\'s my day look like)\b'
                ],
                'keywords': ['schedule', 'meeting', 'appointment', 'planned', 'today', 'have', 'happening', 'day'],
                'priority': 1.2
            },
            
            'meeting_query': {
                'patterns': [
                    r'\b(sales meeting|meeting with taipy|meeting at 10)\b',
                    r'\b(when is my sales meeting|what time is the meeting)\b',
                    r'\b(taipy meeting|sales call|business meeting)\b'
                ],
                'keywords': ['sales', 'meeting', 'taipy', '10:00', '10 am', 'call', 'business'],
                'priority': 1.3
            },
            
            'gym_query': {
                'patterns': [
                    r'\b(gym session|gym with sophie|workout)\b',
                    r'\b(when is my gym|what time is gym)\b',
                    r'\b(exercise|fitness|training session)\b'
                ],
                'keywords': ['gym', 'workout', 'sophie', '17:00', '5 pm', 'exercise', 'fitness', 'training'],
                'priority': 1.3
            },
            
            'task_help': {
                'patterns': [
                    r'\b(help me with tasks|need help with tasks|help me organize)\b',
                    r'\b(help me manage|task assistance|to-do list)\b',
                    r'\b(what should I do|prioritize my tasks|organize my day)\b'
                ],
                'keywords': ['help', 'tasks', 'organize', 'manage', 'assistance', 'to-do', 'prioritize', 'what should'],
                'priority': 1.1
            },
            
            'reminder_request': {
                'patterns': [
                    r'\b(remind me|set a reminder|need a reminder)\b',
                    r'\b(can you remind me|set up a reminder)\b',
                    r'\b(don\'t let me forget|remember to tell me)\b'
                ],
                'keywords': ['remind', 'reminder', 'set up', 'remember', 'forget', 'tell me'],
                'priority': 1.1
            },
            
            'assistant_info': {
                'patterns': [
                    r'\b(who are you|what can you do|what\'s your name)\b',
                    r'\b(tell me about yourself|what are your capabilities)\b',
                    r'\b(what do you do|how do you work)\b'
                ],
                'keywords': ['who are you', 'what can you do', 'your name', 'capabilities', 'yourself', 'what do you do', 'how do you work'],
                'priority': 0.9
            },
            
            'general_chat': {
                'patterns': [
                    r'\b(how\'s your day|how are you doing|what\'s the weather)\b',
                    r'\b(nice day|good day|how\'s it going)\b',
                    r'\b(what\'s new|anything interesting|casual chat)\b'
                ],
                'keywords': ['weather', 'day', 'nice', 'good', 'going', 'new', 'interesting', 'casual', 'chat'],
                'priority': 0.8
            },
            
            'time_query': {
                'patterns': [
                    r'\b(what time is it|what\'s the time|current time)\b',
                    r'\b(time now|what time do I have|next appointment)\b',
                    r'\b(when is|what time|time check)\b'
                ],
                'keywords': ['time', 'when', 'current', 'now', 'appointment', 'next', 'check'],
                'priority': 1.0
            },
            
            'entertainment': {
                'patterns': [
                    r'\b(tell me a joke|make me laugh|entertain me)\b',
                    r'\b(something funny|humor|amuse me)\b',
                    r'\b(story|anecdote|fun fact)\b'
                ],
                'keywords': ['joke', 'laugh', 'funny', 'entertain', 'humor', 'amuse', 'story', 'anecdote', 'fun fact'],
                'priority': 0.9
            },
            
            'emotional_support': {
                'patterns': [
                    r'\b(i\'m feeling|i feel|i am)\s+(sad|happy|angry|worried|stressed|excited)\b',
                    r'\b(not feeling well|down|upset|anxious)\b',
                    r'\b(need support|help me feel|cheer me up)\b'
                ],
                'keywords': ['feeling', 'feel', 'sad', 'happy', 'angry', 'worried', 'stressed', 'excited', 'support', 'cheer', 'anxious'],
                'priority': 1.4
            },
            
            'gratitude': {
                'patterns': [
                    r'\b(thank you|thanks|appreciate it|grateful)\b',
                    r'\b(that\'s helpful|good job|well done)\b'
                ],
                'keywords': ['thank', 'thanks', 'appreciate', 'grateful', 'helpful', 'good job', 'well done'],
                'priority': 1.0
            },
            
            'farewell': {
                'patterns': [
                    r'\b(goodbye|bye|see you|farewell|later)\b',
                    r'\b(take care|have a good day|talk to you later)\b'
                ],
                'keywords': ['goodbye', 'bye', 'see you', 'farewell', 'later', 'take care', 'good day', 'talk to you'],
                'priority': 1.0
            }
        }
        
        # Enhanced response templates with context awareness
        self.response_templates = {
            'greeting': [
                "Hello! How can I help you today? I'm here to assist with your schedule, tasks, or just chat!",
                "Hi there! Great to see you. What can I help you with today? I'm ready to assist with anything you need!",
                "Hey! I'm here and ready to help. What's on your mind today?",
                "Good to see you! How can I make your day better today?"
            ],
            
            'schedule_query': [
                "Looking at your schedule today, you have a Sales Meeting with Taipy at 10:00 and a gym session with Sophie at 17:00. How can I help you manage your time effectively?",
                "Your schedule includes a Sales Meeting with Taipy at 10:00 and gym with Sophie at 17:00. Would you like me to help you prepare for any of these or set up reminders?",
                "Today you're busy with a Sales Meeting at 10:00 and gym at 17:00. I can help you organize your day or prepare for these activities. What would be most helpful?"
            ],
            
            'meeting_query': [
                "Your Sales Meeting with Taipy is scheduled for 10:00 today. That's coming up soon! Would you like me to help you prepare for it or set a reminder?",
                "The Sales Meeting with Taipy is at 10:00. I can help you get ready for it or make sure you don't forget. What would you like me to do?",
                "You have your Sales Meeting with Taipy at 10:00. I'm here to help make sure you're prepared and on time!"
            ],
            
            'gym_query': [
                "Your gym session with Sophie is scheduled for 17:00 today. That's a great way to stay active! Would you like me to remind you about it or help you plan around it?",
                "You're hitting the gym with Sophie at 17:00. I can set a reminder or help you prepare for your workout. What would be most helpful?",
                "Your gym session with Sophie is at 17:00. I'm here to help you stay on track with your fitness goals!"
            ],
            
            'task_help': [
                "I'd be happy to help you with your tasks! Let me know what you need help organizing or prioritizing today.",
                "Task management is one of my specialties! What specific tasks would you like help with today?",
                "I'm here to help you stay organized and productive. What tasks are you working on that I can assist with?"
            ],
            
            'reminder_request': [
                "I'd be happy to help you set reminders! What would you like me to remind you about and when?",
                "Setting reminders is a great way to stay on top of things. What do you need to remember?",
                "I can definitely help with reminders. Just let me know what you need to remember and when!"
            ],
            
            'assistant_info': [
                "I'm your AI voice assistant, designed to help you manage your schedule, organize tasks, and make your day more productive. I can assist with time management, reminders, general conversation, and much more!",
                "I'm here to be your helpful companion throughout the day. I specialize in schedule management, task organization, and providing support when you need it most.",
                "I'm your personal AI assistant, focused on helping you stay organized and productive. I'm here to chat, help with tasks, and make your day run smoothly!"
            ],
            
            'general_chat': [
                "I'm doing great, thanks for asking! I'm here and ready to help you with whatever you need today.",
                "I'm having a wonderful day! I'm excited to help you stay organized and productive.",
                "I'm doing well and ready to assist! What would you like to chat about or work on today?"
            ],
            
            'time_query': [
                "I can help you check the time and manage your schedule. What specific time information do you need?",
                "I'm here to help with time management and scheduling. What would you like to know about your time today?",
                "I can assist with time-related questions and help you stay on schedule. What do you need to know?"
            ],
            
            'entertainment': [
                "I'd love to entertain you! Here's a joke: Why don't scientists trust atoms? Because they make up everything! 😄 What else can I help you with today?",
                "Here's something to make you smile: What do you call a fake noodle? An impasta! 😄 Now, how can I assist you with your day?",
                "Let me share a fun fact: Honey never spoils! Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible. Pretty amazing, right?"
            ],
            
            'emotional_support': [
                "I'm here to listen and support you. It's okay to feel that way, and I want you to know you're not alone. How can I help you feel better today?",
                "I understand that emotions can be challenging. I'm here to support you and help you through this. What would be most helpful right now?",
                "I care about how you're feeling. Let's work together to find ways to help you feel better. What can I do to support you today?"
            ],
            
            'gratitude': [
                "You're very welcome! I'm here to help make your day better. Is there anything else I can assist you with?",
                "It's my pleasure to help! I'm always ready to support you. What else can I do for you today?",
                "Thank you for the kind words! I'm here whenever you need assistance. What's next on your agenda?"
            ],
            
            'farewell': [
                "Goodbye! Have a wonderful day ahead. I'll be here when you need me again!",
                "Take care! I'm always here to help when you need me. Have a great day!",
                "See you later! Don't hesitate to reach out if you need anything. Take care!"
            ]
        }
    
    def parse_intent(self, user_input: str) -> Dict[str, Any]:
        """Enhanced intent parsing with confidence scoring and context awareness"""
        user_input_lower = user_input.lower()
        
        # Track interaction count
        self.conversation_context['interaction_count'] += 1
        
        # Initialize scoring
        intent_scores = {}
        max_confidence = 0
        detected_intent = "general_chat"
        
        # Score each intent based on patterns and keywords
        for intent, config in self.intent_patterns.items():
            score = 0
            
            # Pattern matching (higher weight)
            for pattern in config['patterns']:
                if re.search(pattern, user_input_lower, re.IGNORECASE):
                    score += 0.6
                    break
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in config['keywords'] if keyword in user_input_lower)
            if keyword_matches > 0:
                score += (keyword_matches / len(config['keywords'])) * 0.4
            
            # Apply priority weight
            score *= config['priority']
            
            # Context bonus for conversation flow
            if self.conversation_context['last_intent'] == intent:
                score += 0.1  # Slight bonus for conversation continuity
            
            intent_scores[intent] = score
            
            if score > max_confidence:
                max_confidence = score
                detected_intent = intent
        
        # Update conversation context
        self.conversation_context['last_intent'] = detected_intent
        self.conversation_context['conversation_flow'].append({
            'intent': detected_intent,
            'confidence': max_confidence,
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input
        })
        
        # Keep only recent conversation flow
        if len(self.conversation_context['conversation_flow']) > 10:
            self.conversation_context['conversation_flow'] = self.conversation_context['conversation_flow'][-10:]
        
        # Extract additional context
        context_info = self._extract_context_info(user_input_lower)
        
        return {
            "intent": detected_intent,
            "confidence": min(0.95, max(0.6, max_confidence)),  # Minimum 0.6 confidence
            "context": context_info,
            "conversation_flow": self.conversation_context['conversation_flow'][-3:],
            "interaction_count": self.conversation_context['interaction_count']
        }
    
    def _extract_context_info(self, user_input: str) -> Dict[str, Any]:
        """Extract additional context information from user input"""
        context = {
            'urgency': False,
            'sentiment': 'neutral',
            'has_question': False,
            'word_count': len(user_input.split()),
            'time_mentions': [],
            'emotion_indicators': []
        }
        
        # Check for urgency
        urgency_words = ['urgent', 'asap', 'immediately', 'now', 'quick', 'fast', 'hurry']
        context['urgency'] = any(word in user_input for word in urgency_words)
        
        # Check for questions
        context['has_question'] = '?' in user_input or any(word in user_input for word in ['what', 'how', 'why', 'when', 'where', 'who'])
        
        # Extract time mentions
        time_patterns = [
            r'\b\d{1,2}:\d{2}\b',  # HH:MM format
            r'\b\d{1,2}\s*(am|pm)\b',  # 10 am, 5 pm
            r'\b(morning|afternoon|evening|night)\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            context['time_mentions'].extend(matches)
        
        # Basic sentiment analysis
        positive_words = ['good', 'great', 'awesome', 'excellent', 'wonderful', 'amazing', 'happy', 'excited']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointing', 'sad', 'angry', 'worried']
        
        positive_count = sum(1 for word in positive_words if word in user_input)
        negative_count = sum(1 for word in negative_words if word in user_input)
        
        if positive_count > negative_count:
            context['sentiment'] = 'positive'
        elif negative_count > positive_count:
            context['sentiment'] = 'negative'
        
        # Extract emotion indicators
        emotion_keywords = {
            'joy': ['happy', 'excited', 'thrilled', 'delighted', 'joyful'],
            'sadness': ['sad', 'depressed', 'melancholy', 'sorrowful', 'down'],
            'anger': ['angry', 'furious', 'irritated', 'annoyed', 'mad'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned']
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                context['emotion_indicators'].append(emotion)
        
        return context
    
    def extract_slots(self, user_input: str) -> Dict[str, Any]:
        """Extract slots and entities from user input"""
        slots = {}
        user_input_lower = user_input.lower()
        
        # Extract time slots
        time_patterns = {
            'time': r'\b\d{1,2}:\d{2}\b',
            'am_pm': r'\b\d{1,2}\s*(am|pm)\b',
            'period': r'\b(morning|afternoon|evening|night)\b'
        }
        
        for slot_name, pattern in time_patterns.items():
            matches = re.findall(pattern, user_input_lower, re.IGNORECASE)
            if matches:
                slots[slot_name] = matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])
        
        # Extract task-related slots
        if 'remind' in user_input_lower or 'reminder' in user_input_lower:
            # Try to extract what to remind about
            reminder_patterns = [
                r'remind me to\s+(.+)',
                r'remind me about\s+(.+)',
                r'remind me\s+(.+)',
                r'reminder for\s+(.+)'
            ]
            
            for pattern in reminder_patterns:
                match = re.search(pattern, user_input_lower)
                if match:
                    slots['reminder_task'] = match.group(1).strip()
                    break
        
        # Extract schedule-related slots
        if any(word in user_input_lower for word in ['meeting', 'appointment', 'gym']):
            if 'taipy' in user_input_lower or 'sales' in user_input_lower:
                slots['meeting_type'] = 'sales_meeting'
                slots['meeting_person'] = 'Taipy'
            elif 'sophie' in user_input_lower or 'gym' in user_input_lower:
                slots['activity_type'] = 'gym_session'
                slots['activity_person'] = 'Sophie'
        
        return slots
    
    def get_response(self, intent_data: Dict[str, Any], user_name: str = "Alex") -> Dict[str, Any]:
        """Generate enhanced contextual response based on intent and context"""
        intent = intent_data.get('intent', 'general_chat')
        confidence = intent_data.get('confidence', 0.6)
        context = intent_data.get('context', {})
        
        # Get base response template
        if intent in self.response_templates:
            base_response = random.choice(self.response_templates[intent])
        else:
            base_response = "I'm here to help you with whatever you need today. What can I assist you with?"
        
        # Personalize response with user name if available
        if user_name and user_name != "Alex":
            base_response = base_response.replace("Alex", user_name)
        
        # Enhance response based on context
        enhanced_response = self._enhance_response_with_context(base_response, intent_data, context)
        
        # Generate contextual suggestions
        suggestions = self._generate_contextual_suggestions(intent, context)
        
        # Update conversation memory
        self._update_response_memory(intent, enhanced_response, confidence)
        
        return {
            "response": enhanced_response,
            "intent": intent,
            "confidence": confidence,
            "context": context,
            "suggestions": suggestions,
            "conversation_context": self.conversation_context
        }
    
    def _enhance_response_with_context(self, base_response: str, intent_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Enhance response with contextual information"""
        enhanced = base_response
        
        # Add urgency context
        if context.get('urgency'):
            enhanced += " I can see this is urgent, so let me help you quickly."
        
        # Add sentiment-based enhancement
        if context.get('sentiment') == 'negative':
            enhanced += " I'm here to help you feel better and get things sorted out."
        elif context.get('sentiment') == 'positive':
            enhanced += " I'm glad you're in good spirits! Let's keep that positive energy going."
        
        # Add time context if relevant
        if context.get('time_mentions'):
            enhanced += f" I noticed you mentioned {', '.join(context['time_mentions'])} - I can help you manage your time effectively."
        
        # Add conversation flow context
        if len(self.conversation_context['conversation_flow']) > 1:
            last_intent = self.conversation_context['conversation_flow'][-2]['intent']
            if last_intent == intent_data['intent']:
                enhanced += " I see we're still on this topic - let me know if you need more specific help."
        
        return enhanced
    
    def _generate_contextual_suggestions(self, intent: str, context: Dict[str, Any]) -> List[str]:
        """Generate contextual suggestions based on intent and context"""
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
        
        # Add context-specific suggestions
        if context.get('urgency'):
            suggestions.insert(0, "Let me help you quickly with this urgent matter")
        
        if context.get('has_question'):
            suggestions.insert(0, "I can answer more questions about this topic")
        
        return suggestions[:4]  # Return top 4 suggestions
    
    def _update_response_memory(self, intent: str, response: str, confidence: float):
        """Update response memory for learning and improvement"""
        if intent not in self.response_memory:
            self.response_memory[intent] = []
        
        self.response_memory[intent].append({
            'response': response,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'success': confidence > 0.7  # Consider high confidence responses as successful
        })
        
        # Keep only recent responses for each intent
        if len(self.response_memory[intent]) > 5:
            self.response_memory[intent] = self.response_memory[intent][-5:]
    
    def train_on_example(self, user_input: str, expected_intent: str) -> bool:
        """Train the NLU model on new examples"""
        try:
            # Add new pattern to existing intent
            if expected_intent in self.intent_patterns:
                # Add user input as a new keyword if it's not too long
                if len(user_input.split()) <= 5:
                    self.intent_patterns[expected_intent]['keywords'].append(user_input.lower())
                
                # Create a simple pattern from the input
                simple_pattern = r'\b' + re.escape(user_input.lower()) + r'\b'
                if simple_pattern not in self.intent_patterns[expected_intent]['patterns']:
                    self.intent_patterns[expected_intent]['patterns'].append(simple_pattern)
                
                print(f"✅ Trained on: '{user_input}' -> {expected_intent}")
                return True
            else:
                print(f"❌ Unknown intent: {expected_intent}")
                return False
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def get_conversation_insights(self) -> Dict[str, Any]:
        """Get insights about conversation patterns and performance"""
        return {
            'total_interactions': self.conversation_context['interaction_count'],
            'conversation_flow': self.conversation_context['conversation_flow'][-5:],
            'last_intent': self.conversation_context['last_intent'],
            'user_mood': self.conversation_context['user_mood'],
            'preferred_topics': self.conversation_context['preferred_topics'],
            'response_memory': {intent: len(responses) for intent, responses in self.response_memory.items()}
        }
    
    def reset_conversation_context(self):
        """Reset conversation context and memory"""
        self.conversation_context = {
            'last_intent': None,
            'conversation_flow': [],
            'user_mood': 'neutral',
            'interaction_count': 0,
            'preferred_topics': []
        }
        self.response_memory = {}
        print("✅ Conversation context and memory reset") 