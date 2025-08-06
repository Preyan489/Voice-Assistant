import json
import re
import random
import numpy as np
from datetime import datetime
import pickle
import os
from collections import defaultdict
import math

# Global variable for ML availability
ML_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ ML libraries not available, using enhanced rule-based approach")
    ML_AVAILABLE = False

class AdvancedVoiceAssistantNLU:
    def __init__(self):
        self.initialize_patterns()
        self.initialize_ml_components()
        self.load_or_train_model()
        print("✅ Advanced NLU engine initialized successfully")
    
    def initialize_patterns(self):
        """Initialize intent patterns and training data"""
        
        # Enhanced intent patterns with more variations
        self.intent_patterns = {
            'greeting': {
                'patterns': [
                    r'\b(hello|hi|hey|good morning|good afternoon|good evening|greetings)\b',
                    r'\b(how are you|what\'s up|how\'s it going|how do you do)\b',
                    r'\b(nice to meet you|pleasure to meet you|good to see you)\b'
                ],
                'keywords': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings', 'how are you', "what's up", "how's it going"]
            },
            
            'schedule_query': {
                'patterns': [
                    r'\b(what\'s on my schedule|what\'s my schedule|what do I have today)\b',
                    r'\b(what meetings|what appointments|what\'s planned|what\'s coming up)\b',
                    r'\b(show me my schedule|tell me about my schedule|my agenda)\b',
                    r'\b(what\'s happening today|what\'s on the agenda|my day)\b'
                ],
                'keywords': ['schedule', 'meeting', 'appointment', 'planned', 'today', 'have', 'agenda', 'happening', 'coming up']
            },
            
            'meeting_query': {
                'patterns': [
                    r'\b(sales meeting|meeting with taipy|meeting at 10)\b',
                    r'\b(when is my sales meeting|what time is the meeting)\b',
                    r'\b(meeting details|sales meeting info|taipy meeting)\b'
                ],
                'keywords': ['sales', 'meeting', 'taipy', '10:00', '10 am', 'meeting details']
            },
            
            'gym_query': {
                'patterns': [
                    r'\b(gym session|gym with sophie|workout)\b',
                    r'\b(when is my gym|what time is gym|exercise)\b',
                    r'\b(workout session|fitness|training)\b'
                ],
                'keywords': ['gym', 'workout', 'sophie', '17:00', '5 pm', 'exercise', 'fitness', 'training']
            },
            
            'task_help': {
                'patterns': [
                    r'\b(help me with tasks|need help with tasks|help me organize)\b',
                    r'\b(help me manage|task assistance|to-do list)\b',
                    r'\b(organize my day|manage tasks|get organized)\b'
                ],
                'keywords': ['help', 'tasks', 'organize', 'manage', 'assistance', 'to-do', 'organized']
            },
            
            'reminder_request': {
                'patterns': [
                    r'\b(remind me|set a reminder|need a reminder)\b',
                    r'\b(can you remind me|set up a reminder|remember)\b',
                    r'\b(reminder for|alert me|notify me)\b'
                ],
                'keywords': ['remind', 'reminder', 'set up', 'remember', 'alert', 'notify']
            },
            
            'assistant_info': {
                'patterns': [
                    r'\b(who are you|what can you do|what\'s your name)\b',
                    r'\b(tell me about yourself|what are your capabilities)\b',
                    r'\b(your abilities|your features|what you do)\b'
                ],
                'keywords': ['who are you', 'what can you do', 'your name', 'capabilities', 'yourself', 'abilities', 'features']
            },
            
            'general_chat': {
                'patterns': [
                    r'\b(how\'s your day|how are you doing|what\'s the weather)\b',
                    r'\b(nice day|good day|how\'s it going)\b',
                    r'\b(chat|conversation|talk)\b'
                ],
                'keywords': ['weather', 'day', 'nice', 'good', 'going', 'chat', 'conversation', 'talk']
            },
            
            'time_query': {
                'patterns': [
                    r'\b(what time is it|what\'s the time|current time)\b',
                    r'\b(time now|what time do I have|next appointment)\b',
                    r'\b(when|time|clock)\b',
                    r'\b(what day is it|what day is it today|what day is today)\b',
                    r'\b(what\'s today|what\'s the date|what date is it)\b',
                    r'\b(today\'s date|what day of the week|what day are we)\b'
                ],
                'keywords': ['time', 'when', 'current', 'now', 'appointment', 'clock', 'day', 'date', 'today', 'week']
            }
        }
        
        # Training data for machine learning
        self.training_data = {
            'greeting': [
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "how are you", "what's up", "how's it going", "greetings", "good day",
                "nice to meet you", "pleasure to meet you", "good to see you",
                "hello there", "hi there", "hey there", "good morning to you",
                "how are you doing", "what's new", "how's your day"
            ],
            'schedule_query': [
                "what's on my schedule", "what's my schedule today", "what do I have today",
                "what meetings do I have", "what appointments do I have", "what's planned for today",
                "show me my schedule", "tell me about my schedule", "my agenda",
                "what's happening today", "what's on the agenda", "my day",
                "what's coming up", "what's next", "my calendar", "today's schedule"
            ],
            'meeting_query': [
                "when is my sales meeting", "what time is the meeting with Taipy",
                "do I have a meeting at 10:00", "sales meeting details", "meeting with Taipy",
                "what's the sales meeting about", "sales meeting time", "taipy meeting",
                "meeting at 10", "sales meeting info"
            ],
            'gym_query': [
                "when is my gym session", "what time is gym with Sophie", "do I have gym at 17:00",
                "gym session details", "workout with Sophie", "what's my gym schedule",
                "gym time", "workout session", "fitness training", "exercise time"
            ],
            'task_help': [
                "help me with tasks", "I need help with tasks", "can you help me organize",
                "help me manage my tasks", "I need task assistance", "help me with my to-do list",
                "organize my day", "manage tasks", "get organized", "task management help"
            ],
            'reminder_request': [
                "remind me", "set a reminder", "I need a reminder", "can you remind me",
                "set up a reminder", "I want to be reminded", "reminder for", "alert me",
                "notify me", "remember this"
            ],
            'assistant_info': [
                "who are you", "what can you do", "what's your name", "tell me about yourself",
                "what are your capabilities", "your abilities", "your features", "what you do",
                "what are you", "your name"
            ],
            'general_chat': [
                "how's your day", "how are you doing", "what's the weather like", "nice day",
                "good day", "how's it going", "chat with me", "conversation", "talk to me",
                "how's everything", "what's new"
            ],
            'time_query': [
                "what time is it", "what's the time", "current time", "time now",
                "what time do I have", "next appointment", "when", "time", "clock",
                "what time", "current time", "what day is it", "what day is it today",
                "what day is today", "what's today", "what's the date", "what date is it",
                "what date is today", "today's date", "what day of the week is it",
                "what day of the week is today", "is it monday", "is it tuesday",
                "what day are we", "what's the day today"
            ]
        }
    
    def initialize_ml_components(self):
        """Initialize machine learning components"""
        global ML_AVAILABLE
        if ML_AVAILABLE:
            try:
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.9
                )
                
                self.classifier = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10
                )
                
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
                
            except Exception as e:
                print(f"⚠️ Error initializing ML components: {e}")
                ML_AVAILABLE = False
    
    def preprocess_text(self, text):
        """Preprocess text for machine learning"""
        global ML_AVAILABLE
        if not ML_AVAILABLE:
            return text.lower()
        
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token.isalnum() and token not in self.stop_words]
            
            return ' '.join(tokens)
        except:
            return text.lower()
    
    def load_or_train_model(self):
        """Load existing model or train a new one"""
        global ML_AVAILABLE
        self.model_file = 'nlu_model.pkl'
        
        if ML_AVAILABLE and os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data['vectorizer']
                    self.classifier = model_data['classifier']
                    self.intent_labels = model_data['intent_labels']
                print("✅ Loaded existing ML model")
                return
            except Exception as e:
                print(f"⚠️ Error loading model: {e}")
        
        # Train new model
        if ML_AVAILABLE:
            self.train_model()
        else:
            print("✅ Using enhanced rule-based approach")
    
    def train_model(self):
        """Train the machine learning model"""
        global ML_AVAILABLE
        if not ML_AVAILABLE:
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for intent, examples in self.training_data.items():
                for example in examples:
                    X.append(self.preprocess_text(example))
                    y.append(intent)
            
            # Vectorize the text
            X_vectorized = self.vectorizer.fit_transform(X)
            
            # Train the classifier
            self.classifier.fit(X_vectorized, y)
            
            # Save the model
            model_data = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'intent_labels': list(set(y))
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            print("✅ ML model trained and saved successfully")
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
    
    def parse_intent_ml(self, user_input):
        """Parse intent using machine learning"""
        global ML_AVAILABLE
        if not ML_AVAILABLE:
            return None
        
        try:
            # Preprocess the input
            processed_input = self.preprocess_text(user_input)
            
            # Vectorize
            input_vectorized = self.vectorizer.transform([processed_input])
            
            # Predict
            intent = self.classifier.predict(input_vectorized)[0]
            confidence = max(self.classifier.predict_proba(input_vectorized)[0])
            
            return {
                'intent': intent,
                'confidence': confidence,
                'method': 'ml'
            }
        except Exception as e:
            print(f"Error in ML parsing: {e}")
            return None
    
    def parse_intent_rules(self, user_input):
        """Parse intent using enhanced rule-based approach"""
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
            
            # Check keywords with weighted scoring
            keyword_matches = 0
            total_keywords = len(intent_data['keywords'])
            
            for keyword in intent_data['keywords']:
                if keyword.lower() in user_input_lower:
                    keyword_matches += 1
            
            # Calculate keyword confidence with better weighting
            if keyword_matches > 0:
                keyword_confidence = (keyword_matches / total_keywords) * 0.4
                confidence += keyword_confidence
            
            # Check for exact matches
            if any(keyword.lower() == user_input_lower.strip() for keyword in intent_data['keywords']):
                confidence += 0.2
            
            # Semantic similarity bonus
            if self.calculate_semantic_similarity(user_input_lower, intent_data['keywords']):
                confidence += 0.1
            
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
            'method': 'rules'
        }
    
    def calculate_semantic_similarity(self, text, keywords):
        """Calculate semantic similarity between text and keywords"""
        # Simple word overlap similarity
        text_words = set(text.split())
        keyword_words = set()
        for keyword in keywords:
            keyword_words.update(keyword.split())
        
        if not keyword_words:
            return False
        
        overlap = len(text_words.intersection(keyword_words))
        similarity = overlap / len(keyword_words)
        return similarity > 0.3
    
    def parse_intent(self, user_input):
        """Parse user input using both ML and rule-based approaches"""
        # Try ML first
        ml_result = self.parse_intent_ml(user_input)
        
        # Try rule-based approach
        rule_result = self.parse_intent_rules(user_input)
        
        # Combine results for better accuracy
        if ml_result and ml_result['confidence'] > 0.7:
            # Use ML result if confidence is high
            final_intent = ml_result['intent']
            final_confidence = ml_result['confidence']
            method = 'ml'
        elif rule_result and rule_result['confidence'] > 0.6:
            # Use rule-based result if confidence is high
            final_intent = rule_result['intent']
            final_confidence = rule_result['confidence']
            method = 'rules'
        else:
            # Use the better of the two
            if ml_result and rule_result:
                if ml_result['confidence'] > rule_result['confidence']:
                    final_intent = ml_result['intent']
                    final_confidence = ml_result['confidence']
                    method = 'ml'
                else:
                    final_intent = rule_result['intent']
                    final_confidence = rule_result['confidence']
                    method = 'rules'
            elif ml_result:
                final_intent = ml_result['intent']
                final_confidence = ml_result['confidence']
                method = 'ml'
            else:
                final_intent = rule_result['intent']
                final_confidence = rule_result['confidence']
                method = 'rules'
        
        return {
            'intent': final_intent,
            'confidence': final_confidence,
            'slots': self.extract_slots(user_input.lower()),
            'raw_input': user_input,
            'method': method
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
        
        # Get current date for time queries
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        
        # Define response templates
        responses = {
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
                "I can help you with timing, {user_name}! Your schedule shows 10:00 for the Sales Meeting and 17:00 for gym. What would you like to know?",
                "Today is {current_date}, {user_name}! It's a great day to check your schedule - you have a Sales Meeting at 10:00 and gym at 17:00.",
                "It's {current_date} today, {user_name}! How can I help you with your schedule or time management?"
            ]
        }
        
        # Get response templates for the intent
        templates = responses.get(intent, responses['general_chat'])
        
        # Select a random template
        selected_template = random.choice(templates)
        
        # Format the response with user name and current date
        response = selected_template.format(user_name=user_name, current_date=current_date)
        
        return {
            'response': response,
            'intent': intent,
            'confidence': confidence
        }
    
    def train_on_example(self, user_input, expected_intent):
        """Train the model on a new example"""
        global ML_AVAILABLE
        if ML_AVAILABLE:
            try:
                # Add to training data
                if expected_intent not in self.training_data:
                    self.training_data[expected_intent] = []
                
                self.training_data[expected_intent].append(user_input)
                
                # Retrain the model
                self.train_model()
                print(f"✅ Trained on new example: '{user_input}' -> {expected_intent}")
                return True
            except Exception as e:
                print(f"❌ Error training on example: {e}")
                return False
        else:
            print(f"Training example: '{user_input}' -> {expected_intent}")
            return True 