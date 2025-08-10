import os
import openai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime, timedelta
import random

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

class LLMConversationHandler:
    def __init__(self):
        """Initialize the enhanced LLM conversation handler"""
        self.conversation_history = []
        self.user_preferences = {}
        self.topic_history = []
        self.emotion_tracker = {}
        self.conversation_stats = {
            'total_interactions': 0,
            'favorite_topics': [],
            'response_times': [],
            'user_satisfaction': []
        }
        self.system_prompt = self._get_enhanced_system_prompt()
        
    def _get_enhanced_system_prompt(self) -> str:
        """Get the enhanced system prompt for the assistant"""
        return """You are an advanced AI voice assistant with enhanced conversational capabilities. Your responses should be natural, helpful, and contextually aware.

Key Information:
- User's schedule: Sales Meeting with Taipy at 10:00; Gym with Sophie at 17:00
- Keep responses under 120 words for optimal voice synthesis
- Be friendly, empathetic, and conversational
- Help with schedule management, tasks, and general queries
- Engage in casual conversation, tell jokes, and provide entertainment
- Maintain context from the conversation history and adapt to user preferences
- Show personality while remaining professional and helpful

Enhanced Capabilities:
- Schedule management and optimization
- Task organization and prioritization
- General conversation and Q&A
- Entertainment (jokes, stories, casual chat)
- Emotional intelligence and empathy
- Context-aware suggestions
- Learning from user preferences
- Adaptive response styles

Response Guidelines:
- Be conversational and natural
- Reference previous conversation context when relevant
- Adapt tone based on user's emotional state
- Provide actionable suggestions
- Show understanding of user's schedule and preferences
- Use humor appropriately
- Be concise but engaging

Format responses naturally for speech synthesis while maintaining conversational flow."""
    
    def get_response(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get an enhanced response from the LLM with context awareness"""
        try:
            start_time = datetime.now()
            
            # Analyze user input for enhanced understanding
            input_analysis = self._analyze_user_input(user_input)
            
            # Add user message to history with metadata
            self.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
                "analysis": input_analysis,
                "context": context or {}
            })
            
            # Update conversation statistics
            self.conversation_stats['total_interactions'] += 1
            
            # Prepare enhanced context for the API
            enhanced_context = self._prepare_enhanced_context(user_input, input_analysis, context)
            
            # Prepare messages for the API with enhanced context
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": f"Current context: {json.dumps(enhanced_context)}"}
            ] + self.conversation_history[-6:]  # Keep last 6 messages for better context
            
            # Get completion from OpenAI with enhanced parameters
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=180,  # Slightly increased for better responses
                temperature=0.8,  # More creative responses
                presence_penalty=0.7,  # Encourage varied responses
                frequency_penalty=0.4,  # Reduce repetition
                top_p=0.9  # Better response quality
            )
            
            # Extract the response
            assistant_response = response.choices[0].message.content
            
            # Analyze response for enhanced intent and confidence
            intent_data = self._analyze_enhanced_response(user_input, assistant_response, input_analysis)
            
            # Generate contextual suggestions
            suggestions = self._generate_enhanced_suggestions(user_input, assistant_response, intent_data)
            
            # Update user preferences and topic tracking
            self._update_conversation_insights(user_input, assistant_response, intent_data)
            
            # Add assistant response to history with metadata
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now().isoformat(),
                "intent": intent_data["intent"],
                "confidence": intent_data["confidence"],
                "suggestions": suggestions
            })
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            self.conversation_stats['response_times'].append(response_time)
            
            return {
                "success": True,
                "response": assistant_response,
                "intent": intent_data["intent"],
                "confidence": intent_data["confidence"],
                "suggestions": suggestions,
                "conversation_history": self.conversation_history,
                "context": enhanced_context,
                "analysis": input_analysis,
                "response_time": response_time,
                "conversation_stats": self.conversation_stats
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I'm having trouble processing that request. Could you try rephrasing it?"
            }
    
    def _analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """Enhanced analysis of user input for better understanding"""
        user_input_lower = user_input.lower()
        
        # Enhanced intent detection with confidence scoring
        intents = {
            "greeting": {
                "keywords": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
                "patterns": [r'\b(hello|hi|hey)\b', r'\bgood\s+(morning|afternoon|evening)\b'],
                "weight": 1.0
            },
            "schedule_query": {
                "keywords": ["schedule", "meeting", "appointment", "gym", "workout", "plan"],
                "patterns": [r'\b(schedule|meeting|appointment)\b', r'\b(gym|workout|plan)\b'],
                "weight": 1.2
            },
            "task_management": {
                "keywords": ["task", "todo", "reminder", "organize", "help", "assist"],
                "patterns": [r'\b(task|todo|reminder)\b', r'\b(organize|help|assist)\b'],
                "weight": 1.1
            },
            "question": {
                "keywords": ["what", "how", "why", "when", "where", "who", "which"],
                "patterns": [r'\b(what|how|why|when|where|who|which)\b'],
                "weight": 1.0
            },
            "entertainment": {
                "keywords": ["joke", "funny", "laugh", "story", "entertain", "amuse"],
                "patterns": [r'\b(joke|funny|laugh)\b', r'\b(story|entertain|amuse)\b'],
                "weight": 0.9
            },
            "gratitude": {
                "keywords": ["thank", "thanks", "appreciate", "grateful"],
                "patterns": [r'\b(thank|thanks|appreciate|grateful)\b'],
                "weight": 1.0
            },
            "farewell": {
                "keywords": ["goodbye", "bye", "see you", "farewell", "later"],
                "patterns": [r'\b(goodbye|bye|see\s+you|farewell|later)\b'],
                "weight": 1.0
            },
            "emotional_support": {
                "keywords": ["sad", "happy", "angry", "worried", "stressed", "excited"],
                "patterns": [r'\b(sad|happy|angry|worried|stressed|excited)\b'],
                "weight": 1.3
            }
        }
        
        # Calculate intent scores with pattern matching
        intent_scores = {}
        for intent, config in intents.items():
            score = 0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in config["keywords"] if keyword in user_input_lower)
            if keyword_matches > 0:
                score += (keyword_matches / len(config["keywords"])) * 0.6
            
            # Pattern matching
            for pattern in config["patterns"]:
                if re.search(pattern, user_input_lower):
                    score += 0.4
            
            # Apply weight
            score *= config["weight"]
            intent_scores[intent] = score
        
        # Get top intent
        top_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # Sentiment analysis
        positive_words = ["good", "great", "awesome", "excellent", "wonderful", "amazing"]
        negative_words = ["bad", "terrible", "awful", "horrible", "worst", "disappointing"]
        
        positive_count = sum(1 for word in positive_words if word in user_input_lower)
        negative_count = sum(1 for word in negative_words if word in user_input_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Urgency detection
        urgency_indicators = ["urgent", "asap", "immediately", "now", "quick", "fast"]
        urgency = any(indicator in user_input_lower for indicator in urgency_indicators)
        
        return {
            "detected_intent": top_intent[0],
            "confidence": min(0.95, top_intent[1]),
            "sentiment": sentiment,
            "urgency": urgency,
            "word_count": len(user_input.split()),
            "has_question": "?" in user_input,
            "emotion_indicators": self._extract_emotion_indicators(user_input_lower)
        }
    
    def _extract_emotion_indicators(self, text: str) -> List[str]:
        """Extract emotional indicators from text"""
        emotion_keywords = {
            "joy": ["happy", "excited", "thrilled", "delighted", "joyful"],
            "sadness": ["sad", "depressed", "melancholy", "sorrowful", "down"],
            "anger": ["angry", "furious", "irritated", "annoyed", "mad"],
            "fear": ["scared", "afraid", "terrified", "worried", "anxious"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned"],
            "disgust": ["disgusted", "revolted", "appalled", "sickened"],
            "trust": ["trust", "confident", "sure", "certain", "reliable"]
        }
        
        detected_emotions = []
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected_emotions.append(emotion)
        
        return detected_emotions
    
    def _prepare_enhanced_context(self, user_input: str, analysis: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare enhanced context for the LLM"""
        enhanced_context = {
            "current_time": datetime.now().strftime("%H:%M"),
            "user_schedule": {
                "sales_meeting": "10:00 with Taipy",
                "gym_session": "17:00 with Sophie"
            },
            "conversation_context": {
                "total_interactions": self.conversation_stats['total_interactions'],
                "recent_topics": self.topic_history[-3:] if self.topic_history else [],
                "user_preferences": self.user_preferences,
                "conversation_length": len(self.conversation_history)
            },
            "user_analysis": analysis,
            "previous_context": context or {},
            "assistant_personality": {
                "tone": "friendly and professional",
                "style": "conversational and helpful",
                "expertise": ["schedule management", "task organization", "general assistance", "entertainment"]
            }
        }
        
        return enhanced_context
    
    def _analyze_enhanced_response(self, user_input: str, assistant_response: str, input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced response analysis with better intent detection"""
        # Use the input analysis as base
        intent = input_analysis["detected_intent"]
        confidence = input_analysis["confidence"]
        
        # Adjust confidence based on response quality
        response_quality = self._assess_response_quality(assistant_response)
        confidence = min(0.95, confidence * response_quality)
        
        # Detect if response matches expected intent
        if intent == "schedule_query" and any(word in assistant_response.lower() for word in ["schedule", "meeting", "gym"]):
            confidence = min(0.95, confidence + 0.1)
        
        return {
            "intent": intent,
            "confidence": confidence,
            "response_quality": response_quality,
            "context_relevance": self._assess_context_relevance(user_input, assistant_response)
        }
    
    def _assess_response_quality(self, response: str) -> float:
        """Assess the quality of the generated response"""
        quality_score = 1.0
        
        # Check response length (optimal for voice: 50-120 words)
        word_count = len(response.split())
        if 50 <= word_count <= 120:
            quality_score += 0.1
        elif word_count < 20 or word_count > 150:
            quality_score -= 0.2
        
        # Check for natural language patterns
        if any(pattern in response.lower() for pattern in ["i understand", "that's interesting", "let me help"]):
            quality_score += 0.05
        
        # Check for schedule references when relevant
        if any(word in response.lower() for word in ["10:00", "17:00", "meeting", "gym"]):
            quality_score += 0.05
        
        return max(0.7, min(1.0, quality_score))
    
    def _assess_context_relevance(self, user_input: str, response: str) -> float:
        """Assess how relevant the response is to the user input"""
        user_words = set(user_input.lower().split())
        response_words = set(response.lower().split())
        
        # Calculate word overlap
        common_words = user_words.intersection(response_words)
        if len(user_words) > 0:
            relevance = len(common_words) / len(user_words)
        else:
            relevance = 0.0
        
        return min(1.0, relevance + 0.3)  # Base relevance of 0.3
    
    def _generate_enhanced_suggestions(self, user_input: str, assistant_response: str, intent_data: Dict[str, Any]) -> List[str]:
        """Generate enhanced contextual suggestions"""
        suggestions = []
        intent = intent_data["intent"]
        
        # Schedule-related suggestions
        if intent in ["schedule_query", "meeting_query", "gym_query"]:
            suggestions.extend([
                "What's my next meeting?",
                "When is my gym session?",
                "Help me prepare for the meeting",
                "Set a reminder for my appointments"
            ])
        
        # Task-related suggestions
        elif intent in ["task_management", "reminder_request"]:
            suggestions.extend([
                "What should I prioritize today?",
                "Set a reminder for me",
                "Help me organize my day",
                "Create a to-do list"
            ])
        
        # Entertainment suggestions
        elif intent == "entertainment":
            suggestions.extend([
                "Tell me another joke",
                "Share a motivational quote",
                "Tell me a short story",
                "Play a word game"
            ])
        
        # Emotional support suggestions
        elif intent == "emotional_support":
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
        
        # Add personalized suggestions based on user preferences
        if self.user_preferences.get("favorite_topics"):
            suggestions.extend(self.user_preferences["favorite_topics"][:2])
        
        return suggestions[:4]  # Return top 4 suggestions
    
    def _update_conversation_insights(self, user_input: str, assistant_response: str, intent_data: Dict[str, Any]):
        """Update conversation insights and user preferences"""
        # Track topics
        if intent_data["intent"] not in self.topic_history:
            self.topic_history.append(intent_data["intent"])
        
        # Keep only recent topics
        if len(self.topic_history) > 10:
            self.topic_history = self.topic_history[-10:]
        
        # Update user preferences based on interaction patterns
        if intent_data["confidence"] > 0.8:
            if intent_data["intent"] not in self.user_preferences.get("favorite_topics", []):
                if "favorite_topics" not in self.user_preferences:
                    self.user_preferences["favorite_topics"] = []
                self.user_preferences["favorite_topics"].append(intent_data["intent"])
        
        # Track conversation patterns
        if "conversation_patterns" not in self.user_preferences:
            self.user_preferences["conversation_patterns"] = {}
        
        pattern_key = f"{intent_data['intent']}_count"
        self.user_preferences["conversation_patterns"][pattern_key] = \
            self.user_preferences["conversation_patterns"].get(pattern_key, 0) + 1
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation insights"""
        return {
            "total_interactions": self.conversation_stats['total_interactions'],
            "favorite_topics": self.user_preferences.get("favorite_topics", []),
            "conversation_patterns": self.user_preferences.get("conversation_patterns", {}),
            "recent_topics": self.topic_history[-5:] if self.topic_history else [],
            "average_response_time": sum(self.conversation_stats['response_times']) / len(self.conversation_stats['response_times']) if self.conversation_stats['response_times'] else 0
        }
    
    def clear_history(self):
        """Clear the conversation history and reset insights"""
        self.conversation_history = []
        self.topic_history = []
        self.emotion_tracker = {}
        self.conversation_stats = {
            'total_interactions': 0,
            'favorite_topics': [],
            'response_times': [],
            'user_satisfaction': []
        }
        print("✅ Conversation history and insights cleared")
    
    def set_user_preference(self, key: str, value: Any):
        """Set a user preference"""
        self.user_preferences[key] = value
        print(f"✅ User preference '{key}' set to '{value}'")
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get current user preferences"""
        return self.user_preferences.copy()