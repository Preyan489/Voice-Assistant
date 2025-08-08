import os
import openai
from dotenv import load_dotenv
from typing import List, Dict, Any
import json

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

class LLMConversationHandler:
    def __init__(self):
        """Initialize the LLM conversation handler"""
        self.conversation_history = []
        self.system_prompt = self._get_system_prompt()
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the assistant"""
        return """You are an advanced AI voice assistant. Your responses should be natural, helpful, and concise since they will be converted to speech.

Key Information:
- User's schedule: Sales Meeting with Taipy at 10:00; Gym with Sophie at 17:00
- Keep responses under 100 words for better voice synthesis
- Be friendly and conversational
- Help with schedule management, tasks, and general queries
- You can engage in casual conversation and tell jokes
- Maintain context from the conversation history

Capabilities:
- Schedule management
- Task organization
- General conversation
- Answering questions
- Providing suggestions
- Entertainment (jokes, casual chat)

Format responses naturally but concisely, as they will be spoken aloud."""

    def get_response(self, user_input: str) -> Dict[str, Any]:
        """Get a response from the LLM"""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Prepare messages for the API
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + self.conversation_history[-5:]  # Keep last 5 messages for context
            
            # Get completion from OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,  # Limit response length for voice
                temperature=0.7,  # Some creativity but mostly consistent
                presence_penalty=0.6,  # Encourage varied responses
                frequency_penalty=0.3  # Reduce repetition
            )
            
            # Extract the response
            assistant_response = response.choices[0].message.content
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            
            # Analyze response for intent and confidence
            intent_data = self._analyze_response(user_input, assistant_response)
            
            # Generate suggestions based on the conversation
            suggestions = self._generate_suggestions(user_input, assistant_response)
            
            return {
                "success": True,
                "response": assistant_response,
                "intent": intent_data["intent"],
                "confidence": intent_data["confidence"],
                "suggestions": suggestions,
                "conversation_history": self.conversation_history
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I'm having trouble processing that request. Could you try again?"
            }
    
    def _analyze_response(self, user_input: str, assistant_response: str) -> Dict[str, Any]:
        """Analyze the response to determine intent and confidence"""
        # Simple intent analysis based on keywords
        intents = {
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
            "schedule": ["schedule", "meeting", "appointment", "gym"],
            "task": ["task", "todo", "reminder", "organize"],
            "question": ["what", "how", "why", "when", "where"],
            "joke": ["joke", "funny", "laugh"],
            "gratitude": ["thank", "thanks", "appreciate"],
            "farewell": ["goodbye", "bye", "see you", "farewell"]
        }
        
        # Check user input against intent keywords
        max_confidence = 0
        detected_intent = "general_chat"
        
        user_input_lower = user_input.lower()
        for intent, keywords in intents.items():
            matches = sum(1 for keyword in keywords if keyword in user_input_lower)
            confidence = matches / len(keywords) if matches > 0 else 0
            
            if confidence > max_confidence:
                max_confidence = confidence
                detected_intent = intent
        
        return {
            "intent": detected_intent,
            "confidence": max(0.6, max_confidence)  # Minimum 0.6 confidence
        }
    
    def _generate_suggestions(self, user_input: str, assistant_response: str) -> List[str]:
        """Generate contextual suggestions based on the conversation"""
        suggestions = []
        
        # Add schedule-related suggestions
        if any(word in user_input.lower() for word in ["schedule", "meeting", "time"]):
            suggestions.extend([
                "What's my next meeting?",
                "When is my gym session?",
                "Help me prepare for the meeting"
            ])
        
        # Add task-related suggestions
        elif any(word in user_input.lower() for word in ["task", "todo", "help"]):
            suggestions.extend([
                "What should I prioritize?",
                "Set a reminder for me",
                "Help me organize my day"
            ])
        
        # Add general suggestions
        else:
            suggestions.extend([
                "What's on my schedule?",
                "Tell me a joke",
                "How can you help me?"
            ])
        
        return suggestions[:3]  # Return top 3 suggestions

    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []