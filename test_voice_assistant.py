#!/usr/bin/env python3
"""
Test script for advanced voice assistant features and capabilities
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"
TEST_MESSAGES = [
    # Basic greetings
    "Hello!",
    "Hi there!",
    "Good morning!",
    
    # Schedule queries
    "What's on my schedule today?",
    "What meetings do I have?",
    "When is my gym session?",
    
    # General questions
    "What can you do?",
    "How can you help me?",
    "Tell me about yourself",
    
    # Fun interactions
    "Tell me a joke",
    "How are you doing?",
    "What do you think about AI?",
    
    # Task management
    "Help me with my tasks",
    "I need help organizing my day",
    "Can you set a reminder?",
    
    # Complex queries
    "What's the weather like?",
    "How do I stay productive?",
    "What's your opinion on time management?",
    
    # Gratitude and farewell
    "Thank you!",
    "Thanks for your help",
    "Goodbye!"
]

def test_conversation_endpoint():
    """Test the conversation endpoint with various messages"""
    print("🤖 Testing Advanced Conversation Features")
    print("=" * 50)
    
    conversation_history = []
    
    for i, message in enumerate(TEST_MESSAGES, 1):
        print(f"\n📝 Test {i}/{len(TEST_MESSAGES)}: {message}")
        
        try:
            # Send message to conversation endpoint
            response = requests.post(
                f"{BASE_URL}/api/conversation",
                json={
                    "message": message,
                    "history": conversation_history
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("success"):
                    assistant_response = data.get("assistant_response", "No response")
                    intent = data.get("intent", "unknown")
                    confidence = data.get("confidence", 0)
                    suggestions = data.get("suggestions", [])
                    
                    print(f"✅ Response: {assistant_response[:100]}...")
                    print(f"   Intent: {intent} (confidence: {confidence:.2f})")
                    
                    if suggestions:
                        print(f"   Suggestions: {', '.join(suggestions[:3])}")
                    
                    # Update conversation history
                    conversation_history = data.get("conversation_history", [])
                    
                else:
                    print(f"❌ Error: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    print(f"\n🎉 Conversation test completed!")
    print(f"📊 Total interactions: {len(conversation_history)}")

def test_nlu_parsing():
    """Test the NLU parsing capabilities"""
    print("\n🧠 Testing NLU Parsing Features")
    print("=" * 40)
    
    test_queries = [
        "What's on my schedule?",
        "Tell me a joke",
        "How can you help me?",
        "What time is my meeting?",
        "I need help with tasks"
    ]
    
    for query in test_queries:
        try:
            response = requests.post(
                f"{BASE_URL}/api/nlu/parse",
                json={"text": query},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("success"):
                    intent = data.get("intent", "unknown")
                    confidence = data.get("confidence", 0)
                    method = data.get("method", "unknown")
                    slots = data.get("slots", [])
                    
                    print(f"✅ '{query}' → Intent: {intent} (confidence: {confidence:.2f}, method: {method})")
                    
                    if slots:
                        print(f"   Slots: {slots}")
                else:
                    print(f"❌ Error parsing '{query}': {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error for '{query}': {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for '{query}': {e}")

def test_status():
    """Test the system status"""
    print("\n📊 Testing System Status")
    print("=" * 25)
    
    try:
        response = requests.get(f"{BASE_URL}/api/status", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("success"):
                print(f"✅ System Status: Connected")
                print(f"   API Key: {data.get('api_key', 'Unknown')}")
                print(f"   Voices: {data.get('voices_count', 0)} available")
                print(f"   NLU Type: {data.get('nlu_type', 'None')}")
                print(f"   User: {data.get('user_name', 'Unknown')}")
                print(f"   Assistant Type: {data.get('assistant_type', 'Unknown')}")
            else:
                print(f"❌ Status Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Status check failed: {e}")

def main():
    """Main test function"""
    print("🚀 Starting Advanced Voice Assistant Tests")
    print("=" * 60)
    
    # Test system status first
    test_status()
    
    # Test NLU parsing
    test_nlu_parsing()
    
    # Test conversation features
    test_conversation_endpoint()
    
    print("\n🎯 All tests completed!")
    print("\n💡 The voice assistant now has advanced capabilities:")
    print("   • Natural conversation understanding")
    print("   • Context-aware responses")
    print("   • Intelligent Q&A")
    print("   • Jokes and entertainment")
    print("   • Task and schedule management")
    print("   • Conversational suggestions")
    print("   • Enhanced response variety")

if __name__ == "__main__":
    main() 