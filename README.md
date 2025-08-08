# ChatGPT-like Voice Assistant

An advanced AI-powered voice assistant with ChatGPT-like conversational capabilities, built using ElevenLabs API and enhanced natural language understanding.

## 🚀 Features

### ChatGPT-like Capabilities
- **Natural Conversation**: Understands and responds to any type of user input naturally
- **Context-Aware Responses**: Remembers conversation history and provides contextual responses
- **Intelligent Q&A**: Answers general questions with thoughtful, helpful responses
- **Task Management**: Helps with schedules, reminders, and productivity
- **Entertainment**: Tells jokes and engages in friendly conversation
- **Conversational Suggestions**: Provides relevant follow-up questions and suggestions

### Technical Features
- **Advanced NLU Engine**: Machine learning and rule-based intent recognition
- **Voice Generation**: High-quality voice synthesis with ElevenLabs
- **Web Interface**: Modern, responsive UI with real-time conversation
- **Context Management**: Maintains conversation history and user preferences
- **Multi-Modal Interaction**: Text and voice input/output support

## 🛠️ Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure environment**:
   - Copy `env.example` to `.env`
   - Add your ElevenLabs API key to the `.env` file:
   ```
   ELEVENLABS_API_KEY=your_api_key_here
   ```

3. **Run the application**:
```bash
python app.py
```

4. **Access the web interface**:
   - Open your browser to `http://localhost:5000`
   - Start chatting with your ChatGPT-like assistant!

## 🎯 Usage Examples

### Basic Conversation
```
User: "Hello!"
Assistant: "Hello Alex! How can I help you today? I can assist with your schedule, answer questions, help you stay organized, or just chat with you. What's on your mind?"
```

### Schedule Management
```
User: "What's on my schedule today?"
Assistant: "Looking at your schedule, Alex, you have a Sales Meeting with Taipy at 10:00 and a gym session with Sophie at 17:00. How can I help you manage your time today?"
```

### General Questions
```
User: "Tell me a joke"
Assistant: "I'd love to share a joke with you, Alex! Here's one: Why don't scientists trust atoms? Because they make up everything! 😄 How can I help you with your schedule today?"
```

### Task Assistance
```
User: "Help me with my tasks"
Assistant: "I can help you manage tasks and reminders, Alex! Would you like me to help you organize your day, set up some reminders, or just chat about what you need to get done? I'm here to help you stay productive."
```

## 🏗️ Project Structure

```
elevenlabs-project/
├── app.py                      # Main Flask application
├── advanced_nlu_engine.py      # ChatGPT-like NLU engine
├── nlu_engine.py              # Basic NLU engine (fallback)
├── voice_assistant.py         # Voice assistant utilities
├── test_chatgpt_features.py   # Test script for features
├── templates/
│   └── index.html             # Modern web interface
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── README.md                 # This file
```

## 🧠 NLU Engine

The assistant uses an advanced NLU (Natural Language Understanding) engine with:

- **Machine Learning**: TF-IDF vectorization and Random Forest classification
- **Rule-Based Patterns**: Enhanced regex and keyword matching
- **Context Awareness**: Conversation history and user preferences
- **Intent Recognition**: 15+ different intent types
- **Confidence Scoring**: Intelligent response selection

### Supported Intents
- Greetings and farewells
- Schedule and meeting queries
- Task management and reminders
- General questions and opinions
- Jokes and entertainment
- Personal questions
- Gratitude and appreciation

## 🎨 Web Interface

The modern web interface includes:
- **Real-time Conversation**: Live chat with typing indicators
- **Voice Selection**: Choose from available ElevenLabs voices
- **Conversation Suggestions**: Clickable suggestion chips
- **NLU Information**: Display of AI understanding and confidence
- **Activity Log**: Track conversation history
- **Responsive Design**: Works on desktop and mobile

## 🔧 Configuration

### Customizable Settings
- **User Name**: Personalize the assistant's responses
- **Schedule**: Configure user's daily schedule
- **System Prompt**: Define assistant's personality and capabilities
- **Voice Selection**: Choose preferred voice for responses

### Environment Variables
```bash
ELEVENLABS_API_KEY=your_api_key_here
```

## 🧪 Testing

Run the test script to verify ChatGPT-like features:
```bash
python test_chatgpt_features.py
```

This will test:
- Conversation capabilities
- NLU parsing accuracy
- System status and connectivity
- Response quality and variety

## 🔒 Security

- API keys are stored in environment variables
- No sensitive data is logged or stored
- Secure communication with ElevenLabs API
- Input validation and sanitization

## 🚀 Next Steps

Potential enhancements:
- **Voice Input**: Speech-to-text integration
- **Multi-language Support**: Internationalization
- **Custom Knowledge Base**: Domain-specific information
- **Integration APIs**: Connect with external services
- **Mobile App**: Native mobile application
- **Advanced Analytics**: Conversation insights and metrics

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions:
1. Check the existing issues
2. Create a new issue with detailed information
3. Include system information and error logs

---

**Enjoy your ChatGPT-like voice assistant! 🎉**
