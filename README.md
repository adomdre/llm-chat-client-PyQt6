# llm-chat-client-pyqt6

An enhanced LLM-based chatbot application built with PyQt6. This is an evolution of my [tkinter version](https://github.com/adomdre/llm-chat-client-tkinter), rebuilt from the ground up to explore modern Qt framework capabilities and implement advanced features.

After building the initial tkinter version, I wanted to push my skills further by learning PyQt6 and implementing a more sophisticated desktop application with streaming responses, advanced UI components, and better user experience.

## Motivation

This project represents my progression in GUI development and API integration. I rebuilt the application in PyQt6 to:
- Learn a professional-grade GUI framework
- Implement real-time streaming responses
- Create a more polished, modern interface
- Explore advanced features like search, export, and token tracking
- Understand the differences between frameworks through hands-on experience

System parameters and APIs remain interchangeable. Tested with OpenAI, Meta, and Anthropic. Claude continues to deliver the most sophisticated responses for this application.

## Features

### Core Functionality
- **Real-time Streaming Responses**: Messages stream token-by-token using threaded workers for responsive UI
- **Conversation Management**: Full CRUD operations for chat sessions with automatic persistence
- **Search Functionality**: Quick search across all saved conversations with live filtering
- **Export Capability**: Export conversations to Markdown format for documentation or sharing
- **Token Counter**: Real-time token estimation with tiktoken integration

### Advanced UI/UX
- **Modern PyQt6 Interface**: Professional desktop application with native widgets and layouts
- **Customizable Parameters**: Temperature and Top-P controls with visual sliders and mode selection
- **Conversation Sidebar**: Browse, search, and manage chat history with context menus
- **Keyboard Shortcuts**: Quick actions (Cmd+N for new chat, Cmd+K for search, Cmd+E for export)
- **Loading Screen**: Professional splash screen during application startup
- **Responsive Design**: QSplitter-based layout for adjustable panels

### Technical Features
- **Thread-based Streaming**: Non-blocking API calls with Qt signals/slots for UI updates
- **Settings Persistence**: Model parameters saved across sessions
- **Context Menu Support**: Right-click functionality for chat operations
- **Error Handling**: Graceful fallback and user-friendly error messages
- **Cross-platform Support**: Works on macOS, Windows, and Linux

## Technologies Used

- **Python 3.x**
- **PyQt6**: Professional GUI framework with native widgets
- **Anthropic API**: Claude integration with streaming support
- **python-dotenv**: Secure environment variable management
- **tiktoken**: Accurate token counting for cost estimation
- **JSON**: Conversation data persistence
- **Threading**: Async operations with QThread

## Installation
```bash
# Clone the repository
git clone https://github.com/adomdre/llm-chat-client-pyqt6.git
cd llm-chat-client-pyqt6

# Install dependencies
pip install PyQt6 anthropic python-dotenv tiktoken

# Set up your API key
echo "ANTHROPIC_API_KEY=your_key_here" > ~/.llm_chat.env

# Run the application
python llm_chat_client_pyqt6.py
```

## What I Learned

Rebuilding this application in PyQt6 taught me:
- **Framework Architecture**: Understanding Model-View patterns and Qt's signal/slot system
- **Threaded Programming**: Implementing responsive UIs with background workers
- **Advanced Widgets**: Creating custom components, layouts, and styling
- **Stream Processing**: Handling real-time data streams and updating UI incrementally
- **Cross-platform Development**: Building applications that work across different operating systems

The transition from tkinter to PyQt6 highlighted how framework choice affects application capabilities, code organization, and user experience. PyQt6's event system and native widgets enabled features that would have been challenging in tkinter.

## Comparison with Tkinter Version

| Feature | Tkinter Version | PyQt6 Version |
|---------|----------------|---------------|
| Streaming | ❌ No | ✅ Yes |
| Search | ❌ No | ✅ Yes |
| Export | ❌ No | ✅ Yes |
| Token Counter | ❌ No | ✅ Yes |
| Keyboard Shortcuts | ❌ Limited | ✅ Full Support |
| Threading | Basic | Advanced (QThread) |
| UI Complexity | Simple | Professional |
| Platform Look | Generic | Native |

## Future Improvements

- Voice input/output integration
- Support for additional model providers
- Conversation analytics and insights
- Plugin system for extensibility
- Customizable themes and color schemes

## Acknowledgments

- **Andrew Ng** for his excellent DeepMind.AI courses that inspired this project
- **Anthropic** for providing the Claude API
- The **PyQt6 community** for extensive documentation and examples

## License

MIT License - Feel free to use and modify for your own projects!

---

*This project is part of my journey learning modern Python GUI development and LLM integration. Check out the [original tkinter version](https://github.com/adomdre/llm-chat-client-tkinter) to see where it started!*
