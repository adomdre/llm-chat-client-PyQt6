#!/usr/bin/python3
"""
LLM Chat Client - PyQt6 Edition
A feature-rich chat interface for Claude AI with streaming, search, and export capabilities
Requirements: pip install PyQt6 anthropic python-dotenv tiktoken
"""

import os
import sys
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime
import re

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QFrame, QScrollArea, QLineEdit,
    QDialog, QSlider, QRadioButton, QButtonGroup, QFileDialog, QMessageBox,
    QSplitter, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, QThread, QSize, QPropertyAnimation,
    QEasingCurve, QPoint
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QTextCursor, QPixmap, QIcon, QTextCharFormat,
    QPainter, QLinearGradient, QPen
)

from dotenv import load_dotenv
from anthropic import Anthropic

# Optional dependencies
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

APP_TITLE = "LLM Chat Client"
MODEL = "claude-sonnet-4-20250514"

# Sampling controls
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_P = None

current_temperature = DEFAULT_TEMPERATURE
current_top_p = DEFAULT_TOP_P
current_sampling_mode = "temperature"

# Modern color scheme - customize to your preference
COLORS = {
    "bg": "#0A2E2E",
    "text": "#F0FFFE",
    "muted": "#8FD9D0",
    "border": "#1A4444",
    "accent": "#20B2AA",
    "accent_hover": "#48D1CC",
    "cursor": "#7FFFD4",
    "selection_bg": "#2E8B87",
    "panel": "#0F3838",
    "panel_high": "#144E4E",
    "user_bubble": "#1A5555",
    "bot_bubble": "#0F3838",
    "entry_bg": "#083030",
    "chat_area": "#0A2828",
    "button_disabled": "#1A4444",
}

# Default system prompt - customize for your use case
SYSTEM_PROMPT = """You are a helpful AI assistant. Provide clear, concise, and accurate responses."""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Conversation storage
CONVERSATIONS_DIR = Path.home() / ".llm_chat_conversations"
CONVERSATIONS_DIR.mkdir(exist_ok=True)

current_conversation_id = None
current_conversation_title = "New Chat"

# Settings storage
SETTINGS_FILE = Path.home() / ".llm_chat_settings.json"


def load_settings():
    """Load settings from disk."""
    global current_temperature, current_top_p, current_sampling_mode
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                current_temperature = settings.get("temperature", DEFAULT_TEMPERATURE)
                current_top_p = settings.get("top_p", DEFAULT_TOP_P)
                current_sampling_mode = settings.get("sampling_mode", "temperature")
        except Exception:
            pass


def save_settings():
    """Save settings to disk."""
    settings = {
        "temperature": current_temperature,
        "top_p": current_top_p,
        "sampling_mode": current_sampling_mode
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def generate_conversation_id():
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())


def get_conversation_title(msgs):
    """Generate a title from the first user message."""
    for msg in msgs:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            title = content[:40].strip()
            if len(content) > 40:
                title += "..."
            return title or "New Chat"
    return "New Chat"


def save_conversation(conv_id: str, title: str, msgs: list):
    """Save a conversation to disk."""
    if not conv_id:
        return
    filepath = CONVERSATIONS_DIR / f"{conv_id}.json"
    data = {
        "id": conv_id,
        "title": title,
        "messages": msgs,
        "timestamp": datetime.now().isoformat()
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_conversation(conv_id):
    """Load a conversation from disk."""
    filepath = CONVERSATIONS_DIR / f"{conv_id}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def list_conversations():
    """List all saved conversations, sorted by timestamp."""
    conversations = []
    for filepath in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                conversations.append(data)
        except Exception:
            continue
    conversations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return conversations


def delete_conversation(conv_id: str):
    """Delete a conversation from disk."""
    filepath = CONVERSATIONS_DIR / f"{conv_id}.json"
    if filepath.exists():
        filepath.unlink()


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text) // 4


def load_api_key() -> str:
    """Load API key from environment or prompt user."""
    candidates = [
        Path.home() / ".llm_chat.env",
        Path.cwd() / ".env",
        Path(getattr(sys, "_MEIPASS", Path.cwd())) / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p)
            break
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key

    # For PyQt6, use QInputDialog
    from PyQt6.QtWidgets import QInputDialog
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    entered, ok = QInputDialog.getText(
        None,
        "API Key Required",
        "Enter your ANTHROPIC_API_KEY (it will be saved to ~/.llm_chat.env):",
        echo=QLineEdit.EchoMode.Password
    )

    if ok and entered.strip():
        (Path.home() / ".llm_chat.env").write_text(f"ANTHROPIC_API_KEY={entered.strip()}\n")
        os.environ["ANTHROPIC_API_KEY"] = entered.strip()
        return entered.strip()

    QMessageBox.critical(
        None,
        "Missing API Key",
        "ANTHROPIC_API_KEY not found.\n\nAdd it to ~/.llm_chat.env or create a .env file, then relaunch."
    )
    sys.exit(1)


API_KEY = load_api_key()
client = Anthropic(api_key=API_KEY)


def _sanitize_sampling(temperature=None, top_p=None):
    """Clamp to valid ranges and enforce Anthropic rule."""
    def clamp(x, lo, hi):
        try:
            return max(lo, min(hi, float(x)))
        except Exception:
            return None
    t = clamp(temperature, 0.0, 1.0) if temperature is not None else None
    p = clamp(top_p, 0.0, 1.0) if top_p is not None else None
    if t is not None and p is not None:
        p = None
    return t, p


def _create_args(model_name: str, system: str, convo):
    """Build kwargs for messages.create."""
    temp = current_temperature if current_sampling_mode == "temperature" else None
    top_p_val = current_top_p if current_sampling_mode == "top_p" else None
    t, p = _sanitize_sampling(temp, top_p_val)
    kwargs = dict(model=model_name, system=system, messages=convo, max_tokens=4096)
    if t is not None:
        kwargs["temperature"] = t
    elif p is not None:
        kwargs["top_p"] = p
    return kwargs


# Signal emitter for thread-safe GUI updates
class StreamSignals(QObject):
    text_chunk = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)


class StreamWorker(QThread):
    """Worker thread for streaming API responses."""

    def __init__(self, user_text, messages_list):
        super().__init__()
        self.user_text = user_text
        self.messages_list = messages_list
        self.signals = StreamSignals()

    def run(self):
        """Run the streaming API call in a separate thread."""
        try:
            self.messages_list.append({"role": "user", "content": self.user_text})
            system = "\n".join([m["content"] for m in self.messages_list if m["role"] == "system"]) or SYSTEM_PROMPT
            convo = [m for m in self.messages_list if m["role"] in ("user", "assistant")]

            kwargs = _create_args(MODEL, system, convo)
            out = ""

            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    out += text
                    self.signals.text_chunk.emit(text)

            if not out:
                out = "(Empty response)"

            self.messages_list.append({"role": "assistant", "content": out})
            self.signals.finished.emit(out)

        except Exception as e:
            error_msg = f"(API error: {type(e).__name__}: {e})"
            self.messages_list.append({"role": "assistant", "content": error_msg})
            self.signals.error.emit(error_msg)


# Loading Screen
class LoadingScreen(QWidget):
    """Custom loading screen."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Simple text loading screen
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg']};
                border: 2px solid {COLORS['accent']};
                border-radius: 10px;
            }}
        """)
        container_layout = QVBoxLayout()

        title = QLabel("LLM Chat Client")
        title.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['accent']}; background: transparent; border: none;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Powered by Claude AI")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setStyleSheet(f"color: {COLORS['muted']}; background: transparent; border: none;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        loading_label = QLabel("Loading...")
        loading_label.setFont(QFont("Arial", 11))
        loading_label.setStyleSheet(f"color: {COLORS['text']}; background: transparent; border: none;")
        loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container_layout.addStretch()
        container_layout.addWidget(title)
        container_layout.addWidget(subtitle)
        container_layout.addSpacing(20)
        container_layout.addWidget(loading_label)
        container_layout.addStretch()

        container.setLayout(container_layout)
        layout.addWidget(container)
        self.setFixedSize(400, 250)

        self.setLayout(layout)

        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)


# Main Application Window
class LLMChatClient(QMainWindow):
    """Main application window for LLM Chat Client."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(820, 520)
        self.resize(980, 640)

        # Set window background
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["bg"]))
        self.setPalette(palette)

        # Initialize conversation state
        global current_conversation_id, current_conversation_title, messages
        current_conversation_id = generate_conversation_id()
        current_conversation_title = "New Chat"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Load settings
        load_settings()

        # Setup UI
        self.setup_ui()

        # Initial bot message
        self.append_bot_message("Hello! How can I assist you today?")
        self.update_token_counter()

        # Focus input
        self.input_text.setFocus()

    def setup_ui(self):
        """Setup the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(8)

        # Create splitter for resizable sidebar
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Sidebar
        self.sidebar = self.create_sidebar()
        splitter.addWidget(self.sidebar)

        # Chat area
        chat_widget = self.create_chat_area()
        splitter.addWidget(chat_widget)

        # Set initial sizes
        splitter.setSizes([250, 730])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

        # Apply stylesheet
        self.apply_stylesheet()

        # Setup keyboard shortcuts
        self.setup_shortcuts()

    def create_sidebar(self):
        """Create the conversation history sidebar."""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(200)
        sidebar.setMaximumWidth(400)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # New Chat button
        new_chat_btn = QPushButton("+ New Chat")
        new_chat_btn.setObjectName("newChatBtn")
        new_chat_btn.clicked.connect(self.start_new_chat)
        layout.addWidget(new_chat_btn)

        # Chat History label
        history_label = QLabel("Chat History")
        history_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        history_label.setStyleSheet(f"color: {COLORS['text']}; padding: 4px;")
        layout.addWidget(history_label)

        # Search box
        self.search_entry = QLineEdit()
        self.search_entry.setPlaceholderText("Search conversations...")
        self.search_entry.setObjectName("searchEntry")
        self.search_entry.textChanged.connect(self.refresh_conversation_list)
        layout.addWidget(self.search_entry)

        # Conversation list (scroll area)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.conv_list_widget = QWidget()
        self.conv_list_layout = QVBoxLayout()
        self.conv_list_layout.setContentsMargins(0, 0, 0, 0)
        self.conv_list_layout.setSpacing(2)
        self.conv_list_layout.addStretch()
        self.conv_list_widget.setLayout(self.conv_list_layout)

        scroll.setWidget(self.conv_list_widget)
        layout.addWidget(scroll)

        sidebar.setLayout(layout)
        return sidebar

    def create_chat_area(self):
        """Create the main chat area."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        # Header
        header = self.create_header()
        layout.addWidget(header)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setObjectName("chatDisplay")
        self.chat_display.setFont(QFont("Arial", 12))
        layout.addWidget(self.chat_display)

        # Input area
        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)

        self.input_text = QTextEdit()
        self.input_text.setObjectName("inputText")
        self.input_text.setFont(QFont("Arial", 12))
        self.input_text.setPlaceholderText("Type a message...")
        self.input_text.setMaximumHeight(80)
        self.input_text.installEventFilter(self)
        input_layout.addWidget(self.input_text)

        # Buttons
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(4)

        self.send_btn = QPushButton("Send ↩")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.clicked.connect(self.send_message)
        btn_layout.addWidget(self.send_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self.start_new_chat)
        btn_layout.addWidget(self.clear_btn)

        input_layout.addLayout(btn_layout)
        layout.addLayout(input_layout)

        widget.setLayout(layout)
        return widget

    def create_header(self):
        """Create the chat header with title and buttons."""
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)

        # Title section
        title_layout = QVBoxLayout()
        title_label = QLabel("LLM Chat Client")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {COLORS['text']};")

        subtitle_label = QLabel("Powered by Claude AI")
        subtitle_label.setFont(QFont("Arial", 11))
        subtitle_label.setStyleSheet(f"color: {COLORS['muted']};")

        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        layout.addLayout(title_layout)

        layout.addStretch()

        # Token counter
        self.token_label = QLabel("Tokens: 0")
        self.token_label.setFont(QFont("Arial", 11))
        self.token_label.setStyleSheet(f"color: {COLORS['muted']};")
        layout.addWidget(self.token_label)

        # Export button
        export_btn = QPushButton("Export")
        export_btn.setObjectName("headerBtn")
        export_btn.clicked.connect(self.export_conversation)
        layout.addWidget(export_btn)

        # Settings button
        settings_btn = QPushButton("Settings")
        settings_btn.setObjectName("headerBtn")
        settings_btn.clicked.connect(self.show_settings_dialog)
        layout.addWidget(settings_btn)

        header.setLayout(layout)
        return header

    def apply_stylesheet(self):
        """Apply custom stylesheet."""
        stylesheet = f"""
            QMainWindow {{
                background-color: {COLORS['bg']};
            }}

            QFrame#sidebar {{
                background-color: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}

            QFrame#header {{
                background-color: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px;
            }}

            QPushButton#newChatBtn {{
                background-color: {COLORS['accent']};
                color: #0A1222;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
                font-size: 12px;
            }}

            QPushButton#newChatBtn:hover {{
                background-color: {COLORS['accent_hover']};
            }}

            QPushButton#sendBtn, QPushButton#clearBtn {{
                background-color: {COLORS['accent']};
                color: #0A1222;
                border: none;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: bold;
                font-size: 12px;
                min-width: 80px;
            }}

            QPushButton#sendBtn:hover, QPushButton#clearBtn:hover {{
                background-color: {COLORS['accent_hover']};
            }}

            QPushButton#sendBtn:disabled, QPushButton#clearBtn:disabled {{
                background-color: {COLORS['button_disabled']};
                color: {COLORS['muted']};
            }}

            QPushButton#headerBtn {{
                background-color: {COLORS['accent']};
                color: #0A1222;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }}

            QPushButton#headerBtn:hover {{
                background-color: {COLORS['accent_hover']};
            }}

            QLineEdit#searchEntry {{
                background-color: {COLORS['entry_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 6px;
                font-size: 11px;
            }}

            QLineEdit#searchEntry:focus {{
                border: 1px solid {COLORS['accent']};
            }}

            QTextEdit#chatDisplay {{
                background-color: {COLORS['chat_area']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 16px;
            }}

            QTextEdit#inputText {{
                background-color: {COLORS['entry_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px;
            }}

            QTextEdit#inputText:focus {{
                border: 1px solid {COLORS['accent']};
            }}

            QScrollBar:vertical {{
                background-color: {COLORS['panel']};
                width: 12px;
                border-radius: 6px;
            }}

            QScrollBar::handle:vertical {{
                background-color: {COLORS['accent']};
                border-radius: 6px;
                min-height: 20px;
            }}

            QScrollBar::handle:vertical:hover {{
                background-color: {COLORS['accent_hover']};
            }}

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """
        self.setStyleSheet(stylesheet)

    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        from PyQt6.QtGui import QShortcut, QKeySequence

        # Cmd+N / Ctrl+N - New chat
        QShortcut(QKeySequence("Ctrl+N"), self).activated.connect(self.start_new_chat)

        # Cmd+E / Ctrl+E - Export
        QShortcut(QKeySequence("Ctrl+E"), self).activated.connect(self.export_conversation)

        # Cmd+K / Ctrl+K - Focus search
        QShortcut(QKeySequence("Ctrl+K"), self).activated.connect(lambda: self.search_entry.setFocus())

    def eventFilter(self, obj, event):
        """Handle input text key events."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent

        if obj == self.input_text and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
                if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                    # Shift+Enter - new line
                    return False
                else:
                    # Enter - send message
                    self.send_message()
                    return True
        return super().eventFilter(obj, event)

    def append_user_message(self, text):
        """Append a user message to the chat display."""
        timestamp = datetime.now().strftime("%H:%M")
        self.chat_display.append(f"🧑 [{timestamp}] You: {text}\n")
        self.chat_display.ensureCursorVisible()

    def append_bot_message(self, text):
        """Append a bot message to the chat display."""
        timestamp = datetime.now().strftime("%H:%M")
        self.chat_display.append(f"🤖 [{timestamp}] Assistant: {text}\n")
        self.chat_display.ensureCursorVisible()

    def append_bot_chunk(self, text):
        """Append a chunk of text to the current bot message (streaming)."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def send_message(self):
        """Send a message to the LLM."""
        text = self.input_text.toPlainText().strip()
        if not text:
            return

        # Clear input
        self.input_text.clear()

        # Display user message
        self.append_user_message(text)

        # Disable input during processing
        self.input_text.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        # Show bot message header
        timestamp = datetime.now().strftime("%H:%M")
        self.chat_display.append(f"🤖 [{timestamp}] Assistant: ")

        # Start streaming worker
        self.worker = StreamWorker(text, messages)
        self.worker.signals.text_chunk.connect(self.append_bot_chunk)
        self.worker.signals.finished.connect(self.on_response_finished)
        self.worker.signals.error.connect(self.on_response_error)
        self.worker.start()

    def on_response_finished(self, full_response):
        """Called when streaming finishes."""
        self.chat_display.append("\n")
        self.auto_save_current_conversation()
        self.refresh_conversation_list()

        # Re-enable input
        self.input_text.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.input_text.setFocus()

    def on_response_error(self, error_msg):
        """Called when an error occurs."""
        self.on_response_finished(error_msg)

    def start_new_chat(self):
        """Start a new chat conversation."""
        global messages, current_conversation_id, current_conversation_title

        self.chat_display.clear()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        current_conversation_id = generate_conversation_id()
        current_conversation_title = "New Chat"

        self.append_bot_message("Hello! How can I assist you today?")
        self.refresh_conversation_list()
        self.update_token_counter()
        self.input_text.setFocus()

    def auto_save_current_conversation(self):
        """Auto-save the current conversation."""
        global current_conversation_id, current_conversation_title
        if not current_conversation_id:
            current_conversation_id = generate_conversation_id()
        if current_conversation_title == "New Chat":
            current_conversation_title = get_conversation_title(messages)
        save_conversation(current_conversation_id, current_conversation_title, messages)
        self.update_token_counter()

    def update_token_counter(self):
        """Update the token counter display."""
        total_tokens = 0
        for msg in messages:
            if msg.get("role") in ("user", "assistant"):
                total_tokens += estimate_tokens(msg.get("content", ""))
        self.token_label.setText(f"Tokens: ~{total_tokens}")

    def refresh_conversation_list(self):
        """Refresh the conversation list in the sidebar."""
        # Clear existing items
        while self.conv_list_layout.count() > 1:  # Keep the stretch
            item = self.conv_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        conversations = list_conversations()

        # Filter by search query
        search_query = self.search_entry.text().strip().lower()
        if search_query:
            conversations = [
                c for c in conversations
                if search_query in c.get("title", "").lower()
                or any(search_query in msg.get("content", "").lower()
                       for msg in c.get("messages", []))
            ]

        for conv_data in conversations:
            conv_item = self.create_conversation_item(conv_data)
            self.conv_list_layout.insertWidget(self.conv_list_layout.count() - 1, conv_item)

    def create_conversation_item(self, conv_data):
        """Create a conversation item widget."""
        conv_id = conv_data.get("id")
        title = conv_data.get("title", "Untitled")
        timestamp = conv_data.get("timestamp", "")

        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%b %d, %H:%M")
        except Exception:
            time_str = ""

        is_active = (conv_id == current_conversation_id)
        bg_color = COLORS["user_bubble"] if is_active else COLORS["panel"]

        frame = QFrame()
        frame.setObjectName("convItem")
        frame.setStyleSheet(f"""
            QFrame#convItem {{
                background-color: {bg_color};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px;
            }}
            QFrame#convItem:hover {{
                background-color: {COLORS['panel_high']};
            }}
        """)
        frame.setCursor(Qt.CursorShape.PointingHandCursor)
        frame.mousePressEvent = lambda event: self.load_conversation_to_chat(conv_id)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 11))
        title_label.setStyleSheet(f"color: {COLORS['text']}; background: transparent; border: none;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        if time_str:
            time_label = QLabel(time_str)
            time_label.setFont(QFont("Arial", 9))
            time_label.setStyleSheet(f"color: {COLORS['muted']}; background: transparent; border: none;")
            layout.addWidget(time_label)

        frame.setLayout(layout)
        return frame

    def load_conversation_to_chat(self, conv_id):
        """Load a saved conversation into the chat."""
        global messages, current_conversation_id, current_conversation_title

        conv_data = load_conversation(conv_id)
        if not conv_data:
            return

        # Clear chat
        self.chat_display.clear()

        # Load messages
        messages = conv_data.get("messages", [{"role": "system", "content": SYSTEM_PROMPT}])
        current_conversation_id = conv_id
        current_conversation_title = conv_data.get("title", "Untitled")

        # Display messages
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                timestamp = datetime.now().strftime("%H:%M")
                self.chat_display.append(f"🧑 [{timestamp}] You: {content}\n")
            elif role == "assistant":
                timestamp = datetime.now().strftime("%H:%M")
                self.chat_display.append(f"🤖 [{timestamp}] Assistant: {content}\n")

        self.refresh_conversation_list()
        self.update_token_counter()

    def export_conversation(self):
        """Export the current conversation to markdown."""
        global current_conversation_id

        if not current_conversation_id:
            QMessageBox.warning(self, "Export Error", "No conversation to export.")
            return

        conv_data = load_conversation(current_conversation_id)
        if not conv_data:
            QMessageBox.critical(self, "Export Error", "Could not load conversation.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Conversation",
            f"{conv_data.get('title', 'conversation')}.md",
            "Markdown files (*.md);;Text files (*.txt);;All files (*.*)"
        )

        if not filename:
            return

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# {conv_data.get('title', 'Conversation')}\n\n")
                f.write(f"*Exported from LLM Chat Client on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
                f.write("---\n\n")

                for msg in conv_data.get("messages", []):
                    role = msg.get("role")
                    content = msg.get("content", "")

                    if role == "user":
                        f.write(f"## 🧑 User\n\n{content}\n\n")
                    elif role == "assistant":
                        f.write(f"## 🤖 Assistant\n\n{content}\n\n")

            QMessageBox.information(self, "Export Successful", f"Conversation exported to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export conversation:\n{e}")

    def show_settings_dialog(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self)
        dialog.exec()


class SettingsDialog(QDialog):
    """Settings dialog for adjusting model parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(450, 350)
        self.setModal(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("Model Settings")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text']};")
        layout.addWidget(title)

        # Sampling mode selection
        mode_label = QLabel("Sampling Mode:")
        mode_label.setFont(QFont("Arial", 12))
        mode_label.setStyleSheet(f"color: {COLORS['text']};")
        layout.addWidget(mode_label)

        self.temp_radio = QRadioButton("Temperature (Randomness)")
        self.temp_radio.setStyleSheet(f"color: {COLORS['text']};")

        self.top_p_radio = QRadioButton("Top P (Nucleus Sampling)")
        self.top_p_radio.setStyleSheet(f"color: {COLORS['text']};")

        if current_sampling_mode == "temperature":
            self.temp_radio.setChecked(True)
        else:
            self.top_p_radio.setChecked(True)

        layout.addWidget(self.temp_radio)
        layout.addWidget(self.top_p_radio)

        # Temperature slider
        temp_label_text = QLabel(f"Temperature: {current_temperature:.2f}")
        temp_label_text.setStyleSheet(f"color: {COLORS['text']};")
        layout.addWidget(temp_label_text)

        temp_desc = QLabel("Higher = more creative/random (0.0 - 1.0)")
        temp_desc.setFont(QFont("Arial", 10))
        temp_desc.setStyleSheet(f"color: {COLORS['muted']};")
        layout.addWidget(temp_desc)

        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(int(current_temperature * 100))
        self.temp_slider.valueChanged.connect(
            lambda v: temp_label_text.setText(f"Temperature: {v/100:.2f}")
        )
        layout.addWidget(self.temp_slider)

        # Top P slider
        top_p_val = current_top_p if current_top_p is not None else 0.9
        top_p_label_text = QLabel(f"Top P: {top_p_val:.2f}")
        top_p_label_text.setStyleSheet(f"color: {COLORS['text']};")
        layout.addWidget(top_p_label_text)

        top_p_desc = QLabel("Higher = more diverse responses (0.0 - 1.0)")
        top_p_desc.setFont(QFont("Arial", 10))
        top_p_desc.setStyleSheet(f"color: {COLORS['muted']};")
        layout.addWidget(top_p_desc)

        self.top_p_slider = QSlider(Qt.Orientation.Horizontal)
        self.top_p_slider.setRange(0, 100)
        self.top_p_slider.setValue(int(top_p_val * 100))
        self.top_p_slider.valueChanged.connect(
            lambda v: top_p_label_text.setText(f"Top P: {v/100:.2f}")
        )
        layout.addWidget(self.top_p_slider)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: #0A1222;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
        """)
        apply_btn.clicked.connect(self.apply_settings)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['panel_high']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['user_bubble']};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # Apply dialog styling
        self.setStyleSheet(f"QDialog {{ background-color: {COLORS['panel']}; }}")

    def apply_settings(self):
        """Apply the settings and close dialog."""
        global current_temperature, current_top_p, current_sampling_mode

        current_sampling_mode = "temperature" if self.temp_radio.isChecked() else "top_p"
        current_temperature = self.temp_slider.value() / 100
        current_top_p = self.top_p_slider.value() / 100

        save_settings()
        QMessageBox.information(self, "Settings Saved", "Model settings have been updated!")
        self.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)

    # Show loading screen
    loading_screen = LoadingScreen()
    loading_screen.show()

    # Create main window (hidden initially)
    main_window = LLMChatClient()

    # Close loading screen and show main window after 1.5 seconds
    def show_main():
        loading_screen.close()
        main_window.show()
        main_window.refresh_conversation_list()

    QTimer.singleShot(1500, show_main)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
