"""
Graphical Chat Desktop Application for Knowledge Layer RAG System.

A simple, user-friendly GUI for asking questions and getting answers
from the knowledge base.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import logging
from pathlib import Path
from typing import Optional
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.orchestrator import KnowledgeLayerPipeline

try:
    from src.config import Config, load_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None
    load_config = None

# Configure logging (suppress verbose logs in GUI)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChatApp:
    """Simple chat interface for Knowledge Layer Q&A."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Frankenstein Ai xd")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize pipeline
        self.pipeline = None
        self.initializing = False
        
        # Chat history
        self.chat_history = []
        
        # Setup UI
        self.setup_ui()
        
        # Initialize pipeline in background
        self.init_pipeline()
    
    def setup_ui(self):
        """Create and layout UI components."""
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="Frankenstein Ai",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            fg="#333"
        )
        title_label.pack(pady=(0, 10))
        
        # Status bar
        self.status_label = tk.Label(
            main_frame,
            text="Initializing...",
            font=("Arial", 9),
            bg="#f0f0f0",
            fg="#666",
            anchor="w"
        )
        self.status_label.pack(fill=tk.X, pady=(0, 10))
        
        # Chat display area
        chat_frame = tk.Frame(main_frame, bg="white", relief=tk.SUNKEN, borderwidth=1)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg="white",
            fg="#333",
            padx=10,
            pady=10,
            state=tk.DISABLED,
            relief=tk.FLAT
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for styling
        self.chat_display.tag_config("user", foreground="#0066cc", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("assistant", foreground="#333333", font=("Arial", 11))
        self.chat_display.tag_config("error", foreground="#cc0000", font=("Arial", 11))
        self.chat_display.tag_config("info", foreground="#666666", font=("Arial", 10, "italic"))
        
        # Input area
        input_frame = tk.Frame(main_frame, bg="#f0f0f0")
        input_frame.pack(fill=tk.X)
        
        # Question input
        self.question_entry = tk.Text(
            input_frame,
            height=3,
            font=("Arial", 11),
            wrap=tk.WORD,
            relief=tk.SOLID,
            borderwidth=1,
            padx=5,
            pady=5
        )
        self.question_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.question_entry.bind("<Return>", self.on_enter_key)
        self.question_entry.bind("<Shift-Return>", lambda e: None)  # Allow Shift+Enter for new line
        
        # Send button
        self.send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.send_question,
            font=("Arial", 11, "bold"),
            bg="#0066cc",
            fg="white",
            activebackground="#0052a3",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Settings button (optional)
        settings_button = tk.Button(
            main_frame,
            text="⚙ Settings",
            command=self.show_settings,
            font=("Arial", 9),
            bg="#e0e0e0",
            fg="#666",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor="hand2"
        )
        settings_button.pack(pady=(5, 0))
        
        # Welcome message
        self.add_message("system", "Welcome! I'm ready to answer questions from your knowledge base.\n")
    
    def init_pipeline(self):
        """Initialize the pipeline in a background thread."""
        self.initializing = True
        self.update_status("Initializing pipeline...")
        
        def init():
            try:
                # Load config if available
                config = None
                if CONFIG_AVAILABLE:
                    try:
                        config = load_config()
                    except Exception as e:
                        logger.warning(f"Failed to load config: {e}. Using defaults.")
                
                self.pipeline = KnowledgeLayerPipeline(config=config)
                self.root.after(0, self.on_pipeline_ready)
            except Exception as e:
                logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
                self.root.after(0, lambda: self.on_pipeline_error(str(e)))
        
        thread = threading.Thread(target=init, daemon=True)
        thread.start()
    
    def on_pipeline_ready(self):
        """Called when pipeline is ready."""
        self.initializing = False
        self.update_status("Ready - Ask me anything!")
        self.send_button.config(state=tk.NORMAL)
        self.question_entry.config(state=tk.NORMAL)
        self.add_message("system", "Pipeline initialized successfully. You can now ask questions!\n")
    
    def on_pipeline_error(self, error_msg: str):
        """Called when pipeline initialization fails."""
        self.initializing = False
        self.update_status(f"Error: {error_msg}")
        self.send_button.config(state=tk.DISABLED)
        self.question_entry.config(state=tk.DISABLED)
        self.add_message("error", f"Failed to initialize: {error_msg}\n")
        messagebox.showerror(
            "Initialization Error",
            f"Failed to initialize the knowledge base:\n\n{error_msg}\n\n"
            "Please ensure you have run the embedding step first."
        )
    
    def on_enter_key(self, event):
        """Handle Enter key press (send question)."""
        if event.state & 0x1:  # Shift key
            return  # Allow Shift+Enter for new line
        self.send_question()
        return "break"  # Prevent default behavior
    
    def send_question(self):
        """Send the question and get an answer."""
        question = self.question_entry.get("1.0", tk.END).strip()
        
        if not question:
            return
        
        if self.initializing or self.pipeline is None:
            messagebox.showwarning("Not Ready", "Please wait for the pipeline to initialize.")
            return
        
        # Clear input
        self.question_entry.delete("1.0", tk.END)
        
        # Disable input while processing
        self.send_button.config(state=tk.DISABLED)
        self.question_entry.config(state=tk.DISABLED)
        
        # Show user question
        self.add_message("user", f"Q: {question}\n\n")
        
        # Update status
        self.update_status("Searching knowledge base...")
        
        # Process in background thread
        def process_question():
            try:
                # Get search and context config
                search_config = {}
                context_config = {}
                if self.pipeline.config:
                    search_config = self.pipeline.config.search
                    context_config = self.pipeline.config.context
                
                # Detect if query is asking for "all" items (e.g., "all founders", "kik az alapító tagok")
                question_lower = question.lower()
                is_all_query = any(word in question_lower for word in ['all', 'összes', 'kik', 'minden', 'list', 'lista'])
                
                # Adjust top_k based on query type (use config values)
                if is_all_query:
                    top_k = search_config.get('top_k_all_queries', 50)
                    max_tokens = context_config.get('max_tokens_all_queries', 12000)
                else:
                    top_k = search_config.get('top_k_default', 20)
                    max_tokens = context_config.get('max_tokens_default', 8000)
                
                # Search and get context (using diverse search to include all sources)
                results, context = self.pipeline.step5_search(
                    query=question,
                    top_k=top_k,
                    build_context=True,
                    max_context_tokens=max_tokens,
                    diverse_search=True  # Ensure results from multiple sources
                )
                
                # Log the built context to console
                if context:
                    print("\n" + "="*80)
                    print("BUILT CONTEXT FOR LLM:")
                    print("="*80)
                    print(context)
                    print("="*80 + "\n")
                else:
                    print("\n" + "="*80)
                    print("BUILT CONTEXT FOR LLM: (empty)")
                    print("="*80 + "\n")
                
                if not results:
                    self.root.after(0, lambda: self.add_message(
                        "assistant",
                        "I couldn't find any relevant information in the knowledge base for your question.\n\n"
                    ))
                    self.root.after(0, lambda: self.update_status("Ready"))
                    return
                
                # Generate answer
                self.root.after(0, lambda: self.update_status("Generating answer..."))
                
                if context:
                    try:
                        # Get QA config
                        qa_config = {}
                        if self.pipeline.config:
                            qa_config = self.pipeline.config.llm.get('qa', {})
                        
                        answer = self.pipeline.step6_answer(
                            question=question,
                            context=context,
                            provider=qa_config.get('provider', 'ollama'),
                            model=qa_config.get('model', 'llama3'),
                            temperature=qa_config.get('temperature', 0.7),
                            max_tokens=qa_config.get('max_tokens', 500)
                        )
                    except Exception as e:
                        # Fallback if LLM fails
                        logger.warning(f"LLM answer failed: {e}, using top result")
                        metadata, score = results[0]
                        answer = metadata.get('text', 'No answer available.')[:500]
                else:
                    # Fallback: use top result text
                    metadata, score = results[0]
                    answer = metadata.get('text', 'No answer available.')[:500]
                
                # Show answer
                self.root.after(0, lambda: self.add_message("assistant", f"A: {answer}\n\n"))
                self.root.after(0, lambda: self.update_status("Ready"))
                
            except Exception as e:
                logger.error(f"Error processing question: {e}", exc_info=True)
                error_msg = str(e)
                self.root.after(0, lambda: self.add_message(
                    "error",
                    f"Error: {error_msg}\n\n"
                ))
                self.root.after(0, lambda: self.update_status("Error occurred"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Error",
                    f"An error occurred while processing your question:\n\n{error_msg}"
                ))
            finally:
                # Re-enable input
                self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.question_entry.config(state=tk.NORMAL))
        
        thread = threading.Thread(target=process_question, daemon=True)
        thread.start()
    
    def add_message(self, sender: str, message: str):
        """Add a message to the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        
        if sender == "user":
            self.chat_display.insert(tk.END, message, "user")
        elif sender == "assistant":
            self.chat_display.insert(tk.END, message, "assistant")
        elif sender == "error":
            self.chat_display.insert(tk.END, message, "error")
        else:  # system/info
            self.chat_display.insert(tk.END, message, "info")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)  # Auto-scroll to bottom
    
    def update_status(self, status: str):
        """Update the status label."""
        self.status_label.config(text=status)
    
    def show_settings(self):
        """Show settings dialog."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg="#f0f0f0")
        
        # Settings content
        tk.Label(
            settings_window,
            text="Settings",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0"
        ).pack(pady=20)
        
        tk.Label(
            settings_window,
            text="LLM Provider:",
            font=("Arial", 10),
            bg="#f0f0f0"
        ).pack(pady=5)
        
        provider_var = tk.StringVar(value="ollama")
        provider_combo = ttk.Combobox(
            settings_window,
            textvariable=provider_var,
            values=["ollama", "openai", "anthropic"],
            state="readonly"
        )
        provider_combo.pack(pady=5)
        
        tk.Label(
            settings_window,
            text="Model:",
            font=("Arial", 10),
            bg="#f0f0f0"
        ).pack(pady=5)
        
        model_var = tk.StringVar(value="llama3")
        model_entry = tk.Entry(settings_window, textvariable=model_var, width=30)
        model_entry.pack(pady=5)
        
        # Note: Settings would need to be saved and applied
        tk.Label(
            settings_window,
            text="Note: Settings will be applied to future questions.",
            font=("Arial", 9),
            bg="#f0f0f0",
            fg="#666"
        ).pack(pady=20)
        
        def close_settings():
            # TODO: Save settings
            settings_window.destroy()
        
        tk.Button(
            settings_window,
            text="Close",
            command=close_settings,
            bg="#0066cc",
            fg="white",
            padx=20,
            pady=5
        ).pack(pady=10)


def main():
    """Launch the chat application."""
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
