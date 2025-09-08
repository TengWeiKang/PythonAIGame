"""Theme management for the application."""

import tkinter as tk
from tkinter import ttk

class ThemeManager:
    """Manages application themes."""
    
    def __init__(self):
        self.current_theme = "default"
        self.themes = {
            "default": {
                "bg": "#f0f0f0",
                "fg": "#000000",
                "select_bg": "#0078d4",
                "select_fg": "#ffffff"
            },
            "dark": {
                "bg": "#2d2d2d",
                "fg": "#ffffff",
                "select_bg": "#404040",
                "select_fg": "#ffffff"
            }
        }

    def get_theme(self, name: str = None) -> dict:
        """Get theme configuration."""
        if name is None:
            name = self.current_theme
        return self.themes.get(name, self.themes["default"])

    def set_theme(self, name: str):
        """Set current theme."""
        if name in self.themes:
            self.current_theme = name

    def apply_theme(self, root: tk.Tk, theme_name: str = None):
        """Apply theme to root window."""
        theme = self.get_theme(theme_name)
        
        # Configure root
        root.configure(bg=theme["bg"])
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')  # Use clam as base theme
        
        # Configure ttk widget styles
        style.configure('TLabel', background=theme["bg"], foreground=theme["fg"])
        style.configure('TButton', background=theme["bg"], foreground=theme["fg"])
        style.configure('TFrame', background=theme["bg"])

def apply_theme(root: tk.Tk, theme_name: str = "default"):
    """Convenience function to apply theme."""
    manager = ThemeManager()
    manager.apply_theme(root, theme_name)