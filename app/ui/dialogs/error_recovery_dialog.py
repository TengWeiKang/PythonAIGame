"""Error recovery dialog with actionable solutions and guidance."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
import os
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    """Types of recovery actions."""
    RETRY = "retry"
    USE_DEFAULTS = "use_defaults"
    OPEN_SETTINGS = "open_settings"
    OPEN_FILE = "open_file"
    OPEN_URL = "open_url"
    RUN_COMMAND = "run_command"
    CONTACT_SUPPORT = "contact_support"
    IGNORE = "ignore"
    CANCEL = "cancel"


@dataclass
class RecoveryAction:
    """Recovery action that user can take."""
    type: ActionType
    label: str
    description: str
    callback: Optional[Callable] = None
    data: Optional[Any] = None
    primary: bool = False
    icon: str = ""


@dataclass
class ErrorDetails:
    """Detailed error information."""
    title: str
    message: str
    technical_details: Optional[str] = None
    error_code: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: str = "General"
    suggestions: List[str] = None
    actions: List[RecoveryAction] = None
    help_url: Optional[str] = None
    log_info: Optional[str] = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
        if self.actions is None:
            self.actions = []


class ErrorRecoveryDialog:
    """Error recovery dialog with actionable solutions."""

    # Color scheme
    COLORS = {
        'bg_primary': '#1e1e1e',
        'bg_secondary': '#2d2d2d',
        'bg_tertiary': '#3c3c3c',
        'text_primary': '#ffffff',
        'text_secondary': '#cccccc',
        'text_muted': '#999999',
        'error': '#f44336',
        'warning': '#ff9800',
        'info': '#2196f3',
        'success': '#4caf50',
        'border': '#404040',
        'accent': '#007acc',
    }

    # Severity colors and icons
    SEVERITY_CONFIG = {
        ErrorSeverity.LOW: {'color': COLORS['info'], 'icon': 'ℹ'},
        ErrorSeverity.MEDIUM: {'color': COLORS['warning'], 'icon': '⚠'},
        ErrorSeverity.HIGH: {'color': COLORS['error'], 'icon': '⚠'},
        ErrorSeverity.CRITICAL: {'color': COLORS['error'], 'icon': '🛑'},
    }

    def __init__(self, parent: tk.Tk, error_details: ErrorDetails):
        """Initialize error recovery dialog."""
        self.parent = parent
        self.error_details = error_details
        self.result = None
        self.callback_result = None

        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Error: {error_details.title}")
        self.dialog.geometry("600x500")
        self.dialog.configure(bg=self.COLORS['bg_primary'])
        self.dialog.resizable(True, True)

        # Center on parent
        self._center_on_parent()

        # Make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)

        # Build UI
        self._build_ui()

        # Focus dialog
        self.dialog.focus_set()

    def _center_on_parent(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()

        # Get parent geometry
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Calculate center position
        dialog_width = 600
        dialog_height = 500
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2

        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

    def _build_ui(self):
        """Build the error recovery dialog UI."""
        # Main container
        main_frame = tk.Frame(self.dialog, bg=self.COLORS['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Header section
        self._build_header(main_frame)

        # Content notebook
        self._build_content_notebook(main_frame)

        # Action buttons
        self._build_action_buttons(main_frame)

    def _build_header(self, parent):
        """Build header section with error icon and title."""
        header_frame = tk.Frame(parent, bg=self.COLORS['bg_primary'])
        header_frame.pack(fill='x', pady=(0, 20))

        # Error icon
        severity_config = self.SEVERITY_CONFIG[self.error_details.severity]
        icon_label = tk.Label(
            header_frame,
            text=severity_config['icon'],
            font=('Segoe UI', 24),
            fg=severity_config['color'],
            bg=self.COLORS['bg_primary']
        )
        icon_label.pack(side='left', padx=(0, 15))

        # Title and category
        title_frame = tk.Frame(header_frame, bg=self.COLORS['bg_primary'])
        title_frame.pack(side='left', fill='x', expand=True)

        # Error title
        title_label = tk.Label(
            title_frame,
            text=self.error_details.title,
            font=('Segoe UI', 14, 'bold'),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_primary'],
            anchor='w'
        )
        title_label.pack(fill='x')

        # Category and error code
        info_text = self.error_details.category
        if self.error_details.error_code:
            info_text += f" • Code: {self.error_details.error_code}"

        category_label = tk.Label(
            title_frame,
            text=info_text,
            font=('Segoe UI', 9),
            fg=self.COLORS['text_muted'],
            bg=self.COLORS['bg_primary'],
            anchor='w'
        )
        category_label.pack(fill='x', pady=(2, 0))

    def _build_content_notebook(self, parent):
        """Build content notebook with different tabs."""
        # Create notebook
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=self.COLORS['bg_primary'])
        style.configure('TNotebook.Tab', background=self.COLORS['bg_secondary'],
                       foreground=self.COLORS['text_secondary'])

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True, pady=(0, 20))

        # Overview tab
        self._build_overview_tab()

        # Solutions tab
        self._build_solutions_tab()

        # Technical details tab (if available)
        if self.error_details.technical_details or self.error_details.log_info:
            self._build_technical_tab()

        # Help tab
        self._build_help_tab()

    def _build_overview_tab(self):
        """Build overview tab with error description."""
        overview_frame = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        self.notebook.add(overview_frame, text="Overview")

        # Scrollable content
        canvas = tk.Canvas(overview_frame, bg=self.COLORS['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(overview_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg_primary'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Error message
        message_label = tk.Label(
            scrollable_frame,
            text=self.error_details.message,
            font=('Segoe UI', 10),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_primary'],
            anchor='w',
            justify='left',
            wraplength=520
        )
        message_label.pack(fill='x', pady=(0, 20))

        # Suggestions section
        if self.error_details.suggestions:
            suggestions_title = tk.Label(
                scrollable_frame,
                text="Suggested Solutions:",
                font=('Segoe UI', 11, 'bold'),
                fg=self.COLORS['text_primary'],
                bg=self.COLORS['bg_primary'],
                anchor='w'
            )
            suggestions_title.pack(fill='x', pady=(0, 10))

            for i, suggestion in enumerate(self.error_details.suggestions, 1):
                suggestion_frame = tk.Frame(scrollable_frame, bg=self.COLORS['bg_primary'])
                suggestion_frame.pack(fill='x', pady=2)

                bullet_label = tk.Label(
                    suggestion_frame,
                    text=f"{i}.",
                    font=('Segoe UI', 9),
                    fg=self.COLORS['text_secondary'],
                    bg=self.COLORS['bg_primary'],
                    width=3,
                    anchor='w'
                )
                bullet_label.pack(side='left')

                suggestion_label = tk.Label(
                    suggestion_frame,
                    text=suggestion,
                    font=('Segoe UI', 9),
                    fg=self.COLORS['text_secondary'],
                    bg=self.COLORS['bg_primary'],
                    anchor='w',
                    justify='left',
                    wraplength=480
                )
                suggestion_label.pack(side='left', fill='x', expand=True)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_solutions_tab(self):
        """Build solutions tab with actionable buttons."""
        solutions_frame = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        self.notebook.add(solutions_frame, text="Solutions")

        if not self.error_details.actions:
            # No actions available
            no_actions_label = tk.Label(
                solutions_frame,
                text="No automated solutions available.\nPlease refer to the Overview and Help tabs for manual solutions.",
                font=('Segoe UI', 10),
                fg=self.COLORS['text_muted'],
                bg=self.COLORS['bg_primary'],
                justify='center'
            )
            no_actions_label.pack(expand=True)
            return

        # Actions container
        actions_container = tk.Frame(solutions_frame, bg=self.COLORS['bg_primary'])
        actions_container.pack(fill='both', expand=True, padx=20, pady=20)

        # Instructions
        instructions_label = tk.Label(
            actions_container,
            text="Select an action to resolve this error:",
            font=('Segoe UI', 10),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_primary']
        )
        instructions_label.pack(pady=(0, 15))

        # Action buttons
        for action in self.error_details.actions:
            self._create_action_button(actions_container, action)

    def _create_action_button(self, parent, action: RecoveryAction):
        """Create an action button."""
        # Action frame
        action_frame = tk.Frame(parent, bg=self.COLORS['bg_secondary'], relief='raised', bd=1)
        action_frame.pack(fill='x', pady=5)

        # Make frame clickable
        action_frame.bind('<Button-1>', lambda e: self._execute_action(action))
        action_frame.bind('<Enter>', lambda e: action_frame.configure(bg=self.COLORS['bg_tertiary']))
        action_frame.bind('<Leave>', lambda e: action_frame.configure(bg=self.COLORS['bg_secondary']))

        # Content frame
        content_frame = tk.Frame(action_frame, bg=self.COLORS['bg_secondary'])
        content_frame.pack(fill='x', padx=15, pady=10)

        # Icon and label
        header_frame = tk.Frame(content_frame, bg=self.COLORS['bg_secondary'])
        header_frame.pack(fill='x')

        # Icon
        if action.icon:
            icon_label = tk.Label(
                header_frame,
                text=action.icon,
                font=('Segoe UI', 12),
                fg=self.COLORS['accent'] if action.primary else self.COLORS['text_secondary'],
                bg=self.COLORS['bg_secondary']
            )
            icon_label.pack(side='left', padx=(0, 10))

        # Action label
        label_text = action.label
        if action.primary:
            label_text += " (Recommended)"

        action_label = tk.Label(
            header_frame,
            text=label_text,
            font=('Segoe UI', 10, 'bold' if action.primary else 'normal'),
            fg=self.COLORS['accent'] if action.primary else self.COLORS['text_primary'],
            bg=self.COLORS['bg_secondary'],
            anchor='w'
        )
        action_label.pack(side='left', fill='x', expand=True)

        # Description
        if action.description:
            desc_label = tk.Label(
                content_frame,
                text=action.description,
                font=('Segoe UI', 9),
                fg=self.COLORS['text_secondary'],
                bg=self.COLORS['bg_secondary'],
                anchor='w',
                justify='left',
                wraplength=500
            )
            desc_label.pack(fill='x', pady=(5, 0))

        # Make all child widgets clickable
        for widget in [content_frame, header_frame, action_label]:
            widget.bind('<Button-1>', lambda e: self._execute_action(action))
            widget.bind('<Enter>', lambda e: action_frame.configure(bg=self.COLORS['bg_tertiary']))
            widget.bind('<Leave>', lambda e: action_frame.configure(bg=self.COLORS['bg_secondary']))

    def _build_technical_tab(self):
        """Build technical details tab."""
        technical_frame = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        self.notebook.add(technical_frame, text="Technical Details")

        # Scrollable text area
        text_frame = tk.Frame(technical_frame, bg=self.COLORS['bg_primary'])
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Text widget with scrollbar
        text_widget = tk.Text(
            text_frame,
            bg=self.COLORS['bg_tertiary'],
            fg=self.COLORS['text_secondary'],
            font=('Consolas', 9),
            wrap='word',
            state='normal'
        )

        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        # Add content
        if self.error_details.technical_details:
            text_widget.insert('end', "Technical Details:\n")
            text_widget.insert('end', "=" * 50 + "\n\n")
            text_widget.insert('end', self.error_details.technical_details + "\n\n")

        if self.error_details.log_info:
            text_widget.insert('end', "Log Information:\n")
            text_widget.insert('end', "=" * 50 + "\n\n")
            text_widget.insert('end', self.error_details.log_info + "\n\n")

        # Make read-only
        text_widget.configure(state='disabled')

        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Copy button
        copy_button = tk.Button(
            technical_frame,
            text="Copy to Clipboard",
            command=lambda: self._copy_technical_details(text_widget),
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_primary'],
            relief='flat',
            padx=15,
            pady=5
        )
        copy_button.pack(pady=10)

    def _build_help_tab(self):
        """Build help tab with documentation links."""
        help_frame = tk.Frame(self.notebook, bg=self.COLORS['bg_primary'])
        self.notebook.add(help_frame, text="Help")

        # Help content
        content_frame = tk.Frame(help_frame, bg=self.COLORS['bg_primary'])
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Documentation link
        if self.error_details.help_url:
            doc_title = tk.Label(
                content_frame,
                text="Documentation:",
                font=('Segoe UI', 11, 'bold'),
                fg=self.COLORS['text_primary'],
                bg=self.COLORS['bg_primary']
            )
            doc_title.pack(anchor='w', pady=(0, 5))

            doc_link = tk.Label(
                content_frame,
                text=self.error_details.help_url,
                font=('Segoe UI', 9, 'underline'),
                fg=self.COLORS['accent'],
                bg=self.COLORS['bg_primary'],
                cursor='hand2'
            )
            doc_link.pack(anchor='w', pady=(0, 20))
            doc_link.bind('<Button-1>', lambda e: webbrowser.open(self.error_details.help_url))

        # Common solutions
        common_title = tk.Label(
            content_frame,
            text="Common Solutions:",
            font=('Segoe UI', 11, 'bold'),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_primary']
        )
        common_title.pack(anchor='w', pady=(0, 5))

        common_solutions = [
            "Check that all required files and dependencies are present",
            "Verify that the application has necessary permissions",
            "Restart the application and try again",
            "Check system requirements and available resources",
            "Review log files for additional error information",
            "Contact support if the problem persists"
        ]

        for solution in common_solutions:
            solution_label = tk.Label(
                content_frame,
                text=f"• {solution}",
                font=('Segoe UI', 9),
                fg=self.COLORS['text_secondary'],
                bg=self.COLORS['bg_primary'],
                anchor='w',
                justify='left',
                wraplength=500
            )
            solution_label.pack(fill='x', pady=1)

        # Contact support
        support_frame = tk.Frame(content_frame, bg=self.COLORS['bg_secondary'])
        support_frame.pack(fill='x', pady=(20, 0))

        support_title = tk.Label(
            support_frame,
            text="Need Additional Help?",
            font=('Segoe UI', 10, 'bold'),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_secondary']
        )
        support_title.pack(pady=10)

        support_text = tk.Label(
            support_frame,
            text="If you continue to experience issues, please contact support with the technical details from this dialog.",
            font=('Segoe UI', 9),
            fg=self.COLORS['text_secondary'],
            bg=self.COLORS['bg_secondary'],
            wraplength=500,
            justify='center'
        )
        support_text.pack(pady=(0, 10))

    def _build_action_buttons(self, parent):
        """Build action buttons at bottom of dialog."""
        button_frame = tk.Frame(parent, bg=self.COLORS['bg_primary'])
        button_frame.pack(fill='x', side='bottom')

        # Close button
        close_button = tk.Button(
            button_frame,
            text="Close",
            command=self._on_close,
            bg=self.COLORS['bg_secondary'],
            fg=self.COLORS['text_primary'],
            relief='flat',
            padx=20,
            pady=8,
            font=('Segoe UI', 9)
        )
        close_button.pack(side='right')

        # Primary action button (if available)
        primary_action = next((a for a in self.error_details.actions if a.primary), None)
        if primary_action:
            primary_button = tk.Button(
                button_frame,
                text=primary_action.label,
                command=lambda: self._execute_action(primary_action),
                bg=self.COLORS['accent'],
                fg=self.COLORS['text_primary'],
                relief='flat',
                padx=20,
                pady=8,
                font=('Segoe UI', 9, 'bold')
            )
            primary_button.pack(side='right', padx=(0, 10))

    def _execute_action(self, action: RecoveryAction):
        """Execute a recovery action."""
        try:
            if action.callback:
                # Custom callback
                self.callback_result = action.callback(action.data)
                self.result = action.type
                self.dialog.destroy()
            elif action.type == ActionType.RETRY:
                self.result = ActionType.RETRY
                self.dialog.destroy()
            elif action.type == ActionType.USE_DEFAULTS:
                self.result = ActionType.USE_DEFAULTS
                self.dialog.destroy()
            elif action.type == ActionType.OPEN_URL and action.data:
                webbrowser.open(action.data)
            elif action.type == ActionType.OPEN_FILE and action.data:
                if os.path.exists(action.data):
                    os.startfile(action.data)
                else:
                    messagebox.showerror("Error", f"File not found: {action.data}")
            elif action.type == ActionType.RUN_COMMAND and action.data:
                os.system(action.data)
            elif action.type == ActionType.IGNORE:
                self.result = ActionType.IGNORE
                self.dialog.destroy()
            elif action.type == ActionType.CANCEL:
                self.result = ActionType.CANCEL
                self.dialog.destroy()
            else:
                messagebox.showinfo("Info", f"Action not implemented: {action.label}")

        except Exception as e:
            messagebox.showerror("Action Error", f"Failed to execute action: {str(e)}")

    def _copy_technical_details(self, text_widget):
        """Copy technical details to clipboard."""
        try:
            content = text_widget.get('1.0', 'end-1c')
            self.dialog.clipboard_clear()
            self.dialog.clipboard_append(content)
            messagebox.showinfo("Copied", "Technical details copied to clipboard.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy to clipboard: {str(e)}")

    def _on_close(self):
        """Handle dialog close."""
        self.result = ActionType.CANCEL
        self.dialog.grab_release()
        self.dialog.destroy()

    def show(self) -> ActionType:
        """Show dialog and return result."""
        self.dialog.wait_window()
        return self.result


def show_error_dialog(parent: tk.Tk, error_details: ErrorDetails) -> ActionType:
    """Convenience function to show error dialog."""
    dialog = ErrorRecoveryDialog(parent, error_details)
    return dialog.show()


def create_settings_error(title: str, message: str, suggestions: List[str] = None,
                         retry_callback: Callable = None, defaults_callback: Callable = None) -> ErrorDetails:
    """Create a settings-related error with common actions."""
    actions = []

    if retry_callback:
        actions.append(RecoveryAction(
            type=ActionType.RETRY,
            label="Retry",
            description="Try applying the settings again",
            callback=retry_callback,
            primary=True,
            icon="🔄"
        ))

    if defaults_callback:
        actions.append(RecoveryAction(
            type=ActionType.USE_DEFAULTS,
            label="Use Default Settings",
            description="Reset to default configuration and apply",
            callback=defaults_callback,
            icon="⚙"
        ))

    actions.append(RecoveryAction(
        type=ActionType.CANCEL,
        label="Cancel",
        description="Close dialog without making changes",
        icon="✕"
    ))

    return ErrorDetails(
        title=title,
        message=message,
        category="Settings",
        severity=ErrorSeverity.MEDIUM,
        suggestions=suggestions or [],
        actions=actions
    )


def create_webcam_error(title: str, message: str, technical_details: str = None) -> ErrorDetails:
    """Create a webcam-related error with common solutions."""
    suggestions = [
        "Check that the camera is properly connected",
        "Ensure no other applications are using the camera",
        "Try a different camera resolution or format",
        "Update camera drivers",
        "Restart the application"
    ]

    actions = [
        RecoveryAction(
            type=ActionType.RETRY,
            label="Try Again",
            description="Retry camera initialization",
            primary=True,
            icon="📹"
        ),
        RecoveryAction(
            type=ActionType.OPEN_SETTINGS,
            label="Open Camera Settings",
            description="Configure camera settings manually",
            icon="⚙"
        ),
        RecoveryAction(
            type=ActionType.CANCEL,
            label="Continue Without Camera",
            description="Use application in demo mode",
            icon="◯"
        )
    ]

    return ErrorDetails(
        title=title,
        message=message,
        technical_details=technical_details,
        category="Camera",
        severity=ErrorSeverity.HIGH,
        suggestions=suggestions,
        actions=actions
    )


def create_file_error(title: str, message: str, file_path: str = None) -> ErrorDetails:
    """Create a file-related error with common solutions."""
    suggestions = [
        "Check that the file exists and is accessible",
        "Verify file permissions",
        "Ensure the file is not open in another application",
        "Check available disk space"
    ]

    actions = [
        RecoveryAction(
            type=ActionType.RETRY,
            label="Try Again",
            description="Retry the file operation",
            primary=True,
            icon="📁"
        )
    ]

    if file_path:
        # Add action to open file location
        folder_path = os.path.dirname(file_path) if os.path.isfile(file_path) else file_path
        actions.append(RecoveryAction(
            type=ActionType.OPEN_FILE,
            label="Open File Location",
            description="Open the file location in explorer",
            data=folder_path,
            icon="📂"
        ))

    actions.append(RecoveryAction(
        type=ActionType.CANCEL,
        label="Cancel",
        description="Cancel the operation",
        icon="✕"
    ))

    return ErrorDetails(
        title=title,
        message=message,
        category="File System",
        severity=ErrorSeverity.MEDIUM,
        suggestions=suggestions,
        actions=actions
    )