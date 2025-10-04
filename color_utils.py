# color_utils.py
# Simple color utilities for selective logging enhancement

class Colors:
    """ANSI color codes for terminal output."""
    BLUE = '\033[94m'      # Blue for phase headers and key actions
    GREEN = '\033[92m'     # Green for success/completion messages
    RESET = '\033[0m'      # Reset to default color
    
def blue(text: str) -> str:
    """Highlight text in blue for phase headers and key actions."""
    return f"{Colors.BLUE}{text}{Colors.RESET}"

def green(text: str) -> str:
    """Highlight text in green for success/completion messages."""
    return f"{Colors.GREEN}{text}{Colors.RESET}"

def phase_header(phase_num: int, action: str) -> str:
    """Create a colored phase header."""
    return blue(f"[PHASE {phase_num}] {action}")

def engine_header(action: str) -> str:
    """Create a colored engine header."""
    return blue(f"[MEVE ENGINE] {action}")

def success_message(message: str) -> str:
    """Create a colored success message."""
    return green(message)