# Basic ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"

# Foreground colors
BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

# Helper functions
def color(text, code): 
    return f"{code}{text}{RESET}"

def red(text): 
    return color(text, RED)

def green(text): 
    return color(text, GREEN)

def yellow(text): 
    return color(text, YELLOW)

def blue(text): 
    return color(text, BLUE)

def magenta(text): 
    return color(text, MAGENTA)

def cyan(text): 
    return color(text, CYAN)

def bold(text): 
    return color(text, BOLD)

def dim(text): 
    return color(text, DIM)

def underline(text): 
    return color(text, UNDERLINE)
