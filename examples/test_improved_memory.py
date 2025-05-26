#!/usr/bin/env python3

import subprocess
import time

def test_session():
    """Run an interactive test session"""
    print("Testing improved memory system...")
    print("=" * 60)
    
    # Clear existing memories first
    proc = subprocess.Popen(
        ['python3', 'launch_hermes_fixed.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    proc.stdin.write("/clear\n")
    proc.stdin.flush()
    time.sleep(2)
    proc.stdin.write("/exit\n")
    proc.stdin.flush()
    proc.communicate()
    
    print("Memories cleared. Starting test...")
    
    # Start new session
    proc = subprocess.Popen(
        ['python3', 'launch_hermes_fixed.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0
    )
    
    # Test conversation
    test_inputs = [
        ("Hello, what is your name?", 3),
        ("I am Adam. Who are you?", 3),
        ("My brother is Bob", 3),
        ("My favorite color is grey", 3),
        ("Bob's favorite color is blue", 3),
        ("Gary's my cousin, his favorite color is green.", 3),
        ("What is my brother's favorite color?", 5),
        ("Who is Bob?", 3),
        ("What is my name and favorite color?", 5),
        ("/memory", 3),
        ("/exit", 1)
    ]
    
    output_buffer = []
    
    for user_input, wait_time in test_inputs:
        print(f"\n>>> {user_input}")
        proc.stdin.write(user_input + "\n")
        proc.stdin.flush()
        time.sleep(wait_time)
    
    # Get output
    output, _ = proc.communicate()
    
    # Parse and display relevant output
    lines = output.split('\n')
    in_chat = False
    for line in lines:
        if "Start chatting!" in line:
            in_chat = True
        if in_chat and line.strip():
            # Skip log lines
            if not line.startswith('2025-') and 'INFO' not in line and 'Batches:' not in line:
                # Clean up the line
                if line.startswith('You: ') or line.startswith('Assistant: '):
                    print(line)

if __name__ == "__main__":
    test_session()