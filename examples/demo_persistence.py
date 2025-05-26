#!/usr/bin/env python3

import subprocess
import time
import sys

def run_session(session_num, commands):
    """Run a session with given commands"""
    print(f"\n{'='*60}")
    print(f"SESSION {session_num}")
    print('='*60)
    
    proc = subprocess.Popen(
        ['python3', 'launch_hermes_fixed.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0
    )
    
    output = []
    
    for cmd in commands:
        print(f"\n>>> Sending: {cmd}")
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
        
        # Wait and collect output
        time.sleep(3)
        
    proc.stdin.write("/exit\n")
    proc.stdin.flush()
    
    # Get all output
    full_output, _ = proc.communicate()
    
    # Print relevant parts
    lines = full_output.split('\n')
    in_chat = False
    for line in lines:
        if "Start chatting!" in line:
            in_chat = True
        if in_chat and line.strip():
            if not line.startswith('2025-') and 'INFO' not in line:
                print(line)

def main():
    print("MEMORY PERSISTENCE DEMONSTRATION")
    print("================================")
    
    # Session 1: Tell the AI some facts
    print("\nIn session 1, we'll tell the AI some facts about ourselves...")
    run_session(1, [
        "My name is Alice and I work as a data scientist",
        "I have two dogs named Max and Luna",
        "My favorite programming language is Python",
        "/memory"
    ])
    
    print("\n\n⏰ Waiting 5 seconds before starting new session...")
    time.sleep(5)
    
    # Session 2: Ask the AI to recall
    print("\nIn session 2, we'll see if the AI remembers...")
    run_session(2, [
        "What do you remember about me?",
        "What are my dogs' names?",
        "/memory"
    ])

if __name__ == "__main__":
    main()