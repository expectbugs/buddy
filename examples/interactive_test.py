#!/usr/bin/env python3
"""
Interactive test with automated inputs to simulate real usage
"""

import subprocess
import time
import os

# Test inputs to send
test_inputs = [
    "Hello! My name is Bob and I'm a data scientist.",
    "I specialize in deep learning and computer vision.",
    "What's my name?",
    "What do I specialize in?",
    "I'm working on a project to detect objects in satellite imagery.",
    "Tell me everything you know about me.",
    "/memory",
    "My favorite programming language is Python and I use PyTorch.",
    "What frameworks do I use?",
    "/exit"
]

# Create a script that feeds inputs
script_content = """
import time
inputs = %s

for inp in inputs:
    print(inp)
    time.sleep(2)  # Wait between inputs
""" % repr(test_inputs)

with open('/tmp/test_inputs.py', 'w') as f:
    f.write(script_content)

# Run the main program with piped inputs
print("🧪 Starting interactive test...")
print("="*60)

try:
    # Run the command with input from our script
    cmd = f"cd /home/user/buddy && python3 /tmp/test_inputs.py | python3 launch_hermes_fixed.py"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
        
except subprocess.TimeoutExpired:
    print("Test timed out after 120 seconds")
except Exception as e:
    print(f"Error: {e}")

# Cleanup
if os.path.exists('/tmp/test_inputs.py'):
    os.remove('/tmp/test_inputs.py')