#!/usr/bin/env python3

"""
Test the Hermes system with real user interactions
Simulates a conversation to verify memory functionality
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from launch_hermes_fixed import LocalMemorySystem, load_config, check_services

def simulate_conversation():
    """Simulate a real user conversation"""
    
    print("🧪 Testing Hermes with Local Memory System")
    print("="*60)
    
    # Check services first
    if not check_services():
        print("Services not running! Please start them first.")
        return
    
    # Load config and initialize system
    config = load_config()
    system = LocalMemorySystem(config)
    system.initialize()
    
    # Test conversations
    test_interactions = [
        ("Hi! My name is Alice and I'm a software developer.", "introduction"),
        ("I work with Python and JavaScript mainly.", "skills"),
        ("What's my name?", "recall_test_1"),
        ("I'm currently working on a machine learning project using TensorFlow.", "project"),
        ("What programming languages do I use?", "recall_test_2"),
        ("Tell me about my current work based on what you know about me.", "comprehensive_recall"),
        ("/memory", "view_memories"),
        ("I also enjoy hiking on weekends.", "hobby"),
        ("What do you remember about me?", "final_recall")
    ]
    
    print("\n🎭 Starting simulated conversation...\n")
    
    for user_input, test_type in test_interactions:
        print(f"You: {user_input}")
        
        if user_input == "/memory":
            # Handle memory command
            memories = system.get_all_memories()
            print(f"\n📚 Stored Memories ({len(memories)} total):")
            if memories:
                for i, mem in enumerate(memories, 1):
                    print(f"{i}. {mem['text']}")
            else:
                print("No memories stored yet.")
            print()
            continue
        
        # Store user input
        system.add_memory(f"User: {user_input}", "user_message")
        
        # Search for relevant memories
        relevant_memories = system.search_memories(user_input, limit=5)
        
        # Build context
        context = ""
        if relevant_memories:
            context = "Based on our previous conversations:\n"
            for mem in relevant_memories:
                if mem['text'] != f"User: {user_input}":
                    context += f"- {mem['text']}\n"
            context += "\n"
        
        # Construct prompt
        prompt = f"""{context}User: {user_input}
Assistant: """
        
        # Generate response
        try:
            response = system.llm(
                prompt,
                max_tokens=200,
                temperature=config["model"]["temperature"],
                top_p=config["model"]["top_p"],
                repeat_penalty=config["model"]["repeat_penalty"],
                stop=["User:", "\n\n"]
            )
            
            assistant_response = response['choices'][0]['text'].strip()
            print(f"Assistant: {assistant_response}\n")
            
            # Store assistant response
            system.add_memory(f"Assistant: {assistant_response}", "assistant_message")
            
            # Verify memory operations based on test type
            if test_type == "recall_test_1":
                if "alice" in assistant_response.lower():
                    print("✅ Name recall successful!")
                else:
                    print("❌ Failed to recall name")
            
            elif test_type == "recall_test_2":
                if "python" in assistant_response.lower() or "javascript" in assistant_response.lower():
                    print("✅ Skills recall successful!")
                else:
                    print("❌ Failed to recall programming languages")
            
            elif test_type == "comprehensive_recall":
                mentioned = []
                if "alice" in assistant_response.lower():
                    mentioned.append("name")
                if "python" in assistant_response.lower() or "javascript" in assistant_response.lower():
                    mentioned.append("languages")
                if "machine learning" in assistant_response.lower() or "tensorflow" in assistant_response.lower():
                    mentioned.append("project")
                
                if mentioned:
                    print(f"✅ Comprehensive recall successful! Mentioned: {', '.join(mentioned)}")
                else:
                    print("❌ Failed comprehensive recall")
            
        except Exception as e:
            print(f"❌ Error generating response: {e}\n")
        
        # Small delay between interactions
        time.sleep(0.5)
    
    # Final verification
    print("\n" + "="*60)
    print("📊 FINAL VERIFICATION")
    print("="*60)
    
    # Check total memories
    all_memories = system.get_all_memories()
    print(f"\nTotal memories stored: {len(all_memories)}")
    
    # Check memory types
    user_msgs = sum(1 for m in all_memories if "User:" in m['text'])
    assistant_msgs = sum(1 for m in all_memories if "Assistant:" in m['text'])
    
    print(f"User messages: {user_msgs}")
    print(f"Assistant messages: {assistant_msgs}")
    
    # Test final search
    print("\n🔍 Testing final memory search for 'Alice'...")
    search_results = system.search_memories("Alice software developer", limit=3)
    if search_results:
        print(f"Found {len(search_results)} relevant memories:")
        for i, result in enumerate(search_results, 1):
            print(f"  {i}. {result['text'][:60]}... (score: {result['score']:.3f})")
    else:
        print("❌ No search results found")
    
    print("\n✅ Test completed!")
    
    # Cleanup
    system.neo4j.close()

if __name__ == "__main__":
    simulate_conversation()