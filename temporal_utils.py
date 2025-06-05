"""
Temporal Utilities for Memory System
Provides current time/date awareness to the AI assistant
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any

def get_current_datetime_info() -> Dict[str, Any]:
    """
    Get comprehensive current date/time information for AI awareness
    Returns all the temporal context an AI assistant might need
    """
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now()
    
    return {
        "current_utc_iso": now_utc.isoformat(),
        "current_local_iso": now_local.isoformat(),
        "unix_timestamp": time.time(),
        "formatted_date": now_local.strftime("%A, %B %d, %Y"),
        "formatted_time": now_local.strftime("%I:%M %p"),
        "formatted_datetime": now_local.strftime("%A, %B %d, %Y at %I:%M %p"),
        "timezone_name": str(now_local.astimezone().tzinfo),
        "day_of_week": now_local.strftime("%A"),
        "month_name": now_local.strftime("%B"),
        "year": now_local.year,
        "is_weekend": now_local.weekday() >= 5,
        "season": get_season(now_local.month, now_local.day)
    }

def get_season(month: int, day: int) -> str:
    """Determine the season based on month and day"""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        if month == 3 and day < 20:
            return "Winter"
        return "Spring"
    elif month in [6, 7, 8]:
        if month == 6 and day < 21:
            return "Spring"
        return "Summer"
    else:  # month in [9, 10, 11]
        if month == 9 and day < 22:
            return "Summer"
        elif month == 12 and day >= 21:
            return "Winter"
        return "Fall"

def create_temporal_context_prompt() -> str:
    """
    Create a temporal context string to inject into AI prompts
    This makes the AI aware of the current date and time
    """
    dt_info = get_current_datetime_info()
    
    context = f"""[CURRENT TIME CONTEXT]
Today is {dt_info['formatted_date']} and the current time is {dt_info['formatted_time']}.
It is {dt_info['day_of_week']} in {dt_info['season']}.
You have access to real-time information and should provide current date/time when asked.
[END TIME CONTEXT]

"""
    return context

def inject_temporal_awareness(user_message: str) -> str:
    """
    Inject temporal context into user messages for AI processing
    This ensures the AI knows the current time for every interaction
    """
    temporal_context = create_temporal_context_prompt()
    return temporal_context + user_message

def format_timestamp_for_memory(timestamp: float = None) -> str:
    """
    Format a timestamp for memory storage (human readable)
    """
    if timestamp is None:
        timestamp = time.time()
    
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def is_recent_timestamp(timestamp: float, hours_threshold: int = 24) -> bool:
    """
    Check if a timestamp is recent (within specified hours)
    """
    current_time = time.time()
    return (current_time - timestamp) <= (hours_threshold * 3600)

# Example usage and testing
if __name__ == "__main__":
    # Test the temporal utilities
    print("=== Temporal Utils Test ===")
    
    dt_info = get_current_datetime_info()
    print(f"Current DateTime Info:")
    for key, value in dt_info.items():
        print(f"  {key}: {value}")
    
    print(f"\nTemporal Context Prompt:")
    print(create_temporal_context_prompt())
    
    print(f"Sample Injected Message:")
    sample_msg = "What time is it?"
    injected = inject_temporal_awareness(sample_msg)
    print(injected)