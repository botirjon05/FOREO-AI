#!/usr/bin/env python3
# troubleshooting.py — Simple troubleshooting responses

from typing import Dict, List

def get_troubleshooting_steps(slots: Dict) -> str:
    """Get troubleshooting steps based on slots"""
    issue = slots.get("issue", "").lower()
    device_type = slots.get("device_type", "device")
    
    if "charge" in issue:
        return f"""Here's how to troubleshoot charging issues with your {device_type}:

1. Check that the charger is properly connected to both the device and wall socket
2. If the battery was completely drained, wait up to 5 minutes for the LED to start pulsing
3. Try plugging the DC connector more firmly into the device's charging port
4. Check that the main plug is securely plugged into the wall socket
5. Only use the original FOREO charger - other chargers may damage your battery

If the device still won't charge after these steps, please contact support for warranty service."""

    elif "turn" in issue or "start" in issue or "on" in issue:
        return f"""Here's how to troubleshoot power issues with your {device_type}:

1. Ensure the device has been charged for at least 1 hour
2. Check if the travel lock is enabled (for 3-button devices)
3. To unlock: Hold the + and - buttons simultaneously for 5 seconds
4. When unlocked, the LED will light up to confirm
5. Try pressing the power button firmly once

If your device still won't turn on, the device may be faulty. Please submit a warranty claim on foreo.com."""
    
    elif "clean" in issue:
        return f"""Here's how to clean your {device_type}:

1. Rinse the device thoroughly with warm water after each use
2. Use FOREO Silicone Cleaning Spray for thorough sanitization
3. Wash brush surfaces with mild soap and water
4. Pat dry with a lint-free cloth or let air dry completely
5. Ensure all surfaces are completely dry before charging

Keep your device clean for optimal performance!"""
    
    else:
        return f"""Here are general troubleshooting steps for your {device_type}:

1. Fully charge the device for 1 hour
2. Clean the device according to instructions
3. Check for any visible damage to the device
4. Ensure you're following the proper operating instructions
5. If issues persist, please contact customer support at foreo.com/support"""
