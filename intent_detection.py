#!/usr/bin/env python3
# intent_detection.py — Intent classification for FOREO chatbot with region support

import re
from typing import Dict, List, Tuple, Optional
import difflib

def classify_intent(query: str) -> Tuple[str, float]:
    """
    Simple keyword-based intent classification.
    Returns: (intent_name, confidence)
    """
    q_lower = query.lower()
    
    # Check for troubleshooting keywords first (highest priority)
    if any(kw in q_lower for kw in ["won't", "wont", "not working", "broken", "problem", "issue", "trouble", "not charge", "won't charge", "won't turn", "won't start"]):
        return ("troubleshooting", 0.9)
    
    # Warranty questions
    if any(kw in q_lower for kw in ["warranty", "guarantee", "claim", "defect"]):
        return ("warranty", 0.85)
    
    # Cleaning/care
    if any(kw in q_lower for kw in ["clean", "cleaning", "wash", "maintain", "care"]):
        return ("cleaning", 0.85)
    
    # Charging
    if any(kw in q_lower for kw in ["charge", "charging", "battery", "power", "usb"]):
        return ("charging", 0.85)
    
    # Account help
    if any(kw in q_lower for kw in ["account", "login", "password", "register", "sign"]):
        return ("account", 0.8)
    
    # Orders/shipping
    if any(kw in q_lower for kw in ["order", "shipping", "delivery", "track", "cancel", "return", "refund"]):
        return ("orders", 0.8)
    
    # Product inquiry
    if any(kw in q_lower for kw in ["which", "what", "difference", "compare", "recommend"]):
        return ("products", 0.7)
    
    # Default to general
    return ("general", 0.5)

def extract_device_type(query: str) -> Optional[str]:
    """Extract device type from query"""
    devices = ["luna", "bear", "ufo", "issa", "espa", "iris", "kiwi", "peach", "faq"]
    query_lower = query.lower().strip()
    
    # Check if the entire query is a device name (common case in clarification)
    if query_lower in devices:
        return query_lower.upper()
    
    # Check if any device name appears in the query
    for device in devices:
        if device in query_lower:
            # Check for version numbers
            numbers = re.search(r'\b\d+', query_lower)
            if numbers:
                return f"{device.upper()} {numbers.group()}"
            return device.upper()
    
    return None

def extract_country(query: str) -> Optional[str]:
    """Extract country from query"""
    # Common countries where FOREO operates
    countries = {
        "usa": "United States",
        "united states": "United States",
        "uk": "United Kingdom",
        "united kingdom": "United Kingdom",
        "canada": "Canada",
        "australia": "Australia",
        "sweden": "Sweden",
        "germany": "Germany",
        "france": "France",
        "italy": "Italy",
        "spain": "Spain",
        "netherlands": "Netherlands",
        "belgium": "Belgium",
        "switzerland": "Switzerland",
        "austria": "Austria",
        "japan": "Japan",
        "china": "China",
        "south korea": "South Korea",
        "singapore": "Singapore",
        "turkey": "Turkey",
        "mexico": "Mexico",
        "brazil": "Brazil",
        "argentina": "Argentina",
        "india": "India",
        "indonesia": "Indonesia",
        "philippines": "Philippines",
        "thailand": "Thailand",
        "malaysia": "Malaysia",
        "vietnam": "Vietnam",
        "poland": "Poland",
        "portugal": "Portugal",
        "romania": "Romania",
        "hungary": "Hungary",
        "czech": "Czech Republic",
        "czech republic": "Czech Republic",
        "slovakia": "Slovakia",
        "greece": "Greece",
        "ireland": "Ireland",
        "denmark": "Denmark",
        "norway": "Norway",
        "finland": "Finland",
        "iceland": "Iceland",
        "russia": "Russia",
        "kazakhstan": "Kazakhstan",
        "uzbekistan": "Uzbekistan",
    }
    
    q_lower = query.lower().strip()
    
    # First, check for exact match (common in clarification flow)
    if q_lower in countries:
        return countries[q_lower]
    
    # Then check for partial match
    for key, value in countries.items():
        if key in q_lower:
            return value

    # Fuzzy match against known country keys
    candidates = difflib.get_close_matches(q_lower, list(countries.keys()), n=1, cutoff=0.85)
    if candidates:
        return countries[candidates[0]]
    
    return None

def extract_region(query: str) -> Optional[str]:
    """Extract broader regions when country isn't recognized."""
    q_lower = query.lower()
    region_aliases = {
        "eu": "European Union",
        "europe": "European Union",
        "european union": "European Union",
        "united kingdom": "United Kingdom",
        "uk": "United Kingdom",
        "us": "United States",
        "usa": "United States",
        "international": "International",
        "rest of world": "International",
        "worldwide": "International",
    }
    # Exact
    if q_lower in region_aliases:
        return region_aliases[q_lower]
    # Partial
    for key, value in region_aliases.items():
        if key in q_lower:
            return value
    return None

def simple_extract_slots(query: str) -> Dict[str, str]:
    """Extract slots directly from query"""
    slots = {}
    
    # Extract device type
    device = extract_device_type(query)
    if device:
        slots["device_type"] = device
    
    # Extract country/region
    country = extract_country(query)
    if country:
        slots["country"] = country
    else:
        region = extract_region(query)
        if region:
            slots["region"] = region
    
    # Extract issue type
    q_lower = query.lower()
    if any(kw in q_lower for kw in ["charge", "charging", "battery", "power"]):
        slots["issue"] = "charging"
    elif any(kw in q_lower for kw in ["turn on", "won't turn", "wont turn", "start", "power on"]):
        slots["issue"] = "not_turning_on"
    elif any(kw in q_lower for kw in ["clean", "cleaning", "wash"]):
        slots["issue"] = "cleaning"
    elif any(kw in q_lower for kw in ["button", "buttons"]):
        slots["issue"] = "buttons"
    elif any(kw in q_lower for kw in ["weak", "slow", "performance"]):
        slots["issue"] = "performance"
    
    return slots

def needs_clarification(intent: str, slots: Dict) -> Tuple[bool, str]:
    """
    Check if clarification is needed and return the question to ask.
    Returns: (needs_clarification, clarification_question)
    
    Only device-specific or region-specific intents require clarification.
    """
    if intent == "troubleshooting":
        if not slots.get("device_type"):
            return True, "Which FOREO device are you having issues with? (e.g., LUNA 4, BEAR, UFO)"
        if not slots.get("issue"):
            return True, "What specific issue are you experiencing?"
    
    elif intent == "cleaning":
        if not slots.get("device_type"):
            return True, "Which FOREO device would you like cleaning instructions for?"
    
    elif intent == "charging":
        if not slots.get("device_type"):
            return True, "Which FOREO device are you asking about?"
    
    elif intent == "warranty":
        # Warranty may vary by region
        if not (slots.get("country") or slots.get("region")):
            return True, "Which country are you located in?"
    
    elif intent == "orders":
        # Orders are country/region-specific
        if not (slots.get("country") or slots.get("region")):
            return True, "Which country are you located in?"
    
    # Account, products, and general intents usually don't need clarification
    
    return False, ""
