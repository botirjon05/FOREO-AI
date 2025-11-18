#!/usr/bin/env python3
# ticket_management.py — Support ticket creation and management for Sprint 5

import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Ticket storage directory
TICKETS_DIR = Path("tickets")
TICKETS_DIR.mkdir(exist_ok=True)

# Counter file to track ticket numbers
COUNTER_FILE = TICKETS_DIR / ".ticket_counter.json"

def get_next_ticket_number() -> int:
    """Get the next ticket number from persistent counter"""
    if COUNTER_FILE.exists():
        try:
            with open(COUNTER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("counter", 0) + 1
        except Exception:
            return 1
    return 1

def save_ticket_counter(counter: int):
    """Save the current ticket counter"""
    try:
        with open(COUNTER_FILE, "w", encoding="utf-8") as f:
            json.dump({"counter": counter}, f)
    except Exception:
        pass  # Fail silently if can't save counter

def generate_ticket_id() -> str:
    """Generate a unique ticket ID with 6-digit incremental number (e.g., FOREO-000001)"""
    ticket_number = get_next_ticket_number()
    save_ticket_counter(ticket_number)
    return f"FOREO-{ticket_number:06d}"

def create_ticket(
    name: str,
    email: str,
    device: Optional[str] = None,
    issue: Optional[str] = None,
    chat_history: Optional[List[Dict]] = None,
    metadata: Optional[Dict] = None
) -> Dict:
    """
    Create a support ticket with all relevant information.
    
    Args:
        name: Customer name
        email: Customer email
        device: Device type (optional)
        issue: Issue description (optional)
        chat_history: List of chat messages (optional)
        metadata: Additional metadata (optional)
    
    Returns:
        Dictionary containing ticket information
    """
    ticket_id = generate_ticket_id()
    ticket = {
        "ticket_id": ticket_id,
        "created_at": datetime.now().isoformat(),
        "customer": {
            "name": name,
            "email": email
        },
        "issue_details": {
            "device": device,
            "issue": issue
        },
        "chat_history": chat_history or [],
        "metadata": metadata or {},
        "status": "open"
    }
    
    # Save to JSON file
    ticket_file = TICKETS_DIR / f"{ticket_id}.json"
    with open(ticket_file, "w", encoding="utf-8") as f:
        json.dump(ticket, f, indent=2, ensure_ascii=False)
    
    return ticket

def get_ticket(ticket_id: str) -> Optional[Dict]:
    """Retrieve a ticket by ID"""
    ticket_file = TICKETS_DIR / f"{ticket_id}.json"
    if ticket_file.exists():
        with open(ticket_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def list_all_tickets() -> List[Dict]:
    """List all tickets"""
    tickets = []
    for ticket_file in TICKETS_DIR.glob("*.json"):
        try:
            with open(ticket_file, "r", encoding="utf-8") as f:
                tickets.append(json.load(f))
        except Exception:
            continue
    return sorted(tickets, key=lambda x: x.get("created_at", ""), reverse=True)

