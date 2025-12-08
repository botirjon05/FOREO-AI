# db.py — database models and helpers for FOREO AI support tickets

from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# -------------------------
# Engine & Session
# -------------------------

# SQLite DB file in project root
DB_PATH = Path(__file__).parent / "foreo_support.db"
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()

# Central place to keep ticket statuses consistent across the app + portal
TICKET_STATUS_CHOICES = ("open", "in_progress", "resolved")

# -------------------------
# Models
# -------------------------

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String(50), unique=True, index=True)  # e.g. "FOREO-2024-0012"
    user_name = Column(String(120), nullable=True)
    user_email = Column(String(255), nullable=True)
    intent = Column(String(50), nullable=True)        # warranty / charging / cleaning / other
    device_type = Column(String(80), nullable=True)   # LUNA 4, UFO, etc.
    issue_summary = Column(String(255), nullable=True)
    status = Column(String(30), default="open")       # open / in_progress / resolved / closed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # full chat history as JSON string (we'll store your Streamlit messages array here)
    chat_history = Column(Text, nullable=False)

    # simple flags
    escalated_by_bot = Column(Boolean, default=True)
    source = Column(String(50), default="chatbot")    # chatbot, manual, etc.


class TicketNote(Base):
    __tablename__ = "ticket_notes"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String(50), index=True)         # store FOREO- style ID for easier lookups
    author = Column(String(120), nullable=True)        # optional agent identifier
    note = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# -------------------------
# Initialization helper
# -------------------------

def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

# -------------------------
# CRUD helpers
# -------------------------

def get_session():
    """Context-style session helper."""
    return SessionLocal()


def create_ticket(
    *,
    session,
    ticket_id: str,
    chat_history_json: str,
    user_name: Optional[str] = None,
    user_email: Optional[str] = None,
    intent: Optional[str] = None,
    device_type: Optional[str] = None,
    issue_summary: Optional[str] = None,
    escalated_by_bot: bool = True,
) -> Ticket:
    ticket = Ticket(
        ticket_id=ticket_id,
        chat_history=chat_history_json,
        user_name=user_name,
        user_email=user_email,
        intent=intent,
        device_type=device_type,
        issue_summary=issue_summary,
        escalated_by_bot=escalated_by_bot,
    )
    session.add(ticket)
    session.commit()
    session.refresh(ticket)
    return ticket


def list_tickets(session, status: Optional[str] = None):
    q = session.query(Ticket).order_by(Ticket.created_at.desc())
    if status:
        q = q.filter(Ticket.status == status)
    return q.all()


def get_ticket_by_id(session, ticket_id: str) -> Optional[Ticket]:
    return session.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()


def update_ticket_status(session, ticket_id: str, new_status: str) -> Optional[Ticket]:
    """Update ticket status; returns updated ticket or None if not found."""
    if new_status not in TICKET_STATUS_CHOICES:
        raise ValueError(f"Unsupported status '{new_status}'")

    ticket = get_ticket_by_id(session, ticket_id)
    if not ticket:
        return None

    ticket.status = new_status
    session.commit()
    session.refresh(ticket)
    return ticket


def add_ticket_note(
    session,
    *,
    ticket_id: str,
    note: str,
    author: Optional[str] = None,
):
    """Persist an internal note tied to the FOREO ticket identifier."""
    note_obj = TicketNote(ticket_id=ticket_id, note=note, author=author)
    session.add(note_obj)
    session.commit()
    session.refresh(note_obj)
    return note_obj


def list_ticket_notes(session, ticket_id: str):
    """Return notes for a ticket ordered newest first."""
    return (
        session.query(TicketNote)
        .filter(TicketNote.ticket_id == ticket_id)
        .order_by(TicketNote.created_at.desc())
        .all()
    )