"""
support_portal.py - Sprint 6 "Support Portal" Streamlit UI.

The portal lets internal agents review escalated chatbot tickets, inspect the
full chat log, update ticket status, and store internal notes. The module uses
the shared SQLAlchemy helpers in db.py so it stays in sync with the chatbot's
ticket pipeline. Simple auth hooks are stubbed out so we can layer login later.
"""

from __future__ import annotations

import html
import json
from contextlib import contextmanager
from typing import Generator, Iterable, List, Optional

import streamlit as st

import db as db_module
from db import (
    TICKET_STATUS_CHOICES,
    Ticket,
    add_ticket_note,
    get_ticket_by_id,
    get_session,
    list_ticket_notes,
    list_tickets,
    update_ticket_status,
)

# -------------------------
# Styling
# -------------------------
GLOBAL_CSS = """
<style>
.stApp {
    background: radial-gradient(1200px 600px at 15% 0%, #ffeaf3 0%, rgba(255,234,243,0) 60%),
                radial-gradient(1000px 500px at 100% 20%, #efe7ff 0%, rgba(239,231,255,0) 55%),
                linear-gradient(180deg, #faf8fc 0%, #f7f5fb 100%);
    font-family: "Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif;
}
/* Add border/divider between columns */
div[data-testid*="column"]:first-of-type {
    border-right: 2px solid rgba(226, 232, 240, 0.6);
    padding-right: 24px;
    margin-right: 0;
}
div[data-testid*="column"]:last-of-type {
    padding-left: 24px;
    margin-left: 0;
}
.hero-card {
    background: rgba(255, 255, 255, 0.78);
    backdrop-filter: blur(18px);
    border-radius: 24px;
    padding: 24px 26px;
    border: 1px solid rgba(255,255,255,0.65);
    box-shadow: 0 20px 44px rgba(167, 139, 250, 0.2);
    margin-bottom: 18px;
}
.status-badge {
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.note-card {
    padding: 10px 12px;
    border-radius: 12px;
    background: rgba(252, 252, 253, 0.9);
    border: 1px solid rgba(226, 232, 240, 0.8);
    margin-bottom: 10px;
}
.ticket-divider {
    margin: 12px 0;
    border-top: 1px dashed rgba(148, 163, 184, 0.35);
}
.meta-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 18px;
}
.meta-card {
    border-radius: 18px;
    padding: 12px 14px;
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(226,232,240,0.8);
    box-shadow: 0 10px 26px rgba(15,23,42,0.06);
}
.meta-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-bottom: 4px;
}
.meta-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #111827;
}
.section-card {
    border-radius: 18px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(226,232,240,0.9);
    box-shadow: 0 16px 32px rgba(148,163,184,0.18);
    margin-bottom: 16px;
}
.section-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #4b5563;
    margin-bottom: 6px;
}
.transcript {
    background: rgba(255,255,255,0.85);
    border-radius: 18px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.5);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.4);
}
.bubble-row {
    display:flex;
    margin-bottom: 12px;
}
.bubble {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 12px 30px rgba(15, 23, 42, .08);
}
.bubble.agent {
    margin-left:auto;
    background: linear-gradient(135deg, #87b7ff 0%, #6ae0ea 100%);
    color:#fff;
}
.bubble.user {
    margin-right:auto;
    background: #f5f6fb;
    border: 1px solid #eef1f6;
    color:#0f172a;
}
.ticket-card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 6px;
    border: 2px solid rgba(226, 232, 240, 0.6);
    box-shadow: 0 8px 24px rgba(148, 163, 184, 0.12);
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
}
.ticket-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(148, 163, 184, 0.18);
    border-color: rgba(167, 139, 250, 0.4);
}
.ticket-card.selected {
    border-color: rgba(167, 139, 250, 0.8);
    background: rgba(255, 255, 255, 0.95);
    box-shadow: 0 12px 36px rgba(167, 139, 250, 0.25);
}
.ticket-id {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 6px;
}
.ticket-meta {
    font-size: 0.85rem;
    color: #6b7280;
    margin-bottom: 8px;
}
.ticket-issue {
    font-size: 0.9rem;
    color: #374151;
    line-height: 1.4;
    margin-top: 8px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.ticket-header-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
}
.stats-container {
    display: flex;
    gap: 12px;
    margin-top: 20px;
    margin-bottom: 8px;
}
.stat-card {
    flex: 1;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 16px 18px;
    border: 2px solid rgba(226, 232, 240, 0.6);
    box-shadow: 0 8px 24px rgba(148, 163, 184, 0.12);
    text-align: center;
    transition: all 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(148, 163, 184, 0.18);
    border-color: rgba(167, 139, 250, 0.4);
}
.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 4px;
}
.stat-label {
    font-size: 0.85rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}
</style>
"""


# -------------------------
# Auth scaffolding
# -------------------------
def ensure_authenticated() -> bool:
    """
    Placeholder auth hook.

    We keep the logic isolated so email/password or SSO can be added later
    without rewriting the rest of the view code.
    """

    if "is_authenticated" not in st.session_state:
        # TODO: replace this stub with actual auth (password, OAuth, etc.)
        st.session_state.is_authenticated = True
    return st.session_state.is_authenticated


# -------------------------
# DB helpers
# -------------------------
@contextmanager
def session_scope() -> Generator:
    """Provide a transactional scope around a series of operations."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def fetch_tickets(status_filter: Optional[str], search_term: Optional[str] = None) -> List[Ticket]:
    with session_scope() as session:
        status = None if status_filter in (None, "all") else status_filter
        tickets = list_tickets(session, status=status)

    if search_term:
        needle = search_term.lower()
        filtered = []
        for ticket in tickets:
            haystack = " ".join(
                str(value or "")
                for value in [
                    ticket.ticket_id,
                    ticket.user_name,
                    ticket.user_email,
                    ticket.device_type,
                    ticket.issue_summary,
                ]
            ).lower()
            if needle in haystack:
                filtered.append(ticket)
        return filtered

    return tickets


def fetch_ticket_details(ticket_id: str) -> tuple[Optional[Ticket], list]:
    with session_scope() as session:
        ticket = get_ticket_by_id(session, ticket_id)
        notes = list_ticket_notes(session, ticket_id) if ticket else []
        return ticket, notes


def apply_ticket_updates(ticket_id: str, *, status: str, note: str) -> bool:
    """Update ticket status + optional note. Returns True on success."""
    updated = False
    with session_scope() as session:
        if status:
            update_ticket_status(session, ticket_id, status)
            updated = True
        if note.strip():
            add_ticket_note(session, ticket_id=ticket_id, note=note.strip(), author="Agent")
            updated = True
    return updated


# -------------------------
# UI helpers
# -------------------------
def render_ticket_cards(tickets: Iterable[Ticket], selected_id: Optional[str] = None) -> Optional[str]:
    """Render clickable ticket cards and return the selected ticket ID."""
    if not tickets:
        st.info("No tickets match the current filters.")
        return None

    ticket_list = list(tickets)
    ticket_ids = [t.ticket_id for t in ticket_list]

    # Initialize selected ticket if not set
    if "selected_ticket_id" not in st.session_state or st.session_state.selected_ticket_id not in ticket_ids:
        st.session_state.selected_ticket_id = ticket_ids[0] if ticket_ids else None

    # Render cards only (no selectors for now)
    for ticket in ticket_list:
        is_selected = st.session_state.selected_ticket_id == ticket.ticket_id
        card_class = "ticket-card selected" if is_selected else "ticket-card"

        date_str = ticket.created_at.strftime("%b %d, %H:%M")
        issue_preview = ticket.issue_summary or "No summary"
        if len(issue_preview) > 80:
            issue_preview = issue_preview[:77] + "..."

        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="ticket-header-row">
                    <div>
                        <div class="ticket-id">{ticket.ticket_id}</div>
                        <div class="ticket-meta">{date_str} · {ticket.user_name or 'Unknown'}</div>
                    </div>
                    {status_badge(ticket.status)}
                </div>
                <div class="ticket-issue">{html.escape(issue_preview)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.selected_ticket_id


def status_badge(status: str) -> str:
    palette = {
        "open": ("#fef2f2", "#dc2626"),
        "in_progress": ("#eff6ff", "#2563eb"),
        "resolved": ("#ecfdf5", "#059669"),
    }
    bg, fg = palette.get(status, ("#f1f5f9", "#475569"))
    label = status.replace("_", " ").title()
    return f"<span class='status-badge' style='background:{bg};color:{fg};'><span>●</span>{label}</span>"


def render_ticket_details(ticket: Ticket, notes: list) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="font-size:1.45rem;font-weight:700;">{ticket.ticket_id}</div>
                    <div style="color:#6b7280;">Created {ticket.created_at.strftime("%b %d, %Y · %H:%M UTC")}</div>
                </div>
                {status_badge(ticket.status)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compact meta cards for device / intent / source
    st.markdown(
        f"""
        <div class="meta-grid">
            <div class="meta-card">
                <div class="meta-label">Device</div>
                <div class="meta-value">{ticket.device_type or "N/A"}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Intent</div>
                <div class="meta-value">{ticket.intent or "N/A"}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Source</div>
                <div class="meta-value">{ticket.source or "chatbot"}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two stacked section cards for customer + issue
    email_html = ""
    if ticket.user_email:
        email_html = f' · <a href="mailto:{html.escape(ticket.user_email)}" style="color: #2563eb; text-decoration: underline;">{html.escape(ticket.user_email)}</a>'
    else:
        email_html = " · No email provided"
    
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">Customer</div>
            <div>{html.escape(ticket.user_name or "Unknown customer")}{email_html}</div>
        </div>
        <div class="section-card">
            <div class="section-title">Issue summary</div>
            <div>{html.escape(ticket.issue_summary or "No summary stored.")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View chat transcript", expanded=True):
        render_chat_transcript(ticket.chat_history)

    with st.expander("Internal notes", expanded=True):
        if notes:
            for note in notes:
                ts = note.created_at.strftime("%b %d · %H:%M")
                st.markdown(
                    f"""
                    <div class="note-card">
                        <div style="font-weight:600;">{note.author or 'Agent'} · <span style="color:#64748b;">{ts}</span></div>
                        <div>{note.note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No internal notes yet.")


def render_chat_transcript(raw_history: str) -> None:
    try:
        entries = json.loads(raw_history)
    except json.JSONDecodeError:
        st.code(raw_history)
        return

    if not isinstance(entries, list):
        st.json(entries)
        return

    st.markdown('<div class="transcript">', unsafe_allow_html=True)
    for entry in entries:
        role = entry.get("role", "")
        content = entry.get("content", "")
        bubble_class = "agent" if role == "assistant" else "user"
        label = "FOREO Assistant" if role == "assistant" else "Customer"
        safe_content = html.escape(str(content)).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="bubble-row">
                <div class="bubble {bubble_class}">
                    <div style="font-size:0.75rem; text-transform:uppercase; opacity:0.75; letter-spacing:0.05em;">
                        {label}
                    </div>
                    <div>{safe_content}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_update_form(ticket: Ticket) -> None:
    st.markdown("---")
    st.markdown("### Update status / add note")
    default_index = (
        TICKET_STATUS_CHOICES.index(ticket.status)
        if ticket.status in TICKET_STATUS_CHOICES
        else 0
    )

    with st.form("ticket_update_form"):
        new_status = st.selectbox(
            "Status",
            options=TICKET_STATUS_CHOICES,
            index=default_index,
            help="Keep everyone aligned by updating the ticket stage.",
        )
        note_text = st.text_area(
            "Internal note",
            placeholder="Agents-only note (optional)...",
        )
        submitted = st.form_submit_button("Save changes")

    if submitted:
        updated = apply_ticket_updates(ticket.ticket_id, status=new_status, note=note_text)
        if updated:
            st.success("Ticket updated.")
            st.rerun()
        else:
            st.warning("No changes to save.")


# -------------------------
# Entry point
# -------------------------
def main():
    st.set_page_config(page_title="FOREO Support Portal", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.title("FOREO Support Portal")

    db_module.init_db()

    if not ensure_authenticated():
        st.stop()

    st.sidebar.header("Filters")
    status_filter = st.sidebar.selectbox(
        "Ticket status",
        options=["all", *TICKET_STATUS_CHOICES],
        index=0,
    )
    search_term = st.sidebar.text_input("Search", placeholder="Ticket ID, email, device...")

    tickets = fetch_tickets(status_filter, search_term.strip() or None)
    left_col, right_col = st.columns([1, 1.3])

    # Render styled statistics cards
    open_count = sum(1 for t in tickets if t.status == "open")
    resolved_count = sum(1 for t in tickets if t.status == "resolved")
    
    st.markdown(
        f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-value">{len(tickets)}</div>
                <div class="stat-label">Tickets</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{open_count}</div>
                <div class="stat-label">Open</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{resolved_count}</div>
                <div class="stat-label">Resolved</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with left_col:
        st.subheader("Tickets")
        selected_id = render_ticket_cards(tickets)

    with right_col:
        st.subheader("Details")
        if not tickets:
            st.info("Select filters on the left to load tickets.")
        elif not selected_id:
            st.info("Choose a ticket to see details.")
        else:
            ticket, notes = fetch_ticket_details(selected_id)
            if not ticket:
                st.error("Ticket not found. It may have been removed.")
            else:
                render_ticket_details(ticket, notes)
                render_update_form(ticket)


if __name__ == "__main__":
    main()

