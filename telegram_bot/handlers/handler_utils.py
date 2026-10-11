from __future__ import annotations

from typing import Any, Optional

# Shared pending interactive wizard state dictionaries keyed by Telegram user_id
_PENDING_SETUP_MODE: dict[int, bool] = {}
_PENDING_ADMIN_ACTION: dict[int, dict[str, Any]] = {}


def clear_pending_wizards(user_id: int) -> None:
    """Clear both /setup and /admin pending interactive wizard states for `user_id`."""
    uid = int(user_id)
    _PENDING_SETUP_MODE.pop(uid, None)
    _PENDING_ADMIN_ACTION.pop(uid, None)


def extract_user_identity(event: Any, default_name: str = "Trader") -> tuple[int, str]:
    """Extract (user_id, display_name) from a Telegram Message or CallbackQuery."""
    user = getattr(event, "from_user", None)
    user_id = int(getattr(user, "id", 0) or 0)
    user_name = (
        getattr(user, "full_name", None)
        or getattr(user, "username", None)
        or default_name
    )
    return user_id, str(user_name)


def parse_command_args(text: Optional[str]) -> list[str]:
    """Extract space-delimited arguments following a slash command."""
    if not text:
        return []
    parts = text.strip().split()
    return parts[1:] if parts and parts[0].startswith("/") else parts
