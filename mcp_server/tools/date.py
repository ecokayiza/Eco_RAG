from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def date(timezone: str | None = None) -> dict[str, Any]:
    """Return current date/time info for an optional IANA timezone."""
    try:
        tz_name = str(timezone or "").strip()
        tzinfo = ZoneInfo(tz_name) if tz_name else datetime.now().astimezone().tzinfo
        now = datetime.now(tzinfo).replace(microsecond=0)
        zone_name = tz_name or now.tzname() or str(now.tzinfo or "")
        content = f"{now.strftime('%A, %B %d, %Y %H:%M:%S')} ({zone_name})."
        return {
            "type": "context",
            "skill_name": "date",
            "items": [
                {
                    "title": "Current date and time",
                    "content": content,
                    "date": now.date().isoformat(),
                    "time": now.strftime("%H:%M:%S"),
                    "weekday": now.strftime("%A"),
                    "iso": now.isoformat(),
                    "timezone": zone_name,
                    "utc_offset": _utc_offset(now),
                }
            ],
        }
    except ZoneInfoNotFoundError:
        return {
            "type": "context",
            "skill_name": "date",
            "items": [],
            "error": f"Unknown timezone: {timezone}",
        }
    except Exception as exc:
        return {
            "type": "context",
            "skill_name": "date",
            "items": [],
            "error": str(exc),
        }


def _utc_offset(value: datetime) -> str:
    offset = value.utcoffset()
    if offset is None:
        return ""
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hours, minutes = divmod(total_minutes, 60)
    return f"{sign}{hours:02d}:{minutes:02d}"
