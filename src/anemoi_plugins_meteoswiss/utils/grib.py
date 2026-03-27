import re
from datetime import timedelta

STEP_PATTERN = re.compile(
    r"^\s*(?:(?P<hours>\d+)\s*h)?\s*(?:(?P<minutes>\d+)\s*m)?\s*$"
)


def decode_step(s: str) -> timedelta:
    # if castable to int, assume it's hours
    try:
        hours = int(s)
        return timedelta(hours=hours)
    except ValueError:
        pass

    # else parse with regex (e.g. "1h30m", "45m", "2h", etc.)
    match = STEP_PATTERN.match(s)
    if not match:
        raise ValueError(f"Invalid duration string: {s!r}")
    parts = {k: int(v) if v else 0 for k, v in match.groupdict().items()}
    return timedelta(**parts)


def encode_step(td: timedelta) -> str:
    # if only hours, return as int string unless it's zero (then it's 0m)
    if td.total_seconds() % 3600 == 0 and td.total_seconds() != 0:
        return str(int(td.total_seconds() // 3600))

    # else return as "XhYm" format
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes = remainder // 60
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return "".join(parts)
