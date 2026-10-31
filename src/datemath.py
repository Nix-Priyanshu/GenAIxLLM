"""Deterministic date math so the agent never miscounts business days or
return windows. The LLM should call these instead of doing arithmetic itself.

Business day = Monday-Saturday, excluding data/holidays.csv (kb/00_about_this_
knowledge_base.md [KB-00.3]). Return windows are counted in CALENDAR days.
Refund/logistics timelines are counted in BUSINESS days.
"""
import csv
import os
from datetime import date, datetime, timedelta

try:
    from src.config import DATA_DIR, TODAY
except ImportError:
    from config import DATA_DIR, TODAY

_HOLIDAYS = None


def _holidays() -> set:
    global _HOLIDAYS
    if _HOLIDAYS is None:
        path = os.path.join(DATA_DIR, "holidays.csv")
        with open(path, encoding="utf-8") as f:
            _HOLIDAYS = {datetime.strptime(r["date"], "%Y-%m-%d").date()
                         for r in csv.DictReader(f)}
    return _HOLIDAYS


def _parse(d) -> date:
    if isinstance(d, date):
        return d
    if isinstance(d, datetime):
        return d.date()
    s = str(d).strip()
    # orders.csv timestamps look like "2026-10-01T14:27"; dates like "2026-10-01"
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def is_business_day(d) -> bool:
    d = _parse(d)
    return d.weekday() != 6 and d not in _holidays()  # Sunday = 6


def add_business_days(start, n: int) -> date:
    """n business days after `start` (start itself not counted)."""
    d = _parse(start)
    remaining = n
    while remaining > 0:
        d += timedelta(days=1)
        if is_business_day(d):
            remaining -= 1
    return d


def business_days_between(start, end) -> int:
    """Business days elapsed strictly between start (exclusive) and end (inclusive)."""
    start, end = _parse(start), _parse(end)
    count = 0
    d = start
    while d < end:
        d += timedelta(days=1)
        if is_business_day(d):
            count += 1
    return count


def calendar_days_between(start, end) -> int:
    return (_parse(end) - _parse(start)).days


def return_window_days(category: str, delivered_on, order_placed_on=None) -> int:
    """Return window in calendar days for a product's return_category, applying:
    - RET-3: electronics delivered before 2026-08-01 keep the 10-day window
      from policy v3; from that date it's 7 days.
    - RET-8: clothing/footwear/bags (fashion) ordered 1-15 Oct 2026 get a
      30-day window instead of the normal 15, regardless of delivery date.
    """
    windows = {
        "electronics": 7, "fashion": 15, "home_kitchen": 10,
        "beauty_sealed": 7, "books_media": 10,
        "grocery": 0, "innerwear": 0, "digital": 0,
    }
    if category == "electronics" and delivered_on is not None:
        if _parse(delivered_on) < date(2026, 8, 1):
            return 10  # RETv3-2, still applies (RET-3)
    if category == "fashion" and order_placed_on is not None:
        placed = _parse(order_placed_on)
        if date(2026, 10, 1) <= placed <= date(2026, 10, 15):
            return 30  # RET-8 festive sale window
    return windows.get(category, 0)


def return_deadline(category: str, delivered_on, order_placed_on=None):
    """The actual calendar date a return must be made by."""
    window = return_window_days(category, delivered_on, order_placed_on)
    if window == 0 or delivered_on is None:
        return None
    from datetime import timedelta as _td
    return _parse(delivered_on) + _td(days=window)


def is_within_return_window(category: str, delivered_on, as_of=None, order_placed_on=None) -> bool:
    window = return_window_days(category, delivered_on, order_placed_on)
    if window == 0:
        return False
    as_of = _parse(as_of) if as_of else TODAY
    return calendar_days_between(delivered_on, as_of) <= window


def reverse_pickup_fee_inr(category: str, is_plus: bool, reason_is_damage: bool = False) -> int:
    """RET-6: Rs 49 fee on change-of-mind returns of clothing/footwear/bags,
    waived for Plus members and for any damaged/defective/wrong-item return."""
    if category == "fashion" and not is_plus and not reason_is_damage:
        return 49
    return 0


def hours_since(timestamp, as_of=None) -> float:
    """Hours elapsed since an ISO timestamp like 2026-10-18T14:27, vs TODAY (midnight)."""
    ts = datetime.strptime(str(timestamp)[:16], "%Y-%m-%dT%H:%M")
    as_of_dt = datetime.combine(as_of or TODAY, datetime.min.time())
    return (as_of_dt - ts).total_seconds() / 3600


# REF-2's payment-method -> business-day range (kept here as a stable policy
# constant so it can be computed deterministically; cross-check against
# kb/pol_refunds.md REF-2 if the policy document changes).
REFUND_RANGES_BUSINESS_DAYS = {
    "upi": (1, 3), "card": (5, 7), "credit card": (5, 7), "debit card": (5, 7),
    "net banking": (3, 5), "netbanking": (3, 5),
    "wallet": (0, 0),  # instant, within 2 hours -- not business-day counted
    "cod_bank": (5, 7), "bank transfer": (5, 7), "neft": (5, 7),
}

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}

_DATE_RE = None


def find_date_in_text(text: str):
    """Find a 'DD Month YYYY' date in free text (the format this dataset's
    practice questions use), e.g. '17 October 2026'. Returns a date or None."""
    global _DATE_RE
    if _DATE_RE is None:
        import re
        _DATE_RE = re.compile(
            r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|"
            r"August|September|October|November|December)\s+(\d{4})\b", re.I)
    m = _DATE_RE.search(text)
    if not m:
        return None
    day, month_name, year = m.groups()
    return date(int(year), MONTHS[month_name.lower()], int(day))


def refund_window_for(payment_method: str, initiated_on) -> str:
    """Human-readable earliest-to-latest refund arrival date, computed with
    real business-day math instead of leaving the arithmetic to the LLM."""
    key = payment_method.lower().strip()
    if key not in REFUND_RANGES_BUSINESS_DAYS:
        return ""
    lo, hi = REFUND_RANGES_BUSINESS_DAYS[key]
    if lo == 0:
        return f"{payment_method} refunds are instant (within 2 hours of initiation), not business-day counted."
    earliest = add_business_days(initiated_on, lo)
    latest = add_business_days(initiated_on, hi)

    def fmt(d):
        return f"{d.day} {d.strftime('%B %Y')}"  # no leading zero on the day

    return (f"Refund initiated {fmt(initiated_on)} via {payment_method}: "
            f"{lo} to {hi} business days -> expected between "
            f"{fmt(earliest)} and {fmt(latest)} (i.e. by {fmt(latest)} at the latest).")


if __name__ == "__main__":
    print("Today:", TODAY)
    print("Is 2026-10-19 a business day?", is_business_day("2026-10-19"))  # False: warehouse closure
    print("5 business days after 2026-10-20:", add_business_days("2026-10-20", 5))
    print("Electronics delivered 2026-07-20, window:", return_window_days("electronics", "2026-07-20"), "days")
    print("Electronics delivered 2026-09-01, window:", return_window_days("electronics", "2026-09-01"), "days")
    print("Within window?", is_within_return_window("electronics", "2026-09-01"))
