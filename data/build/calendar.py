from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)


NY_TIME = ZoneInfo("America/New_York")
DATA_READY_TIME = time(17, 30)


class MarketHolidayCalendar(AbstractHolidayCalendar):
    """NYSE-style full-day market holidays.

    Example:
        Juneteenth 2026 is a non-market day.
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def default_as_of_date(now=None):
    """Return the last completed US market date.

    Example:
        2026-06-17 13:40 New York returns 2026-06-16.
    """
    return last_completed_market_date(now).isoformat()


def last_completed_market_date(now=None):
    """Return the newest market date with final daily bars.

    Example:
        after the data cutoff on a trading day, that day is complete.
    """
    now = ny_time(now)
    today = now.date()
    if is_market_day(today) and now.time() >= DATA_READY_TIME:
        return today
    return previous_market_day(today - timedelta(days=1))


def previous_market_day(day):
    """Return the nearest market day on or before day.

    Example:
        a Sunday returns the prior Friday unless Friday was a holiday.
    """
    while not is_market_day(day):
        day -= timedelta(days=1)
    return day


def is_market_day(day):
    """Return True when US equities had a full regular session.

    Example:
        Saturdays and NYSE holidays return False.
    """
    if day.weekday() >= 5:
        return False
    return day not in market_holidays(day.year)


def market_holidays(year):
    """Return market holiday dates for one year.

    Example:
        market_holidays(2026) includes Christmas observed.
    """
    calendar = MarketHolidayCalendar()
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    return set(calendar.holidays(start=start, end=end).date)


def ny_time(value=None):
    """Return a timezone-aware New York datetime.

    Example:
        a naive datetime is treated as New York time.
    """
    if value is None:
        return datetime.now(NY_TIME)
    if value.tzinfo is None:
        return value.replace(tzinfo=NY_TIME)
    return value.astimezone(NY_TIME)
