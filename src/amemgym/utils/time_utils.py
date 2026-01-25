import random
from datetime import datetime, timedelta


def load_date(date_str):
    """
    Parse a date string in YYYY-MM-DD format into a datetime object.
    
    Args:
        date_str (str): Date string in the format "YYYY-MM-DD" (e.g., "2018-05-01")
    
    Returns:
        datetime: A datetime object representing the parsed date
    """
    return datetime.strptime(date_str, "%Y-%m-%d")


def date_plus_months(date, num_months):
    """
    Add a specified number of months to a given date.

    Args:
        date (datetime): The original date.
        num_months (int): The number of months to add.

    Returns:
        datetime: A new datetime object with the added months.
    """
    year = date.year
    month = date.month + num_months
    
    # Handle year overflow/underflow
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    
    month2days = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }
    # Handle day overflow (e.g., Jan 31 + 1 month should be Feb 28/29, not Mar 3)
    day = min(date.day, month2days[month])
    return date.replace(year=year, month=month, day=day)


def sample_session_timestamps(start_date, end_date, num_sessions):
    """
    Sample a list of session timestamps between start_date and end_date.

    Args:
        start_date (datetime): Start date for sampling. Can be None.
        end_date (datetime): End date for sampling.
        num_sessions (int): Number of session timestamps to sample.

    Returns:
        list: A sorted list of sampled datetime objects representing session timestamps.
    """
    if start_date is not None:
        # Calculate the number of days between start (exclusive) and end (inclusive)
        total_days = (end_date - start_date).days
        
        # Ensure num_sessions doesn't exceed available days
        if num_sessions > total_days:
            raise ValueError(f"num_sessions ({num_sessions}) cannot exceed the number of days between start_date and end_date ({total_days})")
    
    # Get the last num_sessions days before (and including) end_date
    session_dates = []
    for i in range(num_sessions):
        session_date = end_date - timedelta(days=i)
        session_dates.append(session_date)
    
    # Sort dates chronologically
    session_dates.sort()
    
    # Sample timestamps between 19:00-22:00 for each date
    session_times = []
    for session_date in session_dates:
        # Random seconds between 19:00 (19*3600) and 22:00 (22*3600)
        random_seconds = random.randint(19 * 3600, 22 * 3600)
        session_time = session_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=random_seconds)
        session_times.append(session_time)
    
    return session_times


def next_round_timestamp(last_timestamp, min_gap_seconds=30, max_gap_seconds=120):
    """
    Generate the next round timestamp by adding a random gap to the last timestamp.
    
    Args:
        last_timestamp (datetime): The previous timestamp.
        min_gap_seconds (int): Minimum gap in seconds to add.
        max_gap_seconds (int): Maximum gap in seconds to add.
    
    Returns:
        datetime: The next timestamp with a random gap added.
    """
    gap_seconds = random.randint(min_gap_seconds, max_gap_seconds)
    return last_timestamp + timedelta(seconds=gap_seconds)
