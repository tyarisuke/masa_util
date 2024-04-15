import time
from datetime import datetime

import pytz


def current_time_str(fmt="%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current JST time as a string in the format 'YYYY-MM-DD HH:MM:SS'.

    Args:
    - None

    Returns:
    - str: A string representing the current JST time.
    """
    tz_tokyo = pytz.timezone("Asia/Tokyo")
    now = datetime.now(tz_tokyo)
    return now.strftime(fmt)


def current_unix_time() -> int:
    """
    Returns the current JST time as a UNIX timestamp.

    Args:
    - None

    Returns:
    - int: A UNIX timestamp representing the current JST time.
    """
    return int(time.time())


def str_time_to_unix(time_str: str, fmt="%Y-%m-%d %H:%M:%S") -> int:
    """
    Converts a given time string to a UNIX timestamp.

    Args:
    - time_str (str): A time string in the format 'YYYY-MM-DD HH:MM:SS'.
    - fmt (str): The format of the input time string. Default is '%Y-%m-%d %H:%M:%S'.

    Returns:
    - int: A UNIX timestamp representing the specified time.
    """
    tz_tokyo = pytz.timezone("Asia/Tokyo")
    dt = datetime.strptime(time_str, fmt)
    dt = tz_tokyo.localize(dt)
    return int(dt.timestamp())


def unix_time_to_str(unix_time: int, fmt="%Y-%m-%d %H:%M:%S") -> str:
    """
    Converts a given UNIX timestamp to a JST time string in the format 'YYYY-MM-DD HH:MM:SS'.

    Args:
    - unix_time (int): A UNIX timestamp.
    - fmt (str): The format for the output time string. Default is '%Y-%m-%d %H:%M:%S'.

    Returns:
    - str: A string representing the specified UNIX timestamp in JST.
    """
    tz_tokyo = pytz.timezone("Asia/Tokyo")
    dt = datetime.fromtimestamp(unix_time, tz_tokyo)
    return dt.strftime(fmt)
