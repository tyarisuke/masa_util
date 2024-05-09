import subprocess
import time

import mss
import numpy as np
import pyautogui as ag
import pygetwindow as gw
from PIL import ImageGrab
from pygetwindow import PyGetWindowException as GetWindowException


def key_press(key):
    ag.press(key)


def type_write(str):
    ag.typewrite(str)


def hotkey(*key):
    ag.hotkey(key)


def key_down(key):
    ag.keyDown(key)


def key_up(key):
    ag.keyUp(key)


def launch_app(path, options=None):
    """
    Launches an application at the specified path with optional command-line options.

    Args:
    - path (str): The path to the application to launch.
    - options (list of str, optional): A list of command-line options to pass to the application. Defaults to None.

    Returns:
    - None: This function does not return any value.
    """
    # Use an empty list for options if none are provided
    if options is None:
        options = []

    # Combine the application path with any options
    command = [path] + options

    # Execute the command
    subprocess.run(command, check=True)


def launch_app_and_detect_new_window(path, options=None):
    """
    Launches an application and attempts to detect the new window opened by the application.

    Args:
    - path (str): The path to the application to launch.
    - options (list of str, optional): Command-line options to pass to the application. Defaults to None.

    Returns:
    - new_windows (list): A list of new window objects opened by the application.
    """
    # Get a set of all current window handles
    before_windows = set(window._hWnd for window in gw.getAllWindows())

    # Launch the application
    if options is None:
        options = []
    command = [path] + options
    subprocess.Popen(command)

    # Wait for the application to possibly open new windows
    time.sleep(2)  # Adjust this sleep time as necessary

    # Get a new set of all window handles
    after_windows = set(window._hWnd for window in gw.getAllWindows())

    # Find the difference in handles
    new_handles = after_windows - before_windows

    # Find the corresponding window objects for the new handles
    new_windows = [
        window for window in gw.getAllWindows() if window._hWnd in new_handles
    ]
    return new_windows[0]


def get_window_id(window_name):
    return gw.getWindowsWithTitle(window_name)[0]


def activate_window(window):
    window.activate()


def get_window_topleft(window):
    return window.topleft


def get_window_size(window):
    return window.size


def screenshot(window):
    with mss.mss() as sct:
        x, y = get_window_topleft(window)
        w, h = get_window_size(window)
        monitor = {"top": y, "left": x, "width": w, "height": h}
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        return img


if __name__ == "__main__":

    window = get_window_id("Monopoly_2 - Snes9x")
    activate_window(window)
    x, y = get_window_topleft(window)
    w, h = get_window_size(window)
    ss = screenshot(window)
    print(x, y)
    print(w, h)
    print(ss)
