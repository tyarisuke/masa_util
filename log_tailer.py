import argparse
import re
import subprocess
import sys


def _default_callback(output):
    """Default callback function to process and print the extracted output."""
    print(output, end="")


def _extract_with_regex(output, patterns, callback):
    """Extracts matching strings from the output using regular expressions and passes them to the callback function."""
    extracted = []
    for pattern in patterns:
        matches = re.findall(pattern, output)
        extracted.extend(matches)
    if extracted:
        callback(" ".join(extracted))


def follow_and_grep_file(
    filename, grep_patterns, extract_patterns=None, callback=_default_callback
):
    """
    (Public) Monitors a log file in real-time, filters lines matching specified patterns using grep,
    extracts parts of these lines using regular expressions, and processes them using a callback function.

    filename: Path to the log file to monitor.
    grep_patterns: List of patterns to filter lines in the log file.
    extract_patterns: List of regular expressions to extract specific strings from the filtered lines.
    callback: Callback function to process the extracted strings.
    """
    # Setup the tail command
    tail_command = ["tail", "--follow=name", "--retry", filename]
    tail_process = subprocess.Popen(
        tail_command, stdout=subprocess.PIPE, text=True
    )

    prev_process = tail_process

    # Chain grep commands for each pattern
    for pattern in grep_patterns:
        grep_command = ["grep", "--line-buffered", pattern]
        grep_process = subprocess.Popen(
            grep_command,
            stdin=prev_process.stdout,
            stdout=subprocess.PIPE,
            text=True,
        )
        prev_process = grep_process

    final_process = prev_process

    try:
        while True:
            output = final_process.stdout.readline()
            if output == "" and final_process.poll() is not None:
                break
            if output and extract_patterns:
                _extract_with_regex(output, extract_patterns, callback)
    except KeyboardInterrupt:
        final_process.kill()
        final_process.wait()
        if final_process != tail_process:
            tail_process.kill()
            tail_process.wait()
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Monitors a log file in real-time, filters lines with given patterns, optionally extracts specific strings using regular expressions, and processes them with a callback function."
    )
    parser.add_argument("logfile", help="Path to the log file to monitor")
    parser.add_argument(
        "--grep",
        nargs="+",
        help="Series of patterns to filter the log file",
        default=[],
    )
    parser.add_argument(
        "--extract",
        nargs="+",
        help="Regular expressions to further extract specific strings from the grep result",
        default=None,
    )

    args = parser.parse_args()

    if args.grep:
        follow_and_grep_file(args.logfile, args.grep, args.extract)
    else:
        print("Please specify at least one pattern.")


if __name__ == "__main__":
    main()
