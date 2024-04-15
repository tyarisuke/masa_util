import csv
import os


class CSVWriter:
    def __init__(self, file_path):
        """Class for CSV file operations

        Args:
            file_path (str): Path to the CSV file to operate on
        """
        self.file_path = file_path

    def write_header(self, header):
        """Writes a header (title row) to the CSV file if it doesn't already exist.

        Args:
            header (list): List of strings to write as the header
        """
        if not os.path.exists(self.file_path):
            with open(
                self.file_path, "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(header)

    def append_row(self, row):
        """Appends a data row to the CSV file

        Args:
            row (list): Data row to append
        """
        with open(self.file_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(row)
