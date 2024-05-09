import os
import re
from typing import List


def get_sorted_file_paths(
    directory: str,
    ascending: bool = True,
    extension_filter: str = "",
    recursive: bool = False,
) -> List[str]:
    """
    指定されたディレクトリからファイルパスを取得し、条件に応じてソートして返します。

    Args:
    - directory (str): ファイルパスを取得するディレクトリのパス。
    - ascending (bool): Trueの場合は昇順、Falseの場合は降順でソートします。デフォルトはTrue。
    - extension_filter (str): 取得するファイルの拡張子を指定します。指定がない場合はすべてのファイルを対象にします。例: 'txt'
    - recursive (bool): Trueの場合はサブフォルダも含めて検索します。デフォルトはFalse。

    Returns:
    - List[str]: ソートされたファイルパスのリスト。

    例:
    >>> get_sorted_file_paths('/path/to/directory', ascending=False, extension_filter='txt', recursive=True)
    ['/path/to/directory/subfolder/file1.txt', '/path/to/directory/file2.txt']
    """
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    if extension_filter:
        pattern += "." + extension_filter.lstrip(".")

    # サブフォルダも含めてパスを取得
    paths = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(extension_filter)
    ]

    # 昇順または降順でソート
    paths.sort(reverse=not ascending)

    return paths


def pad_file_numbers(directory: str, num_digits: int):
    """
    ディレクトリ内のすべてのファイルを検索し、ファイル名内の数字を指定された桁数でゼロ埋めします。

    Args:
    - directory (str): ファイル名を変更するディレクトリのパス。
    - num_digits (int): ゼロ埋めする桁数。

    例:
    ディレクトリに 'image1.png', 'image12.png' がある場合、num_digits=4 とすると、
    ファイル名は 'image0001.png', 'image0012.png' に変更されます。
    """
    # ディレクトリ内のすべてのファイルを列挙
    for filename in os.listdir(directory):
        # ファイル名から数字部分を抽出
        match = re.search(r"\d+", filename)
        if match:
            num = match.group()
            # 数字を指定された桁数でゼロ埋め
            new_num = num.zfill(num_digits)
            # 新しいファイル名を生成
            new_filename = re.sub(r"\d+", new_num, filename, 1)
            # ファイル名を変更
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{filename}' to '{new_filename}'")


if __name__ == "__main__":

    # 使用例
    directory = "."  # 検索するディレクトリのパス
    ascending = True  # Trueで昇順、Falseで降順
    extension_filter = "py"  # フィルタリングする拡張子（例：'py'）
    recursive = True  # Trueでサブフォルダも検索

    sorted_paths = get_sorted_file_paths(
        directory, ascending, extension_filter, recursive
    )
    print(sorted_paths)
