import os
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


# 使用例
directory = "."  # 検索するディレクトリのパス
ascending = True  # Trueで昇順、Falseで降順
extension_filter = "py"  # フィルタリングする拡張子（例：'py'）
recursive = True  # Trueでサブフォルダも検索

sorted_paths = get_sorted_file_paths(directory, ascending, extension_filter, recursive)
print(sorted_paths)
