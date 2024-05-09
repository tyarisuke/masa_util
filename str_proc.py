import re


def replace_pattern(text, pattern, replacement):
    """
    指定された文字列において、指定された正規表現にマッチする部分を修正後の文字列で置換する。

    Args:
    text (str): 入力文字列。
    pattern (str): 検索する正規表現。
    replacement (str): 置換後の文字列。

    Returns:
    str: 置換後の文字列。
    """
    # $0, $1, ..., $9を適切な形に置換
    for i in range(10):
        replacement = replacement.replace(f"${i}", f"\\{i}")
    return re.sub(pattern, replacement, text)
