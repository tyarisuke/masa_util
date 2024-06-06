import ast
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


def _clean_dict_string(dict_str):
    """
    辞書形式の文字列から< および >を"に変換します
    """
    # <...>構文をに置換
    cleaned_str = re.sub(r"<|>", '"', dict_str)
    return cleaned_str


def extract_dict_from_text(content):
    """
    テキストの内容から辞書形式の部分を見つけ出し、辞書に変換して返します。

    Parameters:
    content (str): テキストの内容

    Returns:
    list: 辞書形式の部分を辞書に変換したリスト
    """
    dict_pattern = r"\{[^{}]*\}"
    matches = re.findall(dict_pattern, content)

    dict_list = []
    for match in matches:
        try:
            cleaned_match = _clean_dict_string(match)
            cleaned_match = cleaned_match.replace("'", '"')
            dict_obj = ast.literal_eval(cleaned_match)
            if isinstance(dict_obj, dict):
                dict_list.append(dict_obj)
        except (ValueError, SyntaxError):
            continue

    return dict_list


def extract_lists_from_text(content):
    """
    テキストの内容からリスト形式の部分を見つけ出し、リストに変換して返します。

    Parameters:
    content (str): テキストの内容

    Returns:
    list: リスト形式の部分をリストに変換したリスト
    """
    # リスト形式の部分を見つけるための正規表現パターン
    list_pattern = r"\[[^\[\]]*\]"

    list_matches = re.findall(list_pattern, content)

    list_list = []

    # リスト形式の部分を処理
    for match in list_matches:
        try:
            match = match.replace("'", '"')
            list_obj = ast.literal_eval(match)
            if isinstance(list_obj, list):
                list_list.append(list_obj)
        except (ValueError, SyntaxError):
            continue

    return list_list
