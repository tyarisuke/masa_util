import re


def read_text_file(file_path):
    """
    指定されたファイルパスからテキストファイルを読み込み、その内容を文字列として返します。

    Parameters:
    file_path (str): 読み込むテキストファイルのパス

    Returns:
    str: 読み込まれたテキストファイルの内容
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {file_path}")
    except Exception as e:
        print(f"予期しないエラーが発生しました: {str(e)}")


def read_text_file_lines(file_path):
    """
    指定されたファイルパスからテキストファイルを読み込み、その内容を行ごとのリストとして返します。

    Parameters:
    file_path (str): 読み込むテキストファイルのパス

    Returns:
    list: 読み込まれたテキストファイルの各行のリスト
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {file_path}")
    except Exception as e:
        print(f"予期しないエラーが発生しました: {str(e)}")


def extract_patterns_from_text(content, pattern):
    """
    テキストの内容から指定された正規表現パターンにマッチする文字列を抽出します。

    Parameters:
    content (str): テキストの内容
    pattern (str): 正規表現パターン

    Returns:
    list: 正規表現パターンにマッチする文字列のリスト
    """
    try:
        matches = re.findall(pattern, content)
        return matches
    except re.error as e:
        print(f"正規表現エラーが発生しました: {str(e)}")
