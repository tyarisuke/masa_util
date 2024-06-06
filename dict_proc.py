import csv


def has_value(dict_obj, key, value):
    """
    指定された辞書が指定されたキーで指定された値を持つかを判定します。

    Parameters:
    dict_obj (dict): 判定する辞書
    key (str): 判定するキー
    value: 判定する値

    Returns:
    bool: 指定されたキーが指定された値を持つ場合は True、それ以外の場合は False
    """
    return dict_obj.get(key) == value


def save_dict_list_to_csv(dict_list, csv_file_path):
    """
    辞書のリストをCSVファイルに保存します。

    Parameters:
    dict_list (list of dict): 保存する辞書のリスト
    csv_file_path (str): 保存先のCSVファイルのパス
    """
    if not dict_list:
        print("辞書のリストが空です。")
        return

    # 辞書のリストからフィールド名（キーのリスト）を取得
    fieldnames = dict_list[0].keys()

    try:
        with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # ヘッダーを書き込む
            writer.writeheader()

            # 辞書のリストを書き込む
            for dictionary in dict_list:
                writer.writerow(dictionary)

        print(f"CSVファイルに保存しました: {csv_file_path}")

    except IOError as e:
        print(f"ファイル書き込みエラー: {str(e)}")
