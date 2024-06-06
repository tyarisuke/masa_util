import matplotlib.pyplot as plt
import pandas as pd


def dict_list_to_dataframe(dict_list):
    """
    辞書のリストからPandasのデータフレームを作成します。

    Parameters:
    dict_list (list of dict): データフレームに変換する辞書のリスト

    Returns:
    pd.DataFrame: 作成されたデータフレーム
    """
    if not dict_list:
        print("辞書のリストが空です。")
        return pd.DataFrame()

    try:
        dataframe = pd.DataFrame(dict_list)
        return dataframe
    except Exception as e:
        print(f"データフレームの作成中にエラーが発生しました: {str(e)}")
        return pd.DataFrame()


def plot_multiple_line_graphs(
    df,
    x_col,
    y_cols,
    show_markers=None,
    line_colors=None,
    reverse_x=False,
    reverse_y=False,
):
    """
    データフレームから指定された列を使用して複数の線グラフを表示します。
    y_cols の中にリストが含まれている場合、そのリストに含まれる各列に対して個別の線をプロットします。

    Parameters:
    df (pd.DataFrame): データフレーム
    x_col (str): x軸になる列の名前
    y_cols (list of str or list of list of str): y軸になる列の名前のリストまたはリストのリスト
    show_markers (list of bool or list of list of bool): 各線グラフでプロット点を表示するかどうかのリストまたはリストのリスト
    line_colors (list of str or list of list of str): 各線の色のリストまたはリストのリスト
    reverse_x (bool): X軸を逆順に表示するかどうか
    reverse_y (bool): Y軸を逆順に表示するかどうか
    """
    if x_col not in df.columns:
        print(
            f"指定されたX軸の列がデータフレームに存在しません: x_col={x_col}"
        )
        return

    for y_col in y_cols:
        if isinstance(y_col, list):
            for sub_y_col in y_col:
                if sub_y_col not in df.columns:
                    print(
                        f"指定されたY軸の列がデータフレームに存在しません: y_col={sub_y_col}"
                    )
                    return
        else:
            if y_col not in df.columns:
                print(
                    f"指定されたY軸の列がデータフレームに存在しません: y_col={y_col}"
                )
                return

    if line_colors is None:
        line_colors = [
            ["blue"] * len(y_col) if isinstance(y_col, list) else "blue"
            for y_col in y_cols
        ]
    elif len(line_colors) < len(y_cols):
        print("線の色のリストの長さがY軸の列のリストの長さと一致しません。")
        return

    if show_markers is None:
        show_markers = [
            [True] * len(y_col) if isinstance(y_col, list) else True
            for y_col in y_cols
        ]
    elif len(show_markers) < len(y_cols):
        print(
            "プロット点表示オプションのリストの長さがY軸の列のリストの長さと一致しません。"
        )
        return

    fig, axes = plt.subplots(
        len(y_cols), 1, figsize=(10, 6 * len(y_cols)), sharex=True
    )

    if len(y_cols) == 1:
        axes = [axes]

    for i, (ax, y_col, line_color, show_marker) in enumerate(
        zip(axes, y_cols, line_colors, show_markers)
    ):
        if isinstance(y_col, list):
            for j, sub_y_col in enumerate(y_col):
                marker = "o" if show_marker[j] else None
                color = (
                    line_color[j]
                    if isinstance(line_color, list)
                    else line_color
                )
                ax.plot(
                    df[x_col],
                    df[sub_y_col],
                    marker=marker,
                    color=color,
                    label=sub_y_col,
                )
            ax.legend()
        else:
            marker = "o" if show_marker else None
            ax.plot(df[x_col], df[y_col], marker=marker, color=line_color)

        ax.set_ylabel(y_col if isinstance(y_col, str) else ", ".join(y_col))
        ax.grid(True)
        if reverse_x:
            ax.invert_xaxis()
        if reverse_y:
            ax.invert_yaxis()

    axes[-1].set_xlabel(x_col)
    plt.suptitle("Multiple Line Graphs")
    plt.show()
