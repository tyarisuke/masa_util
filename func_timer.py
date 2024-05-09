import timeit

import matplotlib.pyplot as plt
import numpy as np


class FuncTimer:
    """
    指定された関数を異なる引数セットで複数回実行し、実行時間を測定し、統計情報とグラフを提供するクラス。

    Attributes:
        func (callable): 実行時間を測定する関数。
        arg_sets (list of tuple): 各関数呼び出しに使用する引数のリスト。
        timings (list of float): 各測定の実行時間。

    Methods:
        measure(): 関数の実行時間を測定し、結果を `timings` に記録する。
        display_statistics(): 測定した実行時間の平均と標準偏差を表示する。
        plot_timings(): 測定結果のグラフを表示する。
    """

    def __init__(self, func, arg_sets):
        """
        クラスのインスタンスを初期化する。

        Parameters:
            func (callable): 実行時間を測定する関数。
            arg_sets (list of tuple): 各関数呼び出しで使用する引数のリスト。
        """
        self.func = func
        self.arg_sets = arg_sets
        self.timings = []

    def measure(self):
        """
        関数の実行時間を複数回測定し、結果を `timings` に保存する。
        """
        for args in self.arg_sets:
            timer = timeit.Timer(lambda: self.func(*args))
            time = timer.timeit(number=1)
            self.timings.append(time)

    def display_statistics(self):
        """
        測定した実行時間の統計情報を計算し、表示する。
        """
        mean_time = np.mean(self.timings)
        std_dev = np.std(self.timings)
        print(f"Average execution time: {mean_time:.5f} seconds")
        print(f"Standard deviation: {std_dev:.5f} seconds")

    def plot_timings(self):
        """
        測定した実行時間の累積数を表示する棒グラフを描画する。
        横軸は時間の区間、縦軸はその時間区間に含まれる試行の累積数。
        """
        # 実行時間を区間に分ける
        bins = np.linspace(
            min(self.timings), max(self.timings), 10
        )  # 10個の区間に分ける
        histogram, bin_edges = np.histogram(self.timings, bins=bins)

        # 区間の中心値を計算
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        plt.figure(figsize=(10, 5))
        plt.bar(
            bin_centers,
            histogram,
            width=np.diff(bin_edges),
            edgecolor="black",
            align="center",
        )
        plt.xlabel("Time Interval (seconds)")
        plt.ylabel("Number of Executions")
        plt.title("Cumulative Number of Executions Over Time Intervals")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    # 使用例
    def sample_function(a, b):
        total = 0
        for i in range(a):
            total += i * b

    arg_sets = [
        (1000000, 2),
        (1500000, 1),
        (2000000, 2),
        (500000, 3),
        (1000000, 5),
    ]

    timer = FuncTimer(sample_function, arg_sets=arg_sets)
    timer.measure()
    timer.display_statistics()
    timer.plot_timings()
