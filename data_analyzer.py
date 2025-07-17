import pandas as pd

class DataAnalyzer:
    def __init__(self, data, processed_data):
        self.data = data
        self.processed_data = processed_data

    def analyze(self):
        print("\n数据集的前几行：")
        print(self.data.head())

        print("\n数据集的基本信息：")
        print(self.data.info())

        print("\n数据集的统计信息：")
        print(self.data.describe())

        print("\n清洗后的数据：")
        print(self.processed_data.head())

        print("\n描述性统计信息：")
        print(self.processed_data.describe(include='all'))

        print("\n偏度和峰度：")
        print(self.processed_data['Average Price'].skew())
        print(self.processed_data['Average Price'].kurtosis())

        # 计算不同品种的平均价格
        average_prices = self.processed_data.groupby('Variety')[['Average Price']].mean()
        print("\n不同品种的平均价格：")
        print(average_prices)

        # 计算不同产地的平均价格
        average_prices_by_origin = self.processed_data.groupby('Origin')[['Average Price']].mean()
        print("\n不同产地的平均价格：")
        print(average_prices_by_origin)

        # 计算不同尺寸的平均价格
        average_prices_by_size = self.processed_data.groupby('Item Size')[['Average Price']].mean()
        print("\n不同尺寸的平均价格：")
        print(average_prices_by_size)

        print("\n总结：")
        print("通过可视化和分析，我们可以看到不同品种、产地和尺寸的南瓜价格分布情况。")
        print("例如，某些品种的南瓜价格波动较大，而某些产地的南瓜价格相对稳定。")
        print("这些信息可以帮助我们更好地理解南瓜市场的价格动态。")