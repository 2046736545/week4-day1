import pandas as pd

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        # 日期解析
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%m/%d/%y', errors='coerce')
        self.data = self.data.dropna(subset=['Date'])

        # 衍生时间特征
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Season'] = self.data['Month'].apply(
            lambda x: 'Spring' if 3 <= x <= 5 else 'Summer' if 6 <= x <= 8 else 'Autumn' if 9 <= x <= 11 else 'Winter'
        )
        self.data['Day'] = self.data['Date'].dt.day
        self.data['DayOfWeek'] = self.data['Date'].dt.dayofweek

        # 计算目标变量（平均价格）
        self.data['Average Price'] = (self.data['Low Price'] + self.data['High Price']) / 2

        # 缺失值填充
        for col in ['Type', 'Item Size', 'Color']:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

        # 高基数类别处理
        for col in ['City Name', 'Origin']:
            top_10_cats = self.data[col].value_counts().head(10).index
            self.data[col] = self.data[col].apply(lambda x: x if x in top_10_cats else 'Other')

        # 删除冗余列
        drop_cols = [
            'Grade', 'Environment', 'Unit of Sale', 'Quality',
            'Condition', 'Appearance', 'Storage', 'Crop',
            'Trans Mode', 'Unnamed: 24', 'Unnamed: 25',
            'Low Price', 'High Price', 'Mostly Low', 'Mostly High',
            'Sub Variety', 'Origin District', 'Repack'
        ]
        self.data = self.data.drop(drop_cols, axis=1, errors='ignore')

        return self.data