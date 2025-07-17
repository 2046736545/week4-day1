import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_text

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


class FeatureEngineer:
    def __init__(self, data):
        self.data = data

    def engineer(self):
        numerical_features = ['Month', 'Year', 'Day', 'DayOfWeek']
        categorical_features = [
            'City Name', 'Type', 'Package', 'Variety',
            'Origin', 'Item Size', 'Color', 'Season'
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        return preprocessor, numerical_features, categorical_features


class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_and_evaluate(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, r2

    def improved_training(self):
        results = {}

        # 随机森林回归
        print("开始训练随机森林回归模型...")
        rf_model = RandomForestRegressor(random_state=42)
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5,
                                      scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_rf.fit(self.X_train, self.y_train)
        best_rf_model = grid_search_rf.best_estimator_
        rf_mse, rf_r2 = self.train_and_evaluate(best_rf_model)
        results['Random Forest'] = {'MSE': rf_mse, 'R²': rf_r2}
        print(f"优化后的随机森林回归的均方误差（MSE）: {rf_mse}")
        print(f"优化后的随机森林回归的决定系数（R²）: {rf_r2}")

        # 获取随机森林中某一棵树的树结构
        tree_index = 0  # 选择第一棵树
        tree = best_rf_model.estimators_[tree_index]
        feature_names = [f"f{i}" for i in range(self.X_train.shape[1])]
        tree_structure = export_text(tree, feature_names=feature_names)

        # 将树结构保存到txt文件
        with open('tree_structure.txt', 'w') as file:
            file.write(tree_structure)

        print("随机森林中第一棵树的树结构已保存到 tree_structure.txt 文件中。")

        # 解释一条从树根到叶子节点的分支含义
        branch = tree_structure.split('\n')
        path = []
        for line in branch:
            if line.strip():
                path.append(line)
        print("\n从树根到叶子节点的一条分支：")
        for step in path:
            print(step)
        print("\n该分支的含义：")
        print("该分支描述了样本从根节点开始，根据不同特征的取值进行分裂，最终到达叶子节点的过程。每个节点的分裂条件表示样本在该特征上的取值需要满足的条件，沿着分支的路径可以看到样本是如何逐步被分类或回归到最终结果的。")

        return results


def main():
    # 加载数据
    print("开始加载数据...")
    data = pd.read_csv('US-pumpkins.csv')
    print(data.head())

    # 数据预处理
    print("开始数据预处理...")
    preprocessor = DataPreprocessor(data)
    data_processed = preprocessor.preprocess()
    print(data_processed.head())

    # 特征工程
    print("开始特征工程...")
    engineer = FeatureEngineer(data_processed)
    preprocessor_obj, _, _ = engineer.engineer()

    # 数据划分
    print("开始数据划分...")
    X = data_processed.drop(columns=['Average Price'], errors='ignore')
    y = data_processed['Average Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 预处理数据
    print("开始预处理数据...")
    X_train_processed = preprocessor_obj.fit_transform(X_train)
    X_test_processed = preprocessor_obj.transform(X_test)
    print(X_train_processed.shape)
    print(X_test_processed.shape)

    # 改进后的模型训练和评估
    print("开始改进后的模型训练和评估...")
    trainer = ModelTrainer(X_train_processed, X_test_processed, y_train, y_test)
    results = trainer.improved_training()

    return results


if __name__ == "__main__":
    results = main()