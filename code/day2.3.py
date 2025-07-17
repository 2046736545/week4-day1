import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime


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
    def __init__(self, X, y, model_name, model, preprocessor, cv=5):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.model = model
        self.preprocessor = preprocessor
        self.cv = cv
        self.results = {
            "model_name": self.get_model_abbreviation(model_name),
            "model_params": self.get_model_params(),
            "fea_encoding": "one-hot",  # 因为使用了OneHotEncoder
        }

    def get_model_abbreviation(self, model_name):
        abbreviations = {
            'LinearRegression': 'LR',
            'Ridge': 'Ridge',
            'Lasso': 'Lasso',
            'RandomForestRegressor': 'RF',
            'XGBRegressor': 'XGB',
            'LGBMRegressor': 'LGBM',
        }
        return abbreviations.get(model_name, model_name)

    def get_model_params(self):
        # 获取模型参数
        params = self.model.get_params()
        # 简化复杂的参数
        if 'estimator' in params:
            params['estimator'] = str(params['estimator'])
        return params

    def calculate_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            "rmse": f"{rmse:.2f}",
            "mae": f"{mae:.2f}",
            "r2": f"{r2:.2f}"
        }

    def run_cross_validation(self):
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        train_rmse_list = []
        train_mae_list = []
        train_r2_list = []

        test_rmse_list = []
        test_mae_list = []
        test_r2_list = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(self.X)):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            # 预处理数据
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)

            # 记录数据形状
            self.results[f"{fold}_fold_train_data"] = list(X_train_processed.shape)
            self.results[f"{fold}_fold_test_data"] = list(X_test_processed.shape)

            # 训练模型
            self.model.fit(X_train_processed, y_train)

            # 预测并评估
            y_train_pred = self.model.predict(X_train_processed)
            y_test_pred = self.model.predict(X_test_processed)

            # 计算训练集性能
            train_performance = self.calculate_metrics(y_train, y_train_pred)
            self.results[f"{fold}_fold_train_performance"] = train_performance

            # 计算测试集性能
            test_performance = self.calculate_metrics(y_test, y_test_pred)
            self.results[f"{fold}_fold_test_performance"] = test_performance

            # 收集性能指标用于计算平均值
            train_rmse_list.append(float(train_performance['rmse']))
            train_mae_list.append(float(train_performance['mae']))
            train_r2_list.append(float(train_performance['r2']))

            test_rmse_list.append(float(test_performance['rmse']))
            test_mae_list.append(float(test_performance['mae']))
            test_r2_list.append(float(test_performance['r2']))

            print(f"{self.model_name} Fold {fold} 训练完成")

        # 计算平均性能
        self.results["average_train_performance"] = {
            "rmse": f"{np.mean(train_rmse_list):.2f}",
            "mae": f"{np.mean(train_mae_list):.2f}",
            "r2": f"{np.mean(train_r2_list):.2f}"
        }

        self.results["average_test_performance"] = {
            "rmse": f"{np.mean(test_rmse_list):.2f}",
            "mae": f"{np.mean(test_mae_list):.2f}",
            "r2": f"{np.mean(test_r2_list):.2f}"
        }

        return self.results


def save_results_to_json(all_results, filename=None):
    if not filename:
        filename = f"实验结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"实验结果已保存至 {filename}")
    return filename


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

    # 准备数据
    X = data_processed.drop(columns=['Average Price'], errors='ignore')
    y = data_processed['Average Price']

    # 定义要评估的模型
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBRegressor': XGBRegressor(n_estimators=100, random_state=42),
        'LGBMRegressor': lgb.LGBMRegressor(
            random_state=42,
            min_child_samples=5,
            min_split_gain=0.01,
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1
        )
    }

    # 运行所有模型的交叉验证
    all_results = []
    for model_name, model in models.items():
        print(f"\n开始{model_name}模型的交叉验证...")
        trainer = ModelTrainer(X, y, model_name, model, preprocessor_obj, cv=5)
        model_results = trainer.run_cross_validation()
        all_results.append(model_results)

    # 保存结果到JSON文件
    save_results_to_json(all_results)

    # 打印最终结果摘要
    print("\n===== 实验结果摘要 =====")
    for result in all_results:
        print(f"\n模型: {result['model_name']}")
        print(f"训练集平均性能: RMSE={result['average_train_performance']['rmse']}, "
              f"MAE={result['average_train_performance']['mae']}, "
              f"R²={result['average_train_performance']['r2']}")
        print(f"测试集平均性能: RMSE={result['average_test_performance']['rmse']}, "
              f"MAE={result['average_test_performance']['mae']}, "
              f"R²={result['average_test_performance']['r2']}")


if __name__ == "__main__":
    main()