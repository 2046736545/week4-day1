import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()  # 避免修改原始数据

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
        self.data['DayOfWeek'] = self.data['Date'].dt.dayofweek  # 0=周一，6=周日

        # 计算目标变量（平均价格）
        self.data['Average Price'] = (self.data['Low Price'] + self.data['High Price']) / 2

        # 新增：目标变量异常值处理（IQR法）
        price = self.data['Average Price']
        q1 = price.quantile(0.25)
        q3 = price.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        self.data = self.data[(price >= lower_bound) & (price <= upper_bound)]  # 过滤极端值

        # 缺失值填充
        for col in ['Type', 'Item Size', 'Color', 'Variety']:  # 补充Variety缺失值处理
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

        # 高基数类别处理
        for col in ['City Name', 'Origin']:
            top_10_cats = self.data[col].value_counts().head(10).index
            self.data[col] = self.data[col].apply(lambda x: x if x in top_10_cats else 'Other')

        # 调整：保留可能影响价格的关键特征（移除Quality、Condition等）
        drop_cols = [
            'Grade', 'Environment', 'Unit of Sale',
            'Appearance', 'Storage', 'Crop',
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
    def __init__(self, X_train, X_test, y_train, y_test, X_test_raw):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_test_raw = X_test_raw  # 保留原始测试特征用于样本分析

    def train_and_evaluate(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, r2, y_pred  # 新增返回预测值

    def improved_training(self):
        results = {}
        best_model = None
        best_r2 = -np.inf
        y_pred_best = None

        # 线性回归
        print("开始训练线性回归模型...")
        lr_model = LinearRegression()
        lr_mse, lr_r2, lr_pred = self.train_and_evaluate(lr_model)
        results['Linear Regression'] = {'MSE': lr_mse, 'R²': lr_r2}
        if lr_r2 > best_r2:
            best_r2 = lr_r2
            best_model = 'Linear Regression'
            y_pred_best = lr_pred

        # 岭回归
        print("开始训练岭回归模型...")
        ridge_model = Ridge()
        param_grid_ridge = {'alpha': [0.1, 1, 10]}
        grid_search_ridge = GridSearchCV(estimator=ridge_model, param_grid=param_grid_ridge, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_ridge.fit(self.X_train, self.y_train)
        best_ridge_model = grid_search_ridge.best_estimator_
        ridge_mse, ridge_r2, ridge_pred = self.train_and_evaluate(best_ridge_model)
        results['Ridge Regression'] = {'MSE': ridge_mse, 'R²': ridge_r2}
        if ridge_r2 > best_r2:
            best_r2 = ridge_r2
            best_model = 'Ridge Regression'
            y_pred_best = ridge_pred

        # Lasso回归
        print("开始训练Lasso回归模型...")
        lasso_model = Lasso()
        param_grid_lasso = {'alpha': [0.1, 1, 10]}
        grid_search_lasso = GridSearchCV(estimator=lasso_model, param_grid=param_grid_lasso, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_lasso.fit(self.X_train, self.y_train)
        best_lasso_model = grid_search_lasso.best_estimator_
        lasso_mse, lasso_r2, lasso_pred = self.train_and_evaluate(best_lasso_model)
        results['Lasso Regression'] = {'MSE': lasso_mse, 'R²': lasso_r2}
        if lasso_r2 > best_r2:
            best_r2 = lasso_r2
            best_model = 'Lasso Regression'
            y_pred_best = lasso_pred

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
        rf_mse, rf_r2, rf_pred = self.train_and_evaluate(best_rf_model)
        results['Random Forest'] = {'MSE': rf_mse, 'R²': rf_r2}
        if rf_r2 > best_r2:
            best_r2 = rf_r2
            best_model = 'Random Forest'
            y_pred_best = rf_pred

        # XGBoost回归
        print("开始训练XGBoost回归模型...")
        xgb_model = XGBRegressor(random_state=42)
        param_grid_xgb = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'lambda': [0.1, 1, 10]
        }
        grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5,
                                       scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_xgb.fit(self.X_train, self.y_train)
        best_xgb_model = grid_search_xgb.best_estimator_
        xgb_mse, xgb_r2, xgb_pred = self.train_and_evaluate(best_xgb_model)
        results['XGBoost'] = {'MSE': xgb_mse, 'R²': xgb_r2}
        if xgb_r2 > best_r2:
            best_r2 = xgb_r2
            best_model = 'XGBoost'
            y_pred_best = xgb_pred

        # 支持向量机回归
        print("开始训练支持向量机回归模型...")
        svr_model = SVR()
        param_grid_svr = {
            'C': [0.1, 1, 10],
            'epsilon': [0.1, 0.2, 0.3],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
        grid_search_svr = GridSearchCV(estimator=svr_model, param_grid=param_grid_svr, cv=5,
                                       scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_svr.fit(self.X_train, self.y_train)
        best_svr_model = grid_search_svr.best_estimator_
        svr_mse, svr_r2, svr_pred = self.train_and_evaluate(best_svr_model)
        results['SVR'] = {'MSE': svr_mse, 'R²': svr_r2}
        if svr_r2 > best_r2:
            best_r2 = svr_r2
            best_model = 'SVR'
            y_pred_best = svr_pred

        # 绘制模型对比图
        models = list(results.keys())
        mse_values = [results[model]['MSE'] for model in models]
        r2_values = [results[model]['R²'] for model in models]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(models, mse_values, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Model Comparison - MSE')
        plt.xticks(rotation=45)
        plt.subplot(1, 2, 2)
        plt.bar(models, r2_values, color='lightgreen')
        plt.xlabel('Models')
        plt.ylabel('R-squared (R²)')
        plt.title('Model Comparison - R²')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # 生成样本分析表（参考老师示例）
        self.generate_sample_analysis(y_pred_best)

        return results

    def generate_sample_analysis(self, y_pred):
        """生成老师示例格式的样本分析表"""
        # 构建结果表
        analysis_df = self.X_test_raw[
            ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Year', 'Month', 'DayOfWeek']].copy()
        analysis_df['test y'] = self.y_test.values
        analysis_df['test y_predict'] = y_pred
        analysis_df['delta'] = analysis_df['test y_predict'] - analysis_df['test y']
        analysis_df = analysis_df.reset_index().rename(columns={'index': '样本编号', 'DayOfWeek': 'weekday'})

        # 筛选4个样本（2正确+2错误）
        analysis_df['abs_delta'] = analysis_df['delta'].abs()
        correct_samples = analysis_df.nsmallest(2, 'abs_delta')  # 误差最小的2个
        wrong_samples = analysis_df.nlargest(2, 'abs_delta')  # 误差最大的2个
        final_analysis = pd.concat([correct_samples, wrong_samples], ignore_index=True)

        # 输出老师示例格式的表格
        print("\n===== 样本分析表 =====")
        print(final_analysis[['样本编号', 'City Name', 'Package', 'Variety', 'Origin', 'Item Size',
                              'Year', 'Month', 'weekday', 'test y', 'test y_predict', 'delta']].to_string(index=False))

        # 打印分析说明
        print("\n===== 样本分析说明 =====")
        for idx, row in final_analysis.iterrows():
            print(f"\n样本{row['样本编号']}：")
            print(f"特征：{row['City Name']}，{row['Package']}，{row['Variety']}，{row['Origin']}，{row['Item Size']}")
            print(f"实际价格：{row['test y']}，预测价格：{row['test y_predict']}，误差：{row['delta']}")
            if row['abs_delta'] <= 5:  # 定义误差≤5为正确
                print("预测正确原因：该特征组合在训练集中样本充足，价格分布稳定，模型准确捕捉规律。")
            else:
                print("预测错误原因：训练集中同特征组合样本少或价格异常，模型泛化能力不足。")


class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def visualize(self):
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data['Average Price'], bins=20, alpha=0.5, label='Average Price', kde=True)
        plt.title('Average Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Variety', y='Average Price', data=self.data)
        plt.title('Average Price by Variety')
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Variety', y='Average Price', data=self.data)
        plt.title('Average Price by Variety')
        plt.xticks(rotation=45)
        plt.show()


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
        print(f"平均价格偏度：{self.processed_data['Average Price'].skew()}")
        print(f"平均价格峰度：{self.processed_data['Average Price'].kurtosis()}")

        # 分组统计
        print("\n不同品种的平均价格：")
        print(self.processed_data.groupby('Variety')['Average Price'].mean().round(2))
        print("\n不同产地的平均价格：")
        print(self.processed_data.groupby('Origin')['Average Price'].mean().round(2))
        print("\n不同尺寸的平均价格：")
        print(self.processed_data.groupby('Item Size')['Average Price'].mean().round(2))


def main():
    # 加载数据
    print("开始加载数据...")
    data = pd.read_csv('US-pumpkins.csv')  # 确保该文件在当前目录下
    print("原始数据前5行：")
    print(data.head())

    # 数据预处理
    print("\n开始数据预处理...")
    preprocessor = DataPreprocessor(data)
    data_processed = preprocessor.preprocess()
    print("预处理后数据前5行：")
    print(data_processed.head())

    # 特征工程
    print("\n开始特征工程...")
    engineer = FeatureEngineer(data_processed)
    preprocessor_obj, _, _ = engineer.engineer()

    # 数据划分（保留原始测试集特征用于分析）
    print("\n开始数据划分...")
    X = data_processed.drop(columns=['Average Price', 'Date'], errors='ignore')  # 移除Date
    y = data_processed['Average Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 预处理数据
    print("\n开始预处理数据...")
    X_train_processed = preprocessor_obj.fit_transform(X_train)
    X_test_processed = preprocessor_obj.transform(X_test)
    print(f"训练集形状：{X_train_processed.shape}，测试集形状：{X_test_processed.shape}")

    # 模型训练与样本分析
    print("\n开始模型训练与评估...")
    trainer = ModelTrainer(X_train_processed, X_test_processed, y_train, y_test, X_test)  # 传入原始X_test
    results = trainer.improved_training()

    # 数据可视化
    print("\n开始数据可视化...")
    visualizer = DataVisualizer(data_processed)
    visualizer.visualize()

    # 数据探索性分析
    print("\n开始探索性分析...")
    analyzer = DataAnalyzer(data, data_processed)
    analyzer.analyze()

    return results


if __name__ == "__main__":
    results = main()