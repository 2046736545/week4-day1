import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt

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

        # 线性回归
        print("开始训练线性回归模型...")
        lr_model = LinearRegression()
        lr_mse, lr_r2 = self.train_and_evaluate(lr_model)
        results['Linear Regression'] = {'MSE': lr_mse, 'R²': lr_r2}
        print(f"线性回归的均方误差（MSE）: {lr_mse}")
        print(f"线性回归的决定系数（R²）: {lr_r2}")

        # 岭回归
        print("开始训练岭回归模型...")
        ridge_model = Ridge()
        param_grid_ridge = {'alpha': [0.1, 1, 10]}
        grid_search_ridge = GridSearchCV(estimator=ridge_model, param_grid=param_grid_ridge, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_ridge.fit(self.X_train, self.y_train)
        best_ridge_model = grid_search_ridge.best_estimator_
        ridge_mse, ridge_r2 = self.train_and_evaluate(best_ridge_model)
        results['Ridge Regression'] = {'MSE': ridge_mse, 'R²': ridge_r2}
        print(f"优化后的岭回归的均方误差（MSE）: {ridge_mse}")
        print(f"优化后的岭回归的决定系数（R²）: {ridge_r2}")

        # Lasso回归
        print("开始训练Lasso回归模型...")
        lasso_model = Lasso()
        param_grid_lasso = {'alpha': [0.1, 1, 10]}
        grid_search_lasso = GridSearchCV(estimator=lasso_model, param_grid=param_grid_lasso, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_lasso.fit(self.X_train, self.y_train)
        best_lasso_model = grid_search_lasso.best_estimator_
        lasso_mse, lasso_r2 = self.train_and_evaluate(best_lasso_model)
        results['Lasso Regression'] = {'MSE': lasso_mse, 'R²': lasso_r2}
        print(f"优化后的Lasso回归的均方误差（MSE）: {lasso_mse}")
        print(f"优化后的Lasso回归的决定系数（R²）: {lasso_r2}")

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
        xgb_mse, xgb_r2 = self.train_and_evaluate(best_xgb_model)
        results['XGBoost'] = {'MSE': xgb_mse, 'R²': xgb_r2}
        print(f"优化后的XGBoost回归的均方误差（MSE）: {xgb_mse}")
        print(f"优化后的XGBoost回归的决定系数（R²）: {xgb_r2}")

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
        svr_mse, svr_r2 = self.train_and_evaluate(best_svr_model)
        results['SVR'] = {'MSE': svr_mse, 'R²': svr_r2}
        print(f"优化后的支持向量机回归的均方误差（MSE）: {svr_mse}")
        print(f"优化后的支持向量机回归的决定系数（R²）: {svr_r2}")

        # 绘制柱状图比较模型性能
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

        return results