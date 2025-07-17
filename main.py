import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from data_visualizer import DataVisualizer
from data_analyzer import DataAnalyzer

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

    # 数据可视化
    visualizer = DataVisualizer(data_processed)
    visualizer.visualize()

    # 数据探索性分析
    analyzer = DataAnalyzer(data, data_processed)
    analyzer.analyze()

    return results

if __name__ == "__main__":
    results = main()