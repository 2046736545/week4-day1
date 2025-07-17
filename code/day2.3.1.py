import json

# 读取 JSON 文件
with open('Fin.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历每个模型
for model in data:
    model_name = model['model_name']
    average_train_performance = model['average_train_performance']
    average_test_performance = model['average_test_performance']

    print(f"模型名称: {model_name}")
    print("平均训练性能:")
    print(f"  RMSE: {average_train_performance['rmse']}")
    print(f"  MAE: {average_train_performance['mae']}")
    print(f"  R2: {average_train_performance['r2']}")
    print("平均测试性能:")
    print(f"  RMSE: {average_test_performance['rmse']}")
    print(f"  MAE: {average_test_performance['mae']}")
    print(f"  R2: {average_test_performance['r2']}")
    print("-" * 50)