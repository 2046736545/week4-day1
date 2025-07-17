import pandas as pd
import json

# 读取JSON文件
with open('Fin.json', 'r') as f:
    data = json.load(f)

# 创建一个空列表来存储每行数据
rows = []

# 遍历JSON数据中的每个模型配置
for item in data:
    model_name = item['model_name']
    fea_encoding = item['fea_encoding']
    avg_train_rmse = float(item['average_train_performance']['rmse'])
    avg_train_mae = float(item['average_train_performance']['mae'])
    avg_train_r2 = float(item['average_train_performance']['r2'])
    avg_test_rmse = float(item['average_test_performance']['rmse'])
    avg_test_mae = float(item['average_test_performance']['mae'])
    avg_test_r2 = float(item['average_test_performance']['r2'])

    # 将数据添加到行列表中
    rows.append([model_name, fea_encoding, avg_train_rmse, avg_train_mae, avg_train_r2, avg_test_rmse, avg_test_mae, avg_test_r2])

# 创建DataFrame
df = pd.DataFrame(rows, columns=['model_name', 'fea_encoding', 'average_train_performance (RMSE)', 'average_train_performance (MAE)', 'average_train_performance (R²)', 'average_test_performance (RMSE)', 'average_test_performance (MAE)', 'average_test_performance (R²)'])

# 保存为CSV文件
df.to_csv('your_output.csv', index=False)

print("CSV文件已生成：your_output.csv")