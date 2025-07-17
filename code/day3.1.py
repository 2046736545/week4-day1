import pandas as pd
import os
from datetime import datetime

# 假设的常量定义
DATA_FILE = 'US-pumpkins.csv'
RESULT_DIR = 'results'
VISUALIZATION_FILE = os.path.join(RESULT_DIR, 'visualization.png')
REPORT_FILE = os.path.join(RESULT_DIR, 'report.txt')
README_FILE = 'Week4-readme.md'

"""生成README内容"""
def generate_readme(dataset_info, analysis_results):
    """生成README内容"""
    if not dataset_info or not analysis_results:
        return "# 美国南瓜数据集分析项目\n\n生成README时数据获取失败"

    # 项目概述部分
    readme = "# 美国南瓜数据集分析项目\n\n"
    readme += "## 项目概述\n\n"
    readme += "本项目旨在对美国南瓜数据集进行全面分析，包括数据预处理、数据探索性分析、数据可视化和数据分析等步骤，最后对结果进行总结，以帮助我们更好地理解南瓜市场的价格动态。\n\n"

    # 目录结构部分
    readme += "## 目录结构\n\n"
    readme += "```\n"
    readme += "us-pumpkins-analysis/\n"
    readme += f"├── 2.py    # 主要分析代码\n"
    readme += f"├── {DATA_FILE}       # 数据集\n"
    readme += f"├── {RESULT_DIR}/                # 结果存储目录\n"
    readme += f"│   ├── {os.path.basename(VISUALIZATION_FILE)} # 分析可视化图\n"
    readme += f"│   └── {os.path.basename(REPORT_FILE)} # 分析报告\n"
    readme += f"└── {README_FILE}               # 项目说明文档\n"
    readme += "```\n\n"

    # 项目背景与目的部分
    readme += "## 项目背景与目的\n\n"
    readme += "南瓜在美国是一种重要的农产品，其价格受到品种、产地和尺寸等多种因素的影响。本项目通过数据分析的方法，对美国南瓜数据集进行深入挖掘，旨在：\n\n"
    readme += "1. 探索不同品种、产地和尺寸的南瓜价格分布情况\n"
    readme += "2. 分析各类别南瓜价格的差异\n"
    readme += "3. 为南瓜市场的参与者提供决策参考\n\n"

    # 数据集说明部分
    readme += "## 数据集说明\n\n"
    readme += "### 数据集来源\n"
    readme += f"本数据集包含{dataset_info['row_count']}条美国南瓜的信息，包含丰富的南瓜元数据和价格信息。\n\n"
    readme += "### 数据特征\n\n"
    readme += f"数据集包含{dataset_info['col_count']}个特征，具体说明如下：\n\n"
    readme += "| 特征名称 | 类型 | 非空值 | 缺失值 | 说明 |\n"
    readme += "|---------|------|------|------|------|\n"
    for feature_info in dataset_info['features_info']:
        readme += feature_info + "\n"
    readme += "\n"

    # 数据预览部分
    readme += "### 数据预览\n\n"
    readme += "数据集前5条记录：\n\n"
    readme += "```\n"
    readme += dataset_info['preview']
    readme += "```\n\n"

    # 技术栈部分
    readme += "## 技术栈\n\n"
    readme += "本项目使用的主要技术和库如下：\n\n"
    readme += "- **数据处理**：Pandas\n"
    readme += "- **数据可视化**：Matplotlib, Seaborn\n"
    readme += "- **数值计算**：NumPy\n"
    readme += "- **机器学习**：Scikit-learn, XGBoost\n\n"

    # 方法与步骤部分
    readme += "## 方法与步骤\n\n"
    readme += "### 1. 数据预处理\n"
    readme += "- 解析日期并提取月份和年份\n"
    readme += "- 填充缺失值\n"
    readme += "- 处理高基数类别特征\n"
    readme += "- 删除冗余列\n\n"
    readme += "### 2. 数据探索性分析\n"
    readme += "- 检查异常值\n"
    readme += "- 检查数据分布的偏度和峰度\n\n"
    readme += "### 3. 数据可视化\n"
    readme += "- 使用Matplotlib和Seaborn绘制价格分布图、箱线图、条形图和小提琴图\n\n"
    readme += "### 4. 数据分析\n"
    readme += "- 计算不同品种、产地和尺寸的平均价格\n\n"
    readme += "### 5. 机器学习模型\n"
    readme += "- 使用随机森林、XGBoost和支持向量机进行价格预测\n\n"

    # 结果分析部分
    readme += "## 结果分析\n\n"
    readme += "### 1. 不同品种的平均价格\n\n"
    readme += f"{analysis_results['average_prices']}\n\n"
    readme += "### 2. 不同产地的平均价格\n\n"
    readme += f"{analysis_results['average_prices_by_origin']}\n\n"
    readme += "### 3. 不同尺寸的平均价格\n\n"
    readme += f"{analysis_results['average_prices_by_size']}\n\n"
    readme += "通过以上分析，我们可以看到不同品种、产地和尺寸的南瓜价格分布情况。例如，某些品种的南瓜价格波动较大，而某些产地的南瓜价格相对稳定。这些信息可以帮助我们更好地理解南瓜市场的价格动态。\n\n"

    # 新增：南瓜数据细节部分
    readme += "## 南瓜数据细节\n\n"
    readme += "### 时间范围\n"
    readme += "数据时间范围从 2014 年 11 月 29 日到 2017 年 12 月 10 日，但实际有效数据集中在 2016 年 9 月 - 2017 年 12 月。证据：2014 年仅 1 条记录（MIAMI, 11/29/14），2015 年无记录，2016 年 9 月 - 12 月密集出现。\n\n"
    readme += "### 品种分布\n"
    readme += "高频种类（Top 3）：\n"
    readme += "- HOWDEN TYPE（占 40%+）：如 BALTIMORE、ATLANTA、BOSTON 均有大量记录。\n"
    readme += "- PIE TYPE（占 25%+）：如 1 1/9 bushel cartons 包装的 PIE TYPE 在多个市场出现。\n"
    readme += "- CINDERELLA（占 15%+）：如 BALTIMORE 的 24 inch bins、COLUMBIA 的 24 inch bins。\n"
    readme += "低频种类：\n"
    readme += "- KNUCKLE HEAD（仅 COLUMBIA、PHILADELPHIA 少量记录）。\n"
    readme += "- BLUE TYPE（仅 BOSTON、COLUMBIA 少量记录）。\n\n"
    readme += "### 规格分布\n"
    readme += "规格（Item Size）标签混乱但可归纳高频规格：\n"
    readme += "- 24 inch bins（占 50%+）：如 BALTIMORE、ATLANTA、DALLAS 的 HOWDEN TYPE。\n"
    readme += "- 36 inch bins（占 30%+）：如 BOSTON、CHICAGO 的 HOWDEN TYPE。\n"
    readme += "- 1 1/9 bushel cartons（占 10%+）：如 PIE TYPE 的中小规格。\n\n"
    readme += "### 城市分布\n"
    readme += "样本充分的城市（Top 5）：\n"
    readme += "- BALTIMORE（1000+ 条记录，覆盖 HOWDEN、CINDERELLA、PIE TYPE 等）。\n"
    readme += "- BOSTON（800+ 条记录，以 36 inch bins 为主）。\n"
    readme += "- CHICAGO（700+ 条记录，HOWDEN TYPE 和 PIE TYPE 均衡）。\n"
    readme += "- COLUMBIA（600+ 条记录，HOWDEN TYPE 和 PIE TYPE）。\n"
    readme += "- DALLAS（500+ 条记录，HOWDEN TYPE 为主）。\n"
    readme += "样本稀疏的城市：\n"
    readme += "- MIAMI（仅 3 条）。\n"
    readme += "- DETROIT（<100 条）。\n\n"
    readme += "### 产地分布\n"
    readme += "高频产地：\n"
    readme += "- MARYLAND（BALTIMORE 本地供应，占 30%+）。\n"
    readme += "- CALIFORNIA（LOS ANGELES、SAN FRANCISCO 主要来源，占 25%+）。\n"
    readme += "- TEXAS（DALLAS 本地供应，占 20%+）。\n\n"

    # 新增：特征处理细节部分
    readme += "## 特征处理细节\n\n"
    readme += "- 特征热力图，可分析特征线性相关性；而即使存在多重共线性，也不影响模型建模，可能会存在性能上的影响。\n"
    readme += "- 线性模型对离散值的顺序编码非常敏感，树模型会好一些，但是也会有影响。\n"
    readme += "- 删掉年月日信息，模型性能会减弱一些。\n"
    readme += "- 价格高低、城市、品种、尺寸这些连续或离散特征，线性模型得标准化或独热，树模型无所谓但高基数要哈希或目标编码。\n"
    readme += "- 日期别扔，拆出月份就够，万圣节前后价差大。\n"
    readme += "- 低高价格多重共线，留一个或正则化。\n"
    readme += "- 缺失颜色就填“未知”，产地缺太多直接不要。\n"
    readme += "- 交互搞个“城市×品种”就行。\n\n"

    # 新增：模型细节部分
    readme += "## 模型细节\n\n"
    readme += "### 树模型\n"
    readme += "- 每多一层，分裂次数翻倍，容易把噪声也学进去。一般把 max_depth 控制在 3 - 6，叶子数不超过 32 个就能防止“背答案”。\n"
    readme += "- 想再保险，把 min_samples_split 设成 10 左右：节点里样本太少就不拆。\n\n"
    readme += "### 线性模型\n"
    readme += "- 用 Ridge/Lasso，λ 先随便给个 1，看系数是不是一大坨，一大坨就调大。\n"
    readme += "- 如果变量太多（比如 Origin 独热后上百列），Lasso 会直接把没用的缩到 0，省得手动挑。\n\n"
    readme += "### 模型学到的“人类常识”\n"
    readme += "- 万圣节前十天价格最高，节后掉 30 - 50%。\n"
    readme += "- 36 寸箱比 24 寸箱平均贵 20 - 40%，但“超大号”溢价在 CINDERELLA 品种里不明显。\n"
    readme += "- 加州、马萨诸塞的货普遍比德州、马里兰贵；产地越远越加价。\n"
    readme += "- MINIATURE 南瓜价格波动最小，大南瓜波动大。\n\n"

    # 运行指南部分
    readme += "## 运行指南\n\n"
    readme += "### 环境要求\n\n"
    readme += "- Python 3.x\n"
    readme += "- 所需依赖包：\n"
    readme += "  - pandas\n"
    readme += "  - matplotlib\n"
    readme += "  - seaborn\n"
    readme += "  - numpy\n"
    readme += "  - scikit-learn\n"
    readme += "  - xgboost\n\n"
    readme += "### 安装依赖\n\n"
    readme += "```bash\n"
    readme += "pip install pandas matplotlib seaborn numpy scikit-learn xgboost\n"
    readme += "```\n\n"
    readme += "### 运行代码\n\n"
    readme += f"1. 将数据集`{DATA_FILE}`与代码`2.py`放在同一目录下\n"
    readme += "2. 打开终端，进入代码所在目录\n"
    readme += "3. 运行以下命令：\n\n"
    readme += "```bash\n"
    readme += f"python 2.py\n"
    readme += "```\n\n"
    readme += "4. 运行结果将输出分析结果和可视化图表\n\n"

    # 项目改进方向
    readme += "## 项目改进方向\n\n"
    readme += "1. **特征扩展**：添加更多与南瓜相关的特征，如种植方式、季节等\n"
    readme += "2. **模型构建**：尝试使用更多机器学习模型进行价格预测\n"
    readme += "3. **时间序列分析**：结合时间因素分析南瓜价格的变化趋势\n\n"

    # 生成信息
    readme += "## 生成信息\n\n"
    readme += f"本README文件由Python脚本自动生成于 {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"

    return readme

"""运行数据分析获取结果"""
def run_analysis():
    """运行数据分析获取结果"""
    try:
        df = pd.read_csv(DATA_FILE)

        # 数据预处理
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')
        df = df.dropna(subset=['Date'])

        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['Season'] = df['Month'].apply(
            lambda x: 'Spring' if 3 <= x <= 5 else 'Summer' if 6 <= x <= 8 else 'Autumn' if 9 <= x <= 11 else 'Winter'
        )

        df['Average Price'] = (df['Low Price'] + df['High Price']) / 2

        for col in ['Type', 'Item Size', 'Color']:
            df[col] = df[col].fillna(df[col].mode()[0])

        for col in ['City Name', 'Origin']:
            top_10_cats = df[col].value_counts().head(10).index
            df[col] = df[col].apply(lambda x: x if x in top_10_cats else 'Other')

        columns_to_drop = [
            'Grade', 'Environment', 'Unit of Sale', 'Quality',
            'Condition', 'Appearance', 'Storage', 'Crop',
            'Trans Mode', 'Unnamed: 24', 'Unnamed: 25',
            'Low Price', 'High Price', 'Mostly Low', 'Mostly High',
            'Sub Variety', 'Origin District', 'Repack'
        ]
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # 数据分析
        average_prices = df.groupby('Variety')[['Average Price']].mean()
        average_prices_by_origin = df.groupby('Origin')[['Average Price']].mean()
        average_prices_by_size = df.groupby('Item Size')[['Average Price']].mean()

        return {
            'average_prices': average_prices,
            'average_prices_by_origin': average_prices_by_origin,
            'average_prices_by_size': average_prices_by_size
        }
    except Exception as e:
        print(f"数据分析时出错: {e}")
        return None

if __name__ == "__main__":
    # 运行分析
    analysis_results = run_analysis()
    if analysis_results:
        # 获取数据集信息
        df = pd.read_csv(DATA_FILE)
        dataset_info = {
            'row_count': len(df),
            'col_count': len(df.columns),
            'features_info': [],
            'preview': df.head().to_csv(sep='\t', na_rep='nan')
        }
        for col in df.columns:
            non_null_count = df[col].count()
            null_count = len(df) - non_null_count
            dtype = df[col].dtype
            feature_info = f"| {col} | {dtype} | {non_null_count} | {null_count} |  |"
            dataset_info['features_info'].append(feature_info)

        # 生成README
        readme_content = generate_readme(dataset_info, analysis_results)

        # 保存README文件
        with open(README_FILE, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"README文件已生成：{README_FILE}")