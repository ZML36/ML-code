import os
import pandas as pd
from sklearn import preprocessing
import scipy.stats as stats

# 创建输出文件夹
output_folder = '方差分析'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 指定不需要进行正则化的列
exclude_cols = [0, 1]

# 读取 data 文件夹下的文件
datapath = './data'
filenames = []

for modal in os.listdir(datapath):
    for file in os.listdir(os.path.join(datapath, modal)):
        if '训练集' in file or 'train' in file:
            filenames.append(os.path.join(datapath, modal, file))

# 对每个文件进行处理
for filename in filenames:
    # 根据文件扩展名选择读取方法
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(filename)
    else:
        raise ValueError(f"Unsupported file format for file {filename}")
    
    # 保存第1和第2列以便后续使用
    first_two_cols = df.iloc[:, exclude_cols]
    
    # 批量正则化除了第1和第2列之外的所有列
    for i, col in enumerate(df.columns):
        if i in exclude_cols:
            continue
        df[col] = preprocessing.scale(df[col])
    
    # 将数据按照 Group 进行分组
    groups = df.groupby('Group')
    # 执行方差分析
    p_values = []
    f_values = []
    for col in df.columns:
        if col in ['Group'] or col in df.columns[exclude_cols]:
            continue
        f_stat, p_val = stats.f_oneway(*[group[col] for name, group in groups])
        p_values.append(p_val)
        f_values.append(f_stat)
    
    # 将结果存储到 DataFrame 中
    result = pd.DataFrame({'指标名称': df.columns.drop(df.columns[exclude_cols]), 'F值': f_values, 'P值': p_values})
    # 筛选 P 值小于 0.05 的指标
    significant_columns = result.loc[result['P值'] < 0.05, '指标名称']
    
    # 输出 significant_columns 为 Excel 文件，包括原始的第1和第2列
    significant_columns_df = pd.concat([first_two_cols, df[significant_columns]], axis=1)
    significant_columns_df.to_excel(os.path.join(output_folder, f"方差分析矩阵_{os.path.basename(filename)}.xlsx"), index=False)
    
    # 输出 result 为 Excel 文件
    result.to_excel(os.path.join(output_folder, f"result_{os.path.basename(filename)}.xlsx"), index=False)