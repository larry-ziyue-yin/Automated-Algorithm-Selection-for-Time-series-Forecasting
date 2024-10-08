import pandas as pd
import tsfel

# 加载 CSV 文件
etth1 = pd.read_csv('ETTh1.csv')
etth2 = pd.read_csv('ETTh2.csv')
ettm1 = pd.read_csv('ETTm1.csv')
ettm2 = pd.read_csv('ETTm2.csv')

# 将'date'列转换为日期格式
etth1['date'] = pd.to_datetime(etth1['date'])
etth2['date'] = pd.to_datetime(etth2['date'])
ettm1['date'] = pd.to_datetime(ettm1['date'])
ettm2['date'] = pd.to_datetime(ettm2['date'])

# 提取日期中的特征
for df in [etth1, etth2, ettm1, ettm2]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['weekday'] = df['date'].dt.weekday  # 提取星期几（0-6）
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)  # 是否为周末

# 去掉原始的日期列，保留其他时间特征和数值列
etth1_values = etth1.drop(columns=['date'])
etth2_values = etth2.drop(columns=['date'])
ettm1_values = ettm1.drop(columns=['date'])
ettm2_values = ettm2.drop(columns=['date'])

# 获取默认的时间序列特征配置
cfg = tsfel.get_features_by_domain()

# 进行特征提取
etth1_features = tsfel.time_series_features_extractor(cfg, etth1_values)
etth2_features = tsfel.time_series_features_extractor(cfg, etth2_values)
ettm1_features = tsfel.time_series_features_extractor(cfg, ettm1_values)
ettm2_features = tsfel.time_series_features_extractor(cfg, ettm2_values)

# 确保四个数据集的特征一致，合并所有特征列
all_columns = list(set(etth1_features.columns).union(set(etth2_features.columns), 
                                                     set(ettm1_features.columns), 
                                                     set(ettm2_features.columns)))

# 将四个数据集的特征对齐，缺失特征填充 NaN
etth1_features = etth1_features.reindex(columns=all_columns, fill_value=pd.NA)
etth2_features = etth2_features.reindex(columns=all_columns, fill_value=pd.NA)
ettm1_features = ettm1_features.reindex(columns=all_columns, fill_value=pd.NA)
ettm2_features = ettm2_features.reindex(columns=all_columns, fill_value=pd.NA)

# 查看提取的特征
print(etth1_features.head())
print(etth2_features.head())
print(ettm1_features.head())
print(ettm2_features.head())

# 保存特征提取结果到CSV文件
etth1_features.to_csv('etth1_features.csv', index=False)
etth2_features.to_csv('etth2_features.csv', index=False)
ettm1_features.to_csv('ettm1_features.csv', index=False)
ettm2_features.to_csv('ettm2_features.csv', index=False)
