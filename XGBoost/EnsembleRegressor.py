import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# 定义模型和数据集列表，仅针对 MSE 结尾的模型
models = [
    'GPHT_MSE', 'GPHT\'_MSE', 'Self-supervised_PatchTST_MSE', 'FPT_MSE', 
    'SimMTM_MSE', 'TimeMAE_MSE', 'Supervised_PatchTST_MSE', 
    'iTransformer_MSE', 'TimesNet_MSE', 'DLinear_MSE'
]

# 定义训练集和测试集
train_datasets = [
    #('etth1_features.csv', 'performance/combined_etth1_performance.csv'),
    ('ettm2_features.csv', 'performance/combined_ettm2_performance.csv')
]

test_datasets = [
    #('etth2_features.csv', 'performance/combined_etth2_performance.csv'),
    ('ettm1_features.csv', 'performance/combined_ettm1_performance.csv')
]

# 保存结果的列表
results = []

# 保存每个模型的 AS Model MSE 和 Actual MSE 值，用于计算 Spearman 相关性
as_mse_list = []
actual_mse_list = []

# 1. 训练AS模型
for model in models:
    print(f"\nTraining model: {model}")
    
    X_train_total = pd.DataFrame()
    y_train_total = pd.Series()

    for feature_file, performance_file in train_datasets:
        print(f"Processing training dataset: {feature_file}")
        
        # 加载特征数据和性能数据
        features = pd.read_csv(feature_file)
        performance = pd.read_csv(performance_file)
        
        # 清理性能数据的列名（去除空格）
        performance.columns = performance.columns.str.strip()
        
        # 如果模型在性能数据中存在，继续运行；如果不存在，跳过该模型
        if model not in performance.columns:
            print(f"Model {model} not found in {performance_file}. Skipping.")
            continue
        
        # 合并特征和目标数据
        data = pd.concat([features, performance[[model]]], axis=1)
        
        # 处理NaN值
        data = data.fillna(method='ffill').fillna(method='bfill')  # 前向和后向填充
        data = data.fillna(data.mean())  # 均值填充剩余的 NaN
        data = data.dropna()  # 删除包含 NaN 的行
        
        # 如果数据为空，跳过这个数据集
        if data.shape[0] == 0:
            print(f"Dataset {feature_file} is empty after processing. Skipping.")
            continue
        
        X_train_total = pd.concat([X_train_total, data.drop(columns=[model])], axis=0)
        y_train_total = pd.concat([y_train_total, data[model]], axis=0)

    # 使用合并的训练数据训练 Ensemble Regressor 模型
    ensemble_model = VotingRegressor([
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)),
        ('dt', DecisionTreeRegressor(max_depth=5, random_state=42))
    ])
    
    ensemble_model.fit(X_train_total, y_train_total)

    # 2. 用测试集评估AS模型
    for feature_file, performance_file in test_datasets:
        print(f"Processing testing dataset: {feature_file}")
        
        # 加载测试集特征数据和性能数据
        features = pd.read_csv(feature_file)
        performance = pd.read_csv(performance_file)
        
        # 清理性能数据的列名（去除空格）
        performance.columns = performance.columns.str.strip()
        
        # 如果模型在性能数据中存在，继续运行；如果不存在，跳过该模型
        if model not in performance.columns:
            print(f"Model {model} not found in {performance_file}. Skipping.")
            continue
        
        # 合并特征和目标数据
        data = pd.concat([features, performance[[model]]], axis=1)
        
        # 处理NaN值
        data = data.fillna(method='ffill').fillna(method='bfill')  # 前向和后向填充
        data = data.fillna(data.mean())  # 均值填充剩余的 NaN
        data = data.dropna()  # 删除包含 NaN 的行
        
        # 如果数据为空，跳过这个数据集
        if data.shape[0] == 0:
            print(f"Dataset {feature_file} is empty after processing. Skipping.")
            continue

        # ------------ 获取实际的 MSE 值 ------------ 
        actual_mse = performance[model].mean()  # 获取该列的平均值，代表实际的MSE
        # ------------------------------------------
        
        X_test = data.drop(columns=[model])
        y_test = data[model]

        # 用训练好的 Ensemble Regressor 模型进行预测
        y_pred = ensemble_model.predict(X_test)

        # 计算 AS 模型的预测 MSE
        mse = mean_squared_error(y_test, y_pred)
        print(f"AS Model Test MSE for {feature_file} with model {model}: {y_pred.mean()}")
        print(f"Actual MSE for {model}: {actual_mse}")
        
        # 保存每个模型的 AS Model MSE 和 Actual MSE
        as_mse_list.append(y_pred.mean())
        actual_mse_list.append(actual_mse)

        # 保存结果
        results.append({
            'Dataset': feature_file,
            'Model': model,
            'AS Model Test MSE': y_pred.mean(),  # 预测 MSE
            'Actual MSE': actual_mse   # 实际 MSE
        })

# 排名计算
as_rank = pd.Series(as_mse_list).rank().tolist()
actual_rank = pd.Series(actual_mse_list).rank().tolist()

# 计算 Spearman 相关系数
spearman_corr, _ = spearmanr(as_rank, actual_rank)

# 将排名添加到结果
for i in range(len(results)):
    results[i]['AS Model Rank'] = as_rank[i]
    results[i]['Actual Model Rank'] = actual_rank[i]
    results[i]['Spearman Correlation'] = spearman_corr

# 按照 actual_rank 排序后的索引
sorted_indices = sorted(range(len(actual_rank)), key=lambda k: actual_rank[k])

# 按照 actual_rank 排序后的 as_rank
as_rank_sorted_by_actual = [as_rank[i] for i in sorted_indices]
print(f"AS Rank sorted by Actual Rank: {as_rank_sorted_by_actual}")

# 输出结果
result_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(result_df)

# 将结果保存为 CSV 文件
result_df.to_csv('ensemble_regressor_mse_results_with_spearman_rank.csv', index=False)
