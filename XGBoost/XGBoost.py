import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 定义模型和数据集列表，仅针对 MSE 结尾的模型
models = [
    'GPHT_MSE', 'GPHT\'_MSE', 'Self-supervised_PatchTST_MSE', 'FPT_MSE', 
    'SimMTM_MSE', 'TimeMAE_MSE', 'Supervised_PatchTST_MSE', 
    'iTransformer_MSE', 'TimesNet_MSE', 'DLinear_MSE'
]
datasets = [
    ('etth1_features.csv', 'performance/combined_etth1_performance.csv'),
    #('etth2_features.csv', 'performance/combined_etth2_performance.csv'),
    #('ettm1_features.csv', 'performance/combined_ettm1_performance.csv'),
    #('ettm2_features.csv', 'performance/combined_ettm2_performance.csv')
]

# 保存结果的列表
results = []

# 循环处理不同模型和数据集
for model in models:
    print(f"\nRunning tests for model: {model}")
    
    for feature_file, performance_file in datasets:
        print(f"Processing dataset: {feature_file}")
        
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
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=[model]), 
                                                            data[model], 
                                                            test_size=0.2, random_state=42)

        # XGBoost模型
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)

        # 训练模型
        xgb_model.fit(X_train, y_train)

        # 预测
        y_pred = xgb_model.predict(X_test)

        # 评估模型
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE for {feature_file} with model {model}: {mse}")
        
        # 保存结果
        results.append({
            'Dataset': feature_file,
            'Model': model,
            'Test MSE': mse
        })

# 输出结果
result_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(result_df)

# 将结果保存为 CSV 文件
# result_df.to_csv('xgboost_mse_results.csv', index=False)
