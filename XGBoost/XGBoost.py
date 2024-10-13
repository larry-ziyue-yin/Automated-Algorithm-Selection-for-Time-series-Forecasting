import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr


""""
models = [
    'GPHT_MSE', 'GPHT\'_MSE', 'Self-supervised_PatchTST_MSE', 'FPT_MSE', 
    'SimMTM_MSE', 'TimeMAE_MSE', 'Supervised_PatchTST_MSE', 
    'iTransformer_MSE', 'TimesNet_MSE', 'DLinear_MSE'
]
"""
models = [
    "GPHT'_MAE", 'GPHT_MAE', 'Self-supervised_PatchTST_MAE', 'FPT_MAE',
    'SimMTM_MAE', 'TimeMAE_MAE', 'Supervised_PatchTST_MAE',
    'iTransformer_MAE', 'TimesNet_MAE', 'DLinear_MAE'
]


train_datasets = [
    ('etth1_features.csv', 'performance/combined_etth1_performance.csv'),
    ('etth2_features.csv', 'performance/combined_etth2_performance.csv'),
    ('ettm1_features.csv', 'performance/combined_ettm1_performance.csv'),
    ('ettm2_features.csv', 'performance/combined_ettm2_performance.csv')
]

test_datasets = [
    #('etth1_features.csv', 'performance/combined_etth1_performance.csv'),
    #('etth2_features.csv', 'performance/combined_etth2_performance.csv'),
    #('ettm1_features.csv', 'performance/combined_ettm1_performance.csv'),
    ('ettm2_features.csv', 'performance/combined_ettm2_performance.csv')
]


results = []


as_mse_list = []
actual_mse_list = []


for model in models:
    print(f"\nTraining model: {model}")
    
    X_train_total = pd.DataFrame()
    y_train_total = pd.Series()

    for feature_file, performance_file in train_datasets:
        print(f"Processing training dataset: {feature_file}")
        
        
        features = pd.read_csv(feature_file)
        performance = pd.read_csv(performance_file)
        
        
        performance.columns = performance.columns.str.strip()
        
        
        if model not in performance.columns:
            print(f"Model {model} not found in {performance_file}. Skipping.")
            continue
        
        
        data = pd.concat([features, performance[[model]]], axis=1)
        
        X_train_total = pd.concat([X_train_total, data.drop(columns=[model])], axis=0)
        y_train_total = pd.concat([y_train_total, data[model]], axis=0)

    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
    xgb_model.fit(X_train_total, y_train_total)

    
    for feature_file, performance_file in test_datasets:
        print(f"Processing testing dataset: {feature_file}")
        
        
        features = pd.read_csv(feature_file)
        performance = pd.read_csv(performance_file)
        
        
        performance.columns = performance.columns.str.strip()
        
        
        if model not in performance.columns:
            print(f"Model {model} not found in {performance_file}. Skipping.")
            continue
        
        
        data = pd.concat([features, performance[[model]]], axis=1)

        # ------------ MSE  ------------ 
        actual_mse = performance[model].mean()  
        # ------------------------------------------
        
        X_test = data.drop(columns=[model])
        y_test = data[model]

        
        y_pred = xgb_model.predict(X_test)

        
        mse = mean_squared_error(y_test, y_pred)
        print(f"AS Model Test MSE for {feature_file} with model {model}: {y_pred.mean()}")
        print(f"Actual MSE for {model}: {actual_mse}")
        
      
        as_mse_list.append(y_pred.mean())
        actual_mse_list.append(actual_mse)


        results.append({
            'Dataset': feature_file,
            'Model': model,
            'AS Model Test MSE': y_pred.mean(),
            'Actual MSE': actual_mse 
        })


as_rank = pd.Series(as_mse_list).rank().tolist()
actual_rank = pd.Series(actual_mse_list).rank().tolist()


spearman_corr, _ = spearmanr(as_rank, actual_rank)


for i in range(len(results)):
    results[i]['AS Model Rank'] = as_rank[i]
    results[i]['Actual Model Rank'] = actual_rank[i]
    results[i]['Spearman Correlation'] = spearman_corr


sorted_indices = sorted(range(len(actual_rank)), key=lambda k: actual_rank[k])


as_rank_sorted_by_actual = [as_rank[i] for i in sorted_indices]
print(f"AS Rank sorted by Actual Rank: {as_rank_sorted_by_actual}")


result_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(result_df)


result_df.to_csv('xgboost_mse_results_with_spearman_rank.csv', index=False)
