import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer 



models = [
    'GPHT_MSE', 'GPHT\'_MSE', 'Self-supervised_PatchTST_MSE', 'FPT_MSE', 
    'SimMTM_MSE', 'TimeMAE_MSE', 'Supervised_PatchTST_MSE', 
    'iTransformer_MSE', 'TimesNet_MSE', 'DLinear_MSE'
]
"""
models = [
    'GPHT\'_MAE', 'GPHT_MAE', 'Self-supervised_PatchTST_MAE', 'FPT_MAE', 
    'SimMTM_MAE', 'TimeMAE_MAE', 'Supervised_PatchTST_MAE', 
    'iTransformer_MAE', 'TimesNet_MAE', 'DLinear_MAE'
]
"""


train_datasets = [
    ('etth1_features.txt', 'performance/combined_etth1_performance.csv'),
    ('etth2_features.txt', 'performance/combined_etth2_performance.csv'),
    ('ettm1_features.txt', 'performance/combined_ettm1_performance.csv'),
    ('ettm2_features.txt', 'performance/combined_ettm2_performance.csv')
]

test_datasets = [
    #('etth1_features.txt', 'performance/combined_etth1_performance.csv'),
    #('etth2_features.txt', 'performance/combined_etth2_performance.csv'),
    #('ettm1_features.txt', 'performance/combined_ettm1_performance.csv'),
    ('ettm2_features.txt', 'performance/combined_ettm2_performance.csv')
]


results = []


as_mse_list = []
actual_mse_list = []


for model in models:
    print(f"\nTraining model: {model}")
    
    X_train_total = pd.DataFrame()
    y_train_total = pd.Series(dtype='float64')

    for feature_file, performance_file in train_datasets:
        print(f"Processing training dataset: {feature_file}")
        
     
        features = pd.read_csv(feature_file, sep='\t')  
        performance = pd.read_csv(performance_file)
        
        
        performance.columns = performance.columns.str.strip()
        
        
        if model not in performance.columns:
            print(f"Model {model} not found in {performance_file}. Skipping.")
            continue
        
        
        data = pd.concat([features, performance[[model]]], axis=1)
        
       
        imputer = SimpleImputer(strategy='mean')
        X_train = data.drop(columns=[model]).select_dtypes(include=['number'])
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns) 
        y_train = data[model].fillna(data[model].mean())  

        
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            print(f"Dataset {feature_file} is empty after processing. Skipping.")
            continue
        
        X_train_total = pd.concat([X_train_total, X_train], axis=0)
        y_train_total = pd.concat([y_train_total, y_train], axis=0)

    
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train_total, y_train_total)

    
    for feature_file, performance_file in test_datasets:
        print(f"Processing testing dataset: {feature_file}")
        
    
        features = pd.read_csv(feature_file, sep='\t')  
        performance = pd.read_csv(performance_file)
        
        
        performance.columns = performance.columns.str.strip()
        
        
        if model not in performance.columns:
            print(f"Model {model} not found in {performance_file}. Skipping.")
            continue
        
        
        data = pd.concat([features, performance[[model]]], axis=1)
        
        
        X_test = data.drop(columns=[model]).select_dtypes(include=['number'])
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns) 
        y_test = data[model].fillna(data[model].mean())  

        
        if X_test.shape[0] == 0 or y_test.shape[0] == 0:
            print(f"Dataset {feature_file} is empty after processing. Skipping.")
            continue

        
        y_pred = rf_model.predict(X_test)

        
        mse = mean_squared_error(y_test, y_pred)
        print(f"AS Model Test MSE for {feature_file} with model {model}: {y_pred.mean()}")
        print(f"Actual MSE for {model}: {performance[model].mean()}")
        
        
        as_mse_list.append(y_pred.mean())
        actual_mse_list.append(performance[model].mean())

        
        results.append({
            'Dataset': feature_file,
            'Model': model,
            'AS Model Test MSE': y_pred.mean(),  
            'Actual MSE': performance[model].mean()  
        })


as_rank = pd.Series(as_mse_list).rank().tolist()
actual_rank = pd.Series(actual_mse_list).rank().tolist()


spearman_corr, _ = spearmanr(as_rank, actual_rank)


for i in range(len(results)):
    results[i]['AS Model Rank'] = as_rank[i]
    results[i]['Actual Model Rank'] = actual_rank[i]
    results[i]['Spearman Correlation'] = spearman_corr


result_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(result_df)


result_df.to_csv('ensemble_regressor_mse_results_with_spearman_rank.csv', index=False)

