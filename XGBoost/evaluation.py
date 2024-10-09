import pandas as pd
from scipy.stats import spearmanr

as_mse_list =[]
actual_mse_list = []

#排名计算
as_rank = pd.Series(as_mse_list).rank().tolist()
actual_rank = pd.Series(actual_mse_list).rank().tolist()


# 计算 Spearman 相关系数
spearman_corr, _ = spearmanr(as_rank, actual_rank)

# 按照 actual_rank 排序后的索引
sorted_indices = sorted(range(len(actual_rank)), key=lambda k: actual_rank[k])

# 按照 actual_rank 排序后的 as_rank
as_rank_sorted_by_actual = [as_rank[i] for i in sorted_indices]
print(f"AS Rank sorted by Actual Rank: {as_rank_sorted_by_actual}")
