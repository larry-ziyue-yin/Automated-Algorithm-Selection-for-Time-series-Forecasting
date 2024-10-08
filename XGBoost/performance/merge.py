import pandas as pd

# 加载每个步长的性能数据，确保文件路径正确
etth1_96 = pd.read_csv('performance/etth1_96_performance.csv')
etth1_192 = pd.read_csv('performance/etth1_192_performance.csv')
etth1_336 = pd.read_csv('performance/etth1_336_performance.csv')
etth1_720 = pd.read_csv('performance/etth1_720_performance.csv')

etth2_96 = pd.read_csv('performance/etth2_96_performance.csv')
etth2_192 = pd.read_csv('performance/etth2_192_performance.csv')
etth2_336 = pd.read_csv('performance/etth2_336_performance.csv')
etth2_720 = pd.read_csv('performance/etth2_720_performance.csv')

ettm1_96 = pd.read_csv('performance/ettm1_96_performance.csv')
ettm1_192 = pd.read_csv('performance/ettm1_192_performance.csv')
ettm1_336 = pd.read_csv('performance/ettm1_336_performance.csv')
ettm1_720 = pd.read_csv('performance/ettm1_720_performance.csv')

ettm2_96 = pd.read_csv('performance/ettm2_96_performance.csv')
ettm2_192 = pd.read_csv('performance/ettm2_192_performance.csv')
ettm2_336 = pd.read_csv('performance/ettm2_336_performance.csv')
ettm2_720 = pd.read_csv('performance/ettm2_720_performance.csv')

# 合并不同步长的 CSV 文件
combined_etth1 = pd.concat([etth1_96, etth1_192, etth1_336, etth1_720], axis=0)
combined_etth2 = pd.concat([etth2_96, etth2_192, etth2_336, etth2_720], axis=0)
combined_ettm1 = pd.concat([ettm1_96, ettm1_192, ettm1_336, ettm1_720], axis=0)
combined_ettm2 = pd.concat([ettm2_96, ettm2_192, ettm2_336, ettm2_720], axis=0)

# 保存合并后的数据到新的 CSV 文件
combined_etth1.to_csv('performance/combined_etth1_performance.csv', index=False)
combined_etth2.to_csv('performance/combined_etth2_performance.csv', index=False)
combined_ettm1.to_csv('performance/combined_ettm1_performance.csv', index=False)
combined_ettm2.to_csv('performance/combined_ettm2_performance.csv', index=False)

print("合并完成并保存到新的 CSV 文件中。")
