import os
from collections import Counter
import matplotlib.pyplot as plt

# 1. 设置你的数据路径
root_dir = './datasets/datasets_grl/npy/spatiotemporal'
dirs = ['positives', 'negatives_clc', 'negatives_random']

year_counts = Counter()
positive_years = Counter()
negative_years = Counter()

print("正在扫描数据集，请稍候...")

# 2. 遍历文件夹统计年份
for d in dirs:
    path = os.path.join(root_dir, d)
    if not os.path.exists(path):
        print(f"警告：找不到文件夹 {path}")
        continue
    
    # 只计算 dynamic.npy 避免重复统计
    files = [f for f in os.listdir(path) if f.endswith('dynamic.npy')]
    
    for f in files:
        year = int(f[:4])  # 提取文件名前4个字符作为年份
        year_counts[year] += 1
        if d == 'positives':
            positive_years[year] += 1
        else:
            negative_years[year] += 1

# 3. 在终端打印文字版统计结果
print("\n=== 数据集年份分布统计 ===")
years = sorted(list(year_counts.keys()))
for year in years:
    print(f"{year}年: 总计 {year_counts[year]:>6} 个 (正样本: {positive_years[year]:>5}, 负样本: {negative_years[year]:>6})")

# 4. 绘制并保存柱状图
pos_counts = [positive_years[y] for y in years]
neg_counts = [negative_years[y] for y in years]

plt.figure(figsize=(12, 6))
# 画堆叠柱状图
plt.bar(years, neg_counts, label='Negatives (Background)', color='#1f77b4', alpha=0.8)
plt.bar(years, pos_counts, bottom=neg_counts, label='Positives (Wildfire)', color='#d62728', alpha=0.8)

plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.title('Spatiotemporal Dataset Year Distribution (FireCube)', fontsize=14)
plt.xticks(years, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存为图片
save_path = 'year_distribution.png'
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"\n✅ 柱状图已生成并保存为: {save_path}")