import matplotlib.pyplot as plt

# 1. 准备示例数据（请自行根据实际情况调整数值）
W = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0]

helpful  = [0.80, 0.82, 0.83, 0.86, 0.90, 1.00, 1.05, 1.15, 1.20, 1.25, 1.30, 1.28, 1.32, 1.25, 0.90]
harmless = [0.85, 0.80, 0.82, 0.85, 0.88, 0.90, 0.93, 0.95, 0.98, 1.00, 1.02, 1.05, 1.04, 1.00, 0.85]
diversity= [0.78, 0.79, 0.78, 0.77, 0.76, 0.75, 0.78, 0.80, 0.80, 0.82, 0.85, 0.86, 0.88, 0.90, 0.92]

# Humor 走另外一条 y 轴，数值范围与前面不同
humor    = [0.00, 0.10, 0.20, 0.50, 0.70, 0.90, 1.00, 0.95, 0.90, 0.88, 0.90, 0.90, 0.88, 0.85, 0.80]

# 2. 创建图和轴
fig, ax1 = plt.subplots(figsize=(6,4))
ax2 = ax1.twinx()  # 与 ax1 共享 x 轴的第二个 y 轴

# 3. 在左轴上画 Helpful / Harmless / Diversity
p1 = ax1.plot(W, helpful,   '-o', color='red',    label='Helpful')
p2 = ax1.plot(W, harmless,  '-o', color='blue',   label='Harmless')
p3 = ax1.plot(W, diversity, '-o', color='saddlebrown', label='Diversity')

# 4. 在右轴上画 Humor
p4 = ax2.plot(W, humor, '-s', color='green', label='Humor')

# 5. 设置坐标轴标签和范围（根据需要可自行调整）
ax1.set_xlabel('W')
ax1.set_ylabel('Helpful, Harmless, Diversity Score')
ax2.set_ylabel('Humor Score')

# 6. 联合图例（将两条 y 轴上的图例合并）
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# 7. 网格、标题等细节（可根据需要增改）
ax1.grid(True, linestyle='--', alpha=0.5)
plt.title('Example of Dual-Axis Plot')

plt.tight_layout()
plt.show()
