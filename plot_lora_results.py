from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot

setup_plot()

out = Path(__file__).with_name("hca_rff_lora_cross_receiver.png")

days = ["Day 1", "Day 2", "Day 3", "Day 4", "平均"]
baseline = [27.65, 29.30, 29.45, 29.25, 28.91]
hca = [33.80, 35.80, 35.10, 36.05, 35.19]

x = np.arange(len(days))
width = 0.34
fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=180)
b1 = ax.bar(x - width / 2, baseline, width, label="ASKNet baseline", color="#8DA0CB")
b2 = ax.bar(x + width / 2, hca, width, label="ASKNet + HCA", color="#E78AC3")

for bars in (b1, b2):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.45,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

delta = hca[-1] - baseline[-1]
ax.axhline(baseline[-1], color="#8DA0CB", linestyle="--", linewidth=1, alpha=0.45)
ax.axhline(hca[-1], color="#E78AC3", linestyle="--", linewidth=1, alpha=0.45)
ax.text(
    0.02,
    0.95,
    f"HCA 平均提升 +{delta:.2f} pp（相对 +{delta / baseline[-1] * 100:.1f}%）",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=11,
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
)

ax.set_title("LoRa 跨接收机射频指纹：N210 训练 → RTL-SDR 测试")
ax.set_ylabel("Top-1 准确率（%）")
ax.set_xticks(x, days)
ax.set_ylim(0, 42)
ax.grid(axis="y", linestyle="--", alpha=0.28)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(out, bbox_inches="tight")
print(out)
