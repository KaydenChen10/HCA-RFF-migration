# HCA-RFF-migration

把 HermNet-AMC 的 **HCA（Hermitian 协方差注意力）** 模块移植到 ASKNet-RFF 的两种骨干
（WiSig 的 PatchNet、LoRa 的 ResCISNet），在两个数据集上做跨接收机射频指纹识别的对照实验。

## 最终结论（2026-10-22，拿到作者官方数据后）

HCA 的增益呈**"倒 U"规律**——只在均衡数据的中等难度区间有效：

| 数据设定 | 基线均值 | +HCA | 差 | 胜场 |
|---|---|---|---|---|
| 官方原始非均衡（完整 12 机协议） | 92.32% | 92.21% | −0.10 pp | 3/8 |
| 官方 MMSE 均衡 | 74.86% | 76.37% | **+1.51 pp** | 3/4 |
| 第三方 ZF 均衡镜像 | 69.47% | 73.90% | **+4.43 pp** | 7/8 |
| 原始·4 机小池（2 台训练） | 36.92% | 36.30% | −0.62 pp | — |

**LoRa 独立验证（N210 训练 → RTL-SDR 四天测试，CPU mini 配置）**：

| 模型 | Day1 | Day2 | Day3 | Day4 | 平均 |
|---|---|---|---|---|---|
| ResCISNet 基线 | 27.65% | 29.30% | 29.45% | 29.25% | 28.91% |
| + HCA | **33.80%** | **35.80%** | **35.10%** | **36.05%** | **35.19%（+6.28 pp）** |

机理：ASKNet 的 LSG+SKO 本就为原始数据设计（低频段失真=均匀增益+线性相位，已被其处理），
HCA 在官方原始数据上无增量；均衡数据改变了失真结构，HCA 的二阶相关注意力（z·zᴴ 对全局
相位旋转精确不变）提供互补的鲁棒性。**HCA 与 LSG/SKO 在原始数据上重叠，在均衡数据上互补。**
LoRa 的 N210→RTL-SDR 协议天然处于高失真区间，因此 HCA 增益最大（相对 +21.7%）。

其他发现：官方数据+代码可复现论文（我们 91.2%/93.4% vs 作者日志 93.35%）；
但作者发布的预训练权重损坏（自身训练域仅 23.6%），不可用于复现。

## 方法

HCA 插在 PatchNet 的 Koopman 频谱演化之后、IFFT 重建之前：

1. Koopman 输出复数频谱 (B, 127) 补零至 128，划分为 8 个子带 × 16 频点；
2. 对子带维计算 8×8 Hermitian 协方差 `z·zᴴ`（**全局相位旋转在其中精确抵消**）；
3. Newton–Schulz 迭代求矩阵平方根，取对角线 + 上三角（实虚部）经 MLP 生成子带注意力权重；
4. 加权后的频谱 reshape 回去，继续原 PatchNet 的 IFFT → patch 分类流程。

LoRa 版本使用同一思想：把 HCA 插进 ResCISNet 的 ASKModule 内部、Koopman 演化之后，按 32 个频谱子带计算 Hermitian 协方差注意力；参数量约 1126 万，CPU 可跑 mini 配置。

动机：跨接收机场景中最不稳定的是相位，最稳定的是频谱的二阶相关结构（发射机硬件指纹所在）。
HCA 让模型按"相关结构"而非"绝对相位"加权频谱。

## 复现

```bash
# 1. 数据：Hugging Face 镜像（WiSig 官网 Google Drive 常年配额超限）
#    下载 https://huggingface.co/datasets/bobo-Josef/ManySig_ZF 的 ManySig_zf.zip 并解压
python convert_manysig.py --src /path/to/ManySig_zf.pkl \
    --dst ./WiSig/dataset/ManySig/non_equalized

# 2. 基线
python WiSig/main.py --dataset_name ManySig --exp CRD --test_round 0 --seed 2023

# 3. + HCA
python WiSig/main.py --dataset_name ManySig --exp CRD --test_round 0 --seed 2023 \
    --backbone patchnet_hca --hca_bands 8
```

LoRa 复现（需要作者官方 H5 数据，放在 `LoRa/dataset/{Train,Test}/`）：

```bash
cd LoRa

# 基线
python run_CIS.py --train_rx n210_1 --train_day 1 \
  --test_rx rtl_6 --test_day 1 2 3 4 \
  --candidates 128 256 512 1024 --per_class 200 \
  --epochs 100 --patience_early_stop 15 --batch_size 128 \
  --backbone rescISnet

# + HCA
python run_CIS.py --train_rx n210_1 --train_day 1 \
  --test_rx rtl_6 --test_day 1 2 3 4 \
  --candidates 128 256 512 1024 --per_class 200 \
  --epochs 100 --patience_early_stop 15 --batch_size 128 \
  --backbone rescISnet_hca --hca_bands 32
```

WiSig 纯 CPU 单次实验约 2–4 分钟；LoRa mini 配置在 24 核 CPU 上约数小时（本仓库提供断点续训参数 `--max_seconds`）。

## 数据说明

- ASKNet 官方预处理数据在百度网盘（需登录）；WiSig 官网 Google Drive 链接测试时全部配额超限；
- WiSig 对照含官方原始/MMSE 数据与 HF 镜像 ZF 数据；基线与 HCA 使用完全相同的数据与划分；
- LoRa 使用作者官方 H5 数据，但受 CPU 限制采用 mini 配置（每类 200、候选到 1024）；因此绝对值不能与作者全量配置（每类 800、候选到 4096）相比，README 报告的是严格同配置 A/B 相对增益；
- 详细口径与全部表格见 `docs_report_zh.md`。

## 文件结构

```
WiSig/
  main.py                    # 新增 --backbone / --hca_bands / --hca_reduction
  backbones/PatchNet.py      # ASKNet 原版骨干
  backbones/HCA.py           # 从 HermNet-AMC 抽取的 Hermitian 协方差注意力
  backbones/PatchNetHCA.py   # PatchNet + 频谱子带 HCA
  utils/load_data.py         # ASKNet 原版加载器
LoRa/
  run_CIS.py                 # 新增 --backbone / --hca_bands / --max_seconds / 断点续训
  backbones/ResCISNet.py     # ASKNet 原版 LoRa 骨干
  backbones/HCA.py           # 同一个 HCA 模块
  backbones/ResCISNetHCA.py  # ResCISNet + 32 子带 HCA
  utils/load_data_h5.py      # LoRa H5 加载器
convert_manysig.py           # HF 镜像数据 -> ASKNet 目录格式
plot_lora_results.py         # LoRa 结果图脚本
hca_rff_lora_cross_receiver.png
logs/                        # LoRa 两臂完整训练日志
docs_report_zh.md            # 中文完整实验报告
```

## 致谢

- [ASKNet-RFF](https://github.com/BeechburgPieStar/ASKNet-RFF)（基线与数据管线）
- [HermNet-AMC](https://github.com/BeechburgPieStar/HermNet-AMC)（HCA 模块来源）
- [WiSig 数据集](https://cores.ee.ucla.edu/downloads/datasets/wisig/)（Hanna et al., IEEE Access 2022）
