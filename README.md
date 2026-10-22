# HCA-RFF-migration

把 HermNet-AMC 的 **HCA（Hermitian 协方差注意力）** 模块移植到 ASKNet-RFF 的 PatchNet 骨干，
在 WiSig ManySig 数据集上做跨接收机 + 跨天（CRD）射频指纹识别的对照实验。

## 最终结论（2026-10-22，拿到作者官方数据后）

HCA 的增益呈**"倒 U"规律**——只在均衡数据的中等难度区间有效：

| 数据设定 | 基线均值 | +HCA | 差 | 胜场 |
|---|---|---|---|---|
| 官方原始非均衡（完整 12 机协议） | 92.32% | 92.21% | −0.10 pp | 3/8 |
| 官方 MMSE 均衡 | 74.86% | 76.37% | **+1.51 pp** | 3/4 |
| 第三方 ZF 均衡镜像 | 69.47% | 73.90% | **+4.43 pp** | 7/8 |
| 原始·4 机小池（2 台训练） | 36.92% | 36.30% | −0.62 pp | — |

机理：ASKNet 的 LSG+SKO 本就为原始数据设计（低频段失真=均匀增益+线性相位，已被其处理），
HCA 在官方原始数据上无增量；均衡数据改变了失真结构，HCA 的二阶相关注意力（z·zᴴ 对全局
相位旋转精确不变）提供互补的鲁棒性。**HCA 与 LSG/SKO 在原始数据上重叠，在均衡数据上互补。**

其他发现：官方数据+代码可复现论文（我们 91.2%/93.4% vs 作者日志 93.35%）；
但作者发布的预训练权重损坏（自身训练域仅 23.6%），不可用于复现。

## 方法

HCA 插在 PatchNet 的 Koopman 频谱演化之后、IFFT 重建之前：

1. Koopman 输出复数频谱 (B, 127) 补零至 128，划分为 8 个子带 × 16 频点；
2. 对子带维计算 8×8 Hermitian 协方差 `z·zᴴ`（**全局相位旋转在其中精确抵消**）；
3. Newton–Schulz 迭代求矩阵平方根，取对角线 + 上三角（实虚部）经 MLP 生成子带注意力权重；
4. 加权后的频谱 reshape 回去，继续原 PatchNet 的 IFFT → patch 分类流程。

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

纯 CPU 即可运行（torch 2.x，无需 GPU），单次实验约 2–4 分钟。

## 数据说明

- ASKNet 官方预处理数据在百度网盘（需登录）；WiSig 官网 Google Drive 链接测试时全部配额超限；
- 本实验使用 HF 镜像 ManySig_ZF 的 **ZF 均衡数据**（原始非均衡数据在镜像中只覆盖 12 个接收机中的 4 个）；
- 基线与 HCA 使用完全相同的数据与划分，对比严格公平；但绝对数值不可与 ASKNet 论文（原始非均衡数据）直接对比。

## 文件结构

```
WiSig/
  main.py                    # 新增 --backbone / --hca_bands / --hca_reduction
  backbones/PatchNet.py      # ASKNet 原版骨干
  backbones/HCA.py           # 从 HermNet-AMC 抽取的 Hermitian 协方差注意力
  backbones/PatchNetHCA.py   # PatchNet + 频谱子带 HCA
  utils/load_data.py         # ASKNet 原版加载器
convert_manysig.py           # HF 镜像数据 -> ASKNet 目录格式
```

## 致谢

- [ASKNet-RFF](https://github.com/BeechburgPieStar/ASKNet-RFF)（基线与数据管线）
- [HermNet-AMC](https://github.com/BeechburgPieStar/HermNet-AMC)（HCA 模块来源）
- [WiSig 数据集](https://cores.ee.ucla.edu/downloads/datasets/wisig/)（Hanna et al., IEEE Access 2022）
