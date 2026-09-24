# RecBole 推荐系统实验

基于 [RecBole](https://recbole.io/) 的推荐系统实验项目，当前使用 BPR 在 MovieLens-1M 和 Amazon Beauty 数据集上进行 Top-K 推荐评估。

## 项目结构

```text
.
├── config/                 # 数据集与实验配置
├── data/raw/               # 原始数据，例如 Beauty_5.json
├── dataset/                # RecBole 使用的数据文件
├── log/                    # RecBole 训练日志
├── log_tensorboard/        # TensorBoard 事件文件
├── results/                # 实验结果 CSV
├── saved/                  # 已保存的模型检查点
└── scripts/
		├── convert_amazon.py   # Amazon JSON 转 RecBole .inter
		└── run_bench.py        # 训练、评估并保存结果
```

## 环境配置

项目提供了适用于 Python 3.10、PyTorch 2.5.1 和 CUDA 12.4 的 Conda 环境：

```bash
conda env create -f environment.yml
conda activate mbrec
```

也可以在已有 Python 环境中安装依赖：

```bash
pip install -r requirements.txt
```

运行前可检查 PyTorch 是否能使用 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 数据准备

### MovieLens-1M

MovieLens-1M 的 RecBole 数据文件已放在 `dataset/ml-1m/`，包括：

- `ml-1m.inter`
- `ml-1m.item`
- `ml-1m.user`

对应配置为 `config/ml-1m.yaml`。

### Amazon Beauty

将 Amazon Beauty 的原始 review JSON 文件放在 `data/raw/Beauty_5.json`，然后运行：

```bash
python scripts/convert_amazon.py \
	--input data/raw/Beauty_5.json \
	--output dataset/amazon-beauty/amazon-beauty.inter
```

转换脚本读取以下字段，并生成 RecBole 所需的交互文件：

| 原始字段 | 输出字段 |
|:--|:--|
| `reviewerID` | `user_id:token` |
| `asin` | `item_id:token` |
| `overall` | `rating:float` |
| `unixReviewTime` | `timestamp:float` |

缺少用户或物品标识的记录会被跳过。Amazon Beauty 的配置还会过滤少于 5 次用户交互或少于 5 次物品交互的数据，详见 `config/amazon-beauty.yaml`。

## 运行实验

使用 BPR 运行 MovieLens-1M：

```bash
python scripts/run_bench.py \
	--model BPR \
	--dataset ml-1m \
	--config config/ml-1m.yaml
```

运行 Amazon Beauty：

```bash
python scripts/run_bench.py \
	--model BPR \
	--dataset amazon-beauty \
	--config config/amazon-beauty.yaml
```

脚本默认使用以下设置：

- 训练轮数：50
- 学习率：0.001
- 训练 batch size：2048
- 评估 batch size：4096
- Top-K：10
- 随机种子：2020
- 使用 RecBole 默认的数据划分与 full ranking 评估

快速检查配置和代码是否可运行：

```bash
python scripts/run_bench.py \
	--model BPR \
	--dataset ml-1m \
	--config config/smoke.yaml \
	--output results/smoke_results.csv
```

也可以通过 `--output` 指定结果文件。默认输出为 `results/first_results.csv`；如果文件已存在，新的实验结果会追加到已有 CSV 中。

## 结果

当前基线：BPR，Top-K 为 10，随机种子为 2020，使用 RecBole 默认数据划分和 full ranking 评估。完整结果保存在 `results/first_results.csv`。

| dataset       | model    |   recall@10 |   ndcg@10 |   mrr@10 |   hit@10 |   precision@10 |
|:--------------|:---------|------------:|----------:|---------:|---------:|---------------:|
| ml-1m         | BPR      |      0.1625 |    0.2556 |   0.4452 |   0.7424 |         0.1998 |
| amazon-beauty | BPR      |      0.0647 |    0.0373 |   0.0305 |   0.0711 |         0.0076 |
| ml-1m         | LightGCN |      0.1698 |    0.266  |   0.459  |   0.7505 |         0.2076 |
| amazon-beauty | LightGCN |      0.082  |    0.0466 |   0.0376 |   0.0884 |         0.0093 |

指标含义：Recall 衡量召回的相关物品比例，NDCG 和 MRR 同时考虑排序位置，Hit 表示是否至少命中一个相关物品，Precision 衡量推荐列表中的相关物品比例。

## TensorBoard

训练日志位于 `log_tensorboard/`。启动 TensorBoard：

```bash
tensorboard --logdir log_tensorboard
```

然后在浏览器打开 TensorBoard 输出的本地地址。

## 说明

- 模型检查点保存在 `saved/`。
- 普通训练日志保存在 `log/`。
- 修改数据集、训练参数或评估指标时，优先编辑对应的 YAML 配置文件。