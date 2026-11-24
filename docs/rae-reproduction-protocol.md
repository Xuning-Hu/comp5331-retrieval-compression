# RAE 独立复现协议与证据边界

核实日期：2026-11-04。本文区分论文描述、作者发布代码的实际行为、以及本项目自己的实验选择。`scripts/reproduce_rae.py` 为独立实现，不导入、复制或执行作者工程；标准线性层和优化器来自 PyTorch。

**当前阶段索引（2026-11-04 更新）**：第 1–10 节保留先前核查和 raw-input 实验的时间顺序；其中“严格数字对齐仍未完成”等句子描述当时阶段。第 11 节补充固定 split、仅改变训练 seed 的稳健性：两轨的弱正则改善在三个 seed 中均保持。第 12 节补充固定四次训练的 unit-input 诊断：released 轨得到 **91.88% / 87.82%** 的 Euclidean/cosine NNOverlap@5，独立 FAISS 检查为 **91.87% / 87.82%**，已在所声明的重建输入条件下近似重现论文 **91.88% / 87.80%** 的发表性能。原作者向量、精确预处理和表格配置身份仍未知；这项近似性能复现不等于确认原作者使用 unit inputs，也不覆盖此前记录。两项补充均为看过先前结果后的有界诊断，没有额外 λ 搜索，未更改冻结 BEIR 主实验。

## 1. 版本和可得材料

方法描述固定为 [arXiv v1](https://arxiv.org/html/2509.25839v1)，发布代码固定到 [commit 138b00f](https://github.com/explorerZH/RAE-KDD2026/tree/138b00f19033121cec947be28a41c9a6bf0892e3)。ACM 向 Crossref 提交的[出版记录](https://api.crossref.org/works/10.1145/3770855.3817906)确认正式论文发表于 KDD 2026，DOI 为 `10.1145/3770855.3817906`。这不是作者 README 中的占位 DOI。此次访问 ACM 正式全文返回 403，因此不能声称已逐项检查正式版全文与 v1 的变化。

作者的 [README](https://github.com/explorerZH/RAE-KDD2026/blob/138b00f19033121cec947be28a41c9a6bf0892e3/README.md)明确数据不随代码分发，要求使用者另行产生预计算向量。[Figshare 归档元数据](https://api.figshare.com/v2/articles/32413305)只列出约 16 KB 的代码 ZIP；没有原始 embedding、采样 ID 或训练 checkpoint。因此重建同类数据不等于获得论文所用的相同向量。

## 2. 论文与发布代码的事实核对

v1 使用单层线性 encoder/decoder，重构项加 Frobenius 正则；训练为 Adam、batch 128、3,000 updates，余弦学习率从 `1e-3` 降至 `1e-5`。数据按 9:1 划分，以原空间和压缩空间邻居交集比例评估。原论文实验对象如下；这里的检索目标是向量邻居，没有 BEIR relevance qrels。

| v1 数据 | 表示 | 样本数 | 原始维度 |
|---|---|---:|---:|
| CelebA | ViT | 10,000 | 512 |
| IMDb | MPNet | 20,000 | 768 |
| ImageNet-Tiny | DINOv2 | 15,000 | 384 |
| Flickr30k | CLIP 图像与文字拼接 | 15,000 | 1,024 |

来源：[v1 §3.2、§4.1](https://arxiv.org/html/2509.25839v1)。这些名称没有充分指定 checkpoint、预处理和完整采样顺序。

代码核查得到下列差异，不能以 README 或函数注释代替实际执行语义：

| 项目 | 发布代码实际行为 | 本项目处理 |
|---|---|---|
| bias | 两个 `Linear` 都使用默认 bias，尽管文件注释说无 bias | `released` 保留；`formula` 无 bias |
| 正则覆盖范围 | Adam 接收全部 model 参数，含 encoder/decoder 和 bias；并非只 encoder | 两轨分别明记范围 |
| reconstruction | 全部 batch×dimension 元素的 MSE mean | `released` 保留；`formula` 用每向量 SSE 的 batch mean |
| centering | 没有减 corpus mean | 两轨均不中心化 |
| 训练归一化 | 默认使用 raw vectors，可显式启用单位化 | 默认保持；`--normalize-training` 明记变更 |
| cosine 评价 | 输入 encoder 前单位化，输出后再次单位化 | 两轨都采用此评价流程 |
| update budget | 转为完整 epoch，实际更新次数可能小于 3,000 | `released` 记录取整；`formula` 恰好指定步数 |
| scheduler | 每 epoch 更新 | `released` 保留；`formula` 每 update 更新 |

上述实现信息来自作者 [model.py](https://github.com/explorerZH/RAE-KDD2026/blob/138b00f19033121cec947be28a41c9a6bf0892e3/code/model.py)、[train.py](https://github.com/explorerZH/RAE-KDD2026/blob/138b00f19033121cec947be28a41c9a6bf0892e3/code/train.py)、[config.py](https://github.com/explorerZH/RAE-KDD2026/blob/138b00f19033121cec947be28a41c9a6bf0892e3/code/config.py)和 [data_utils.py](https://github.com/explorerZH/RAE-KDD2026/blob/138b00f19033121cec947be28a41c9a6bf0892e3/code/data_utils.py)。模型默认值的解释以 [PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)为准。

### 正则系数不能只照抄数字

以下为我们的数学核对。令输入宽度为 D、batch 大小为 B、名义 Adam decay 为 w。Coupled Adam 的 `weight_decay=w` 向梯度加入 `wθ`，对应显式惩罚 `(w/2)||θ||²`。全元素 MSE 为 `Σ||x_hat-x||²/(BD)`。把重构改为 `Σ||x_hat-x||²/B` 后，要保持正则与重构的相对比例，显式系数应成为 `D*w/2`。

所以脚本 `--lambdas` 始终表示名义 w；`formula` 实际优化：

`mean_i ||Wd We x_i - x_i||² + (D*w/2)(||We||F² + ||Wd||F²)`。

`released` 使用 element MSE 与优化器 coupled decay，包括 bias。每次输出分别保存 `nominal_adam_decay` 和 `explicit_frobenius_coefficient`。两轨不是数学等价的模型：bias、采样顺序、学习率时点、Adam epsilon 的相对尺度仍可能不同。AdamW 的 decoupled decay 也不能直接当作这里的显式 L2 梯度。依据：[Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)、[MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)、[AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)。

## 3. 可执行的独立几何验证

脚本接受一个已有的二维 float32 `.npy` 文件。它不生成新 embedding，不下载模型，也不读取 relevance 标签。初始验证使用已有 SciFact corpus 向量；这检查算法与实现，不复现论文表格。若另行编码 IMDb 20,000 条文本，必须标为“同类数据重新编码”，记录文本 ID、model revision、截断和 pooling，不称为作者相同样本。

初始预先声明的名义 decay 网格（predeclared nominal-decay grid）为：

- seed `2026`；90% vectors 拟合，10% held-out vectors 查询训练集；保存所有行号。
- 目标维度 `128`；名义 w 为 `0, 2e-5, 2e-4`；同时做 `formula`、`released`；PCA 在相同训练数据拟合。
- 3,000 requested updates，batch 128，Adam；CPU 4 threads。更高线程数必须先实际计时，不能假定越多越快。
- held-out 的全部向量均评价；另选最多 256 个 train anchors 作 in-sample 诊断。所有方法共享这些 anchors 和候选池。
- 分别报告 Euclidean 和 cosine 的 NNOverlap@1/5/10/20/50；PCA 的 fit 也按 metric 使用 raw 或单位化输入。
- 用 NumPy 精确搜索；同分按候选行号升序；in-sample 以真实行号排除自身。不会把“第一名”无条件删掉，因此重复向量仍可正确排除 self。
- 保存每次训练的两项 loss、learning rate、最终 checkpoint、encoder Frobenius norm、最大/最小奇异值、非零奇异谱 condition number 和实际更新步数。不按 held-out 结果挑 checkpoint。

训练日志的 `probe_reconstruction_per_vector` 在训练顺序的前最多 1,024 条向量上计算每向量 SSE 的均值；以下给出的重构 SSE 均指这个固定训练 probe，不是完整训练集或 held-out 重构均值。它与用于邻居评价的最多 256 个随机 train anchors 是两个不同的诊断子集。

网格来源必须区分：`w=0` 为官方 `config.py` 默认值和零正则对照；`2e-5` 来自官方 README 的 **CelebA / CLIP / cosine / 256d** 示例，并不是 IMDb 的最优配置；`2e-4` 是本项目预先声明的较强正则诊断点，没有来自原表最优值的依据。它们均不能描述为“原表指定的三个配置”。随后直接查看原文 Figure 1 发现此粗网格漏掉了弱正则区域；透明的源图驱动补充见第 10 节。

运行命令（替换输入路径和 dataset label）：

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_rae.py --vectors cache/scifact/VECTOR_FILE.npy --output runs/rae-geometry-scifact --dataset-label scifact-existing-frozen-embeddings --dimensions 128 --seeds 2026 --lambdas 0 0.00002 0.0002 --steps 3000 --threads 4
```

输出目录须为空，以免将旧实验和新实验混在一起。`protocol.json` 记录输入和脚本 SHA256、环境及解释边界；`split-*.json` 保存分割；每个模型保存参数和训练曲线；`results.json/csv` 保存完整指标。已有标准库不支持的平台会显式报错，不把失败运行填成零分。

### IMDb 同类数据重建

`scripts/prepare_imdb_reproduction.py` 另行准备可追溯输入，不训练压缩器。默认从 [Stanford IMDb](https://ai.stanford.edu/~amaas/data/sentiment/) 的 25,000 条有标注 training reviews 中按 seed 2026 无放回选择 20,000 条；情感标签不用作训练或采样依据，官方 test 和 unsup 池均排除。保留原始文本、HTML breaks、源路径 ID、文本和 archive SHA256。抽样是我们自己的选择，未声称恢复作者顺序。

由于原文没有指定 MPNet checkpoint，默认选择 [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)，运行时解析并固定完整 commit SHA。最大长度 384、原始文本直接输入、float32、eval 模式并冻结所有权重，均为本项目披露的选择。脚本检查模型模块：仅移除末尾 `Normalize`（若存在），保留原 Transformer 和 pooling，且 `encode` 不再次单位化。因此 `raw_embeddings.npy` 是原始 pooled 输出，不会因模型内部已有 Normalize 而误称 raw。输出记录原模块、实际模块、移除动作和向量范数统计。

可以先准备文本，等 GPU 空闲再用同一命令去掉 `--prepare-only` 完成编码：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_imdb_reproduction.py --output cache/imdb-rae-reencoded --sample-size 20000 --seed 2026 --device cuda --batch-size 16 --prepare-only
.\.venv\Scripts\python.exe scripts/prepare_imdb_reproduction.py --output cache/imdb-rae-reencoded --sample-size 20000 --seed 2026 --device cuda --batch-size 16
.\.venv\Scripts\python.exe scripts/reproduce_rae.py --vectors cache/imdb-rae-reencoded/raw_embeddings.npy --output runs/rae-geometry-imdb-reencoded --dataset-label imdb-reencoded-mpnet-not-original-author-vectors --dimensions 256 --seeds 2026 --lambdas 0 0.00002 0.0002 --steps 3000 --threads 4
```

`--archive` 可以复用已有官方 tar.gz，避免重复下载。数据准备阶段不导入 Torch；编码阶段复用固定 SHA 的模型下载缓存。准备完成后再次运行会校验输入文件哈希，最终 embedding cache 也校验 SHA256。每 256 条输出实际耗时和 ETA；未做实测前不承诺 GPU 编码用时。20,000×768 个 float32 向量本身约 61.4 MB（58.6 MiB），此外需要 archive、选中文本和模型缓存空间。

网络不稳定时，`--cache-model-only` 可以先缓存模型并保存固定 SHA，不加载模型或占用 GPU；`--offline` 在已有数据和模型缓存上运行，不重新请求模型 API。下载采用系统可信证书，未关闭 TLS 验证。缓存或明确提供的完整 40 位 SHA 会直接复用。

Stanford tar 站点的 TLS 访问失败时，可以显式选择[Stanford NLP 官方 Hugging Face 数据集](https://huggingface.co/datasets/stanfordnlp/imdb)。已核实 train parquet 的固定数据 revision 为 `e6281661ce1c48d982bc483cf8a173c1bbeb5d31`。该路线在独立目录保存 parquet、真实来源 URL、文件 SHA256 和稳定的 HF 行号 ID；**不制造原 tar 的哈希或文件名**。它需要项目环境中的 `pyarrow`，本次安装记录为 26.0.1。

```powershell
.\.venv\Scripts\python.exe scripts/prepare_imdb_reproduction.py --output cache/imdb-rae-reencoded-hf --dataset-source hf --dataset-revision e6281661ce1c48d982bc483cf8a173c1bbeb5d31 --sample-size 20000 --seed 2026 --device cuda --batch-size 16 --prepare-only
```

准备完成后，去掉 `--prepare-only`，可加 `--offline` 使用已经下载并固定的 MPNet 模型。此时 `source_ids.json` 对应固定 parquet 的零起始行号，不能把这些 ID 当成原始 Stanford 文件名或论文作者的样本 ID。

这不是对作者脚本的 bitwise replay：本实现使用独立 NumPy RNG 和自己的 batching/search，且与作者的 FAISS ties 处理不同。获得作者向量也不自动意味着数值完全一致。三种结论必须分开：

| 结论级别 | 必要证据 | 当前可做的工作 |
|---|---|---|
| 实现检查 | 手算 kNN、duplicate/self、loss scaling、低秩训练、保存加载测试 | 本脚本配套单元测试 |
| 算法复现 | 固定数据/训练/搜索；RAE、零正则 AE、PCA；几何和谱随正则变化 | SciFact 或重新编码 IMDb 的独立实验 |
| 论文数字复现 | 相同向量、IDs、完整方法/超参数、原表结果及差值 | 公开材料不足，尚不能宣称完成 |

## 4. 如何与 BEIR / QRAE 主实验衔接

主实验中的 centered AE、controlled RAE 和 QRAE 应共享 preprocessing、reconstruction pool、训练预算和搜索协议，才能解释 λ、ranking loss 和负例变化的影响。这里的 raw geometry 两轨单独报告；不把它们悄悄替换成原来 centered AE 的结果。训练向量单位化且存为缓存时，raw 在本脚本仅指“不再中心化”，不表示能恢复模型归一化前的向量。

BEIR 的 nDCG/Recall/MRR 是本项目增加的 task evaluation，不是从 RAE v1 已发表 relevance 结果中复现。向老师报告时，应逐项展示“原论文算法与几何验证”和“本项目检索扩展”的证据。若暂无原始向量，不可把原论文分数对齐这一要求标为完全达成；可把向作者索取 embedding、sample IDs 和确切超参数作为后续事项，不自动发送任何消息。

## 5. 理论和结果表述边界

我们的线性代数检查：m<D 的 encoder 必有非平凡零空间，故正的最小非零奇异值并不为整个原空间提供正的范数下界。Frobenius 正则约束上界，也不能独自保证最小奇异值、condition number 或 kNN 顺序单调改善。condition number 在这里是经验诊断，不应称为 QRAE 的全空间检索保证。

所有已声明的零正则与非零正则结果都保留，不只挑最好曲线；是否属于适中强度需由原始尺度和实测解释，不能先给粗网格中的非零点贴上“适中”标签。用独立 seeds 扩展稳定性检查时报告全部种子；在观察几何 held-out sweep 后，不应继续把它称作未接触过的 final test。lambda 趋势验证的数据与 BEIR 最终 test queries/qrels 分开。

当前文档不写未经运行的 accuracy、耗时或“复现成功”结论。结果必须来自保存的运行记录，并据真实证据说明是否与论文趋势一致。

## 6. SciFact 已完成的独立几何验证

真实输出位于 `runs/rae-geometry-scifact-resumed/`，运行日志为 `runs/environment/rae-geometry-scifact-resumed.log`。此前中断的 `runs/rae-geometry-scifact/` 保留为 partial，不与这次完成结果合并。

输入为现有 E5-base-v2 SciFact 文档缓存：5,183×768，float32 向量范数在约 0.99999988 至 1.00000012；seed 2026，4,664 条训练、519 条 held-out、256 个 train probes，目标维度 128。formula 完成 3,000 updates，released 按完整 epoch 完成 2,997 updates。这里只评价文档几何，不使用 relevance qrels；只有一个 seed，不推断统计显著性。

| 方法/轨道 | 名义 w | Held-out cosine NNOverlap@5 | Held-out Euclidean NNOverlap@5 |
|---|---:|---:|---:|
| PCA | 不适用 | 0.694798 | 0.721387 |
| formula | 0 | 0.700578 | 0.683237 |
| formula | 0.00002 | 0.006551 | 0.006551 |
| formula | 0.0002 | 0.001541 | 0.001541 |
| released | 0 | 0.659345 | 0.639306 |
| released | 0.00002 | 0.010019 | 0.010405 |
| released | 0.0002 | 0.000771 | 0.002697 |

本次固定网格没有出现“正则改善邻居保持”的趋势。两个正则强度都使最小奇异值缩至约 `1e-7` 或 `1e-8` 的数量级，非零谱 condition number 增至约 `9e5` 至 `1e7`。formula 的重构误差从零正则时 0.0743 增至约 0.2258/0.2424；released 从 0.0825 增至约 0.2258/0.2383，均按每向量 SSE 计算。这些结果与几何指标退化一致。

解释边界：这是不同于原文的 E5 单位化文本向量。原文向量和具体模型处理未公开，不能假定相同数值的 decay 在不同输入尺度与分布下具有同样作用。本次结果支持“公开 README 的其他数据示例强度不能不加检查地迁移到这份缓存”的判断，不证明 RAE 在其原实验中无效，也不是已完成原表数字复现。该 SciFact 几何网格没有为了提高分数追加调参；IMDb 后续源图驱动补充另列于第 10 节。运行与主项目任务并行，因此这里不比较训练时延。

## 7. 原始文献与实现出处核查

| 项目 | 核实到的一手来源 | 本项目如何使用、哪些内容没有核实 |
|---|---|---|
| RAE 正式出版 | Han Zhang、Dongfang Zhao；KDD 2026 V.2；[DOI 10.1145/3770855.3817906](https://doi.org/10.1145/3770855.3817906)。[ACM 提交的 Crossref 记录](https://api.crossref.org/works/10.1145/3770855.3817906)列出版日期为 2026-08-08 | 核实书目身份。ACM 正式全文访问返回 403，不能据此声称已检查正式版每项公式或表格。 |
| RAE 方法版本 | [arXiv:2509.25839v1](https://arxiv.org/html/2509.25839v1)，2025-09-30；方法 §3、实验 §4 | 固定本次算法解释所依据的文本版本，不将 v1 自动等同于 KDD 最终正文。 |
| 作者发布实现 | [explorerZH/RAE-KDD2026 固定 commit](https://github.com/explorerZH/RAE-KDD2026/tree/138b00f19033121cec947be28a41c9a6bf0892e3)；[Figshare 作者归档](https://doi.org/10.6084/m9.figshare.32413305.v1) | 仅核对 bias、loss reduction、正则作用范围和训练协议；本项目没有导入或复制该工程。README 的 DOI 占位符不能作正式引文。 |
| IMDb 原数据出处 | [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) | 原 tar 站点在本次任务中的 TLS 请求失败。未下载成功的 tar 不生成伪造哈希或来源记录。 |
| 实际使用的 IMDb 分发 | [Stanford NLP 官方 HF 数据集固定 revision](https://huggingface.co/datasets/stanfordnlp/imdb/tree/e6281661ce1c48d982bc483cf8a173c1bbeb5d31) | 使用其 train parquet，保存实际文件 hash 和源行号；原 tar 文件名不可恢复。来自同一数据集不意味着相同采样或相同原论文向量。 |
| 实际使用的 MPNet 模型 | [sentence-transformers/all-mpnet-base-v2 固定 revision](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/e8c3b32edf5434bc2275fc9bab85f82640a19130) | 本地核对该 snapshot 的 `modules.json`、`sentence_bert_config.json` 和 `1_Pooling/config.json`：Transformer、mean pooling、末尾 Normalize，768 维、最大长度 384。本项目显式去掉末尾 Normalize 以保留 raw pooled 输出；原论文是否使用这一精确 checkpoint 未知。 |
| Product Quantization | H. Jégou、M. Douze、C. Schmid，*Product Quantization for Nearest Neighbor Search*；TPAMI 33(1):117-128，2011-01；[DOI 10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57) | 通过 [IEEE 提交的 Crossref 记录](https://api.crossref.org/works/10.1109/TPAMI.2010.57)核实作者、标题、卷期页码及 DOI。仅作相关工作，没有声称完成 PQ 实验，也未因书目验证而声称已读其全部正文。 |
| HNSW 正式期刊版 | Yu A. Malkov、D. A. Yashunin，*Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*；TPAMI 42(4):824-836；[DOI 10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473) | [IEEE 提交的 Crossref 记录](https://api.crossref.org/works/10.1109/TPAMI.2018.2889473)确认正式卷期出版日期为 2020-04-01；DOI 字符串中的 2018 不应替代此期刊卷期的 2020 年份。 |

课程报告应分别引用正式出版物、实际读取的方法版本和实际执行的数据/模型版本。实验缓存中的 manifest 是运行来源证据；作者论文或官方库文档不会替代本项目自己的结果文件。

## 8. IMDb 同类数据重建的完成结果与原表参考

本次完整缓存位于 `cache/imdb-rae-reencoded-hf/`。固定数据 revision 为 `e6281661ce1c48d982bc483cf8a173c1bbeb5d31`，parquet SHA256 为 `db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c`；模型 SHA 为 `e8c3b32edf5434bc2275fc9bab85f82640a19130`。从官方 HF train 的 25,000 条抽样 20,000 条，保存 20,000 个唯一源行号 ID，18,000/2,000 几何训练/held-out 划分。

GPU 编码正常结束，输出 20,000×768 个 raw float32 pooled vectors。实际模块为 Transformer + mean pooling，移除了原模型末尾 Normalize；向量范数 min/mean/max 为 1.942159/2.595955/3.294391。`embedding_manifest.json` 保存哈希、规范化处理、软硬件环境和实际执行的 `encoding_script.py` 快照。raw embedding 文件 SHA256 为 `48edd54702b747d879788c465c7fa1ce39b1c028bfeab0a782ea14f35add60f1`。

初始完整几何输出位于 `runs/rae-geometry-imdb-reencoded-hf/`；维度 256、seed 2026、CPU 2 threads、上述粗 w 网格。formula 使用 3,000 updates，released 使用 2,961 updates（18,000 条、batch 128、完整 epoch 取整）。下面列出初始全部配置的 held-out NNOverlap@5，而非挑选表现最好的模型。后续源图驱动的单点诊断单独保存和报告于第 10 节。

| 本项目方法/轨道 | 名义 w | Euclidean NNOverlap@5 | Cosine NNOverlap@5 |
|---|---:|---:|---:|
| PCA | 不适用 | 93.55% | 71.17% |
| formula | 0 | 77.05% | 74.66% |
| formula | 0.00002 | 66.30% | 69.36% |
| formula | 0.0002 | 8.12% | 8.16% |
| released | 0 | 78.28% | 74.17% |
| released | 0.00002 | 66.43% | 69.34% |
| released | 0.0002 | 8.13% | 7.90% |

以下仅为 **published reference, different vectors/protocol; not a matched reproduction**。数值来自 [v1 Table 1 的 IMDb 768→256 列](https://arxiv.org/html/2509.25839v1#S4.T1)，不得混入本项目观测值或当作匹配条件下的检验：

| 已发表参考 | Euclidean top-5 preservation | Cosine top-5 preservation |
|---|---:|---:|
| PCA | 93.38% | 70.86% |
| RAE | 91.88% | 87.80% |

该表行没有公布足够信息把其确切 w、embedding checkpoint、样本 ID 和完整协议与本项目一一对应。本次 PCA 数值接近原表；固定网格中的 RAE 两轨没有匹配原表 RAE，且正则均未改善该网格下的结果。零正则两轨的重构 SSE 约 0.253，w=2e-5 时约 1.301，w=2e-4 时约 3.695；非零奇异谱 condition number 从约 3.2 增至约 1.2e4，再至约 2.4e5/4.3e5。这里提供的是可审计的未匹配结果，不能据此断言论文结果错误。

老师关于“干净复现并核验已发表结果”的要求，因此有了实现、独立测试、明确设置、同类数据和原表参照等证据；**严格数字对齐仍未完成**。本项目 BEIR/QRAE 实验与这条复现证据分别报告，不能用其下游分数填补这项缺口。

## 9. 作者 PCA 的 cosine 预处理审计

再次核对固定 commit 的 [baselines.py](https://github.com/explorerZH/RAE-KDD2026/blob/138b00f19033121cec947be28a41c9a6bf0892e3/code/baselines.py#L213)：cosine 分支先将 train 和 validation 输入都单位化，再对 train fit PCA，用同一种处理后的 train/validation 做 transform，最后单位化降维输出。Euclidean 分支则在 raw 输入上 fit 和 transform。作者调用的是标准 sklearn PCA，内部中心化；没有发现“raw fit 后才把 query 单位化”造成均值不匹配的这项错误。

本脚本对 PCA 的这套按 metric 预处理顺序与作者发布代码一致。因此不能把原表 cosine/Euclidean 的差距归咎于上述未发生的处理错配，也不能将其作为本次无法数字对齐的解释。可确认的实现差异是：作者使用默认 `svd_solver=auto` 和未严格锁定的 [scikit-learn>=1.2 依赖](https://github.com/explorerZH/RAE-KDD2026/blob/138b00f19033121cec947be28a41c9a6bf0892e3/requirements.txt)，本脚本使用 float64 full SVD 拟合；精确搜索分别为作者 FAISS 和本脚本 NumPy，同分约定也不同。未做归因控制实验，故不宣称其中某项造成了观测差值。

RAE 发布训练脚本是另一个事实：默认 raw training，而 cosine evaluation 前会单位化输入；本脚本的 released 轨保留了此行为。它与 PCA 的 fit/transform 顺序不能混为一谈。

## 10. Figure 1 弱正则区域核查与透明补充

直接查看 [v1 Figure 1(a) 原图](https://arxiv.org/html/2509.25839v1/picture/analysis_imdb_mpnet_20000_euclidean_val_256d_Adam.png)，IMDb Euclidean、256d 的邻居保持峰值位于 `1e-7` 和 `1e-6` 刻度之间，图读约为 `3e-7`；到约 `1e-6` 后精度和最小奇异值明显下降。[Figure 1(b) Flickr30k cosine 原图](https://arxiv.org/html/2509.25839v1/picture/analysis_flickr30k_CLIP_VIT__15000_cosine_val_768d_Adam.png)的峰值也在 `1e-7` 附近的弱正则区域。以上均为图形读数，**不是从数值文件核实的精确超参数**；固定 commit 的公开仓库只含通用训练/基线脚本，没有这两张图的数值 sweep 配置或绘图源数据。

因此初始 `{0, 2e-5, 2e-4}` 网格没有覆盖原图的收益区域，不能用它检验“原论文最佳强度也没有收益”。这是初始实验设计的覆盖不足，不能归咎于论文。本项目据外部源图增加一个固定名义 decay `w=3e-7`，不围绕本项目 held-out 分数继续搜索。输入向量、256d、seed 2026、18k/2k 划分、batch 128、3,000 requested updates、学习率、两条实现轨及 CPU 2 threads 全部保持原样；不改 BEIR 主实验任何参数或选择。

此补充是在已经看到粗网格结果后作出的 **source-driven follow-up diagnostic**，不是原始预注册，也不使已看过的几何 held-out 重新变为未接触的 final test。运行前已写入 `runs/rae-geometry-imdb-figure1-check-source-driven-amendment.json`，随后原样复制到新运行目录的 `source-driven-amendment.json`；它记载上述事实、依据、不确定性和原图/README/config 的 SHA256。原图与源码依据保存在 `runs/rae-geometry-imdb-figure1-source/`，原始实验不覆盖。

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_rae.py --vectors cache/imdb-rae-reencoded-hf/raw_embeddings.npy --output runs/rae-geometry-imdb-figure1-check --dataset-label imdb-reencoded-mpnet-source-figure1-followup-not-original-author-vectors --dimensions 256 --seeds 2026 --lambdas 0.0000003 --steps 3000 --threads 2
```

该值并非已经核实的 Table 1 IMDb 精确 λ；原文 Figure 1(a) 是 Euclidean 诊断，也不能据此声称其峰值就是 Table 1 cosine 的选择。最终报告应同时保留粗网格、该固定补充点以及单独标注的发表参考，不进行 best-point 选择。

补充已正常完成，保存全部 6 行指标（2 行 PCA、两轨各 2 个 metric），没有追加搜索。新旧 `split-2026.json` 逐字节一致，PCA 的全部指标行与初始运行一致；`experiment_script.py` 快照的 SHA256 与运行开始时 `protocol.json` 一致。下表差值均为 **百分点**；最后一列只作不同向量/协议下的参考差，不能当作匹配复现误差。

| 固定补充点 w=3e-7 | NNOverlap@5 | 相同轨道 w=0 的差 | 相同输入 PCA 的差 | 与发表 RAE 参考的差（非匹配条件） |
|---|---:|---:|---:|---:|
| formula / Euclidean | 83.72% | +6.67 | −9.83 | −8.16 |
| formula / cosine | 79.34% | +4.68 | +8.17 | −8.46 |
| released / Euclidean | 84.17% | +5.89 | −9.38 | −7.71 |
| released / cosine | 78.07% | +3.90 | +6.90 | −9.73 |

formula 的 encoder 最小/最大奇异值为 0.430722/1.017269、非零谱 condition number 为 2.361775；released 分别为 0.424191/1.031150、2.430863。相较各自零正则的 condition number 约 3.17/3.28，这个固定弱正则点同时降低了 condition number、提高了邻居保持，方向与原文 Figure 1 的经验趋势一致；粗网格的更强正则仍保持明显退化。仅凭这些点不能证明最佳强度、全空间理论保证或统计显著性。

补充改进了算法层面的趋势验证，但并未达到原表数值：发表的 IMDb RAE Euclidean/cosine 参考为 91.88%/87.80%，本项目同类数据两轨仍有上述差距。结论应为“一个外部源图指示的弱正则点复现了相对零正则的改善趋势；严格发表数字复现未完成”，而不是“复现失败所以 RAE 无效”，也不是“已完整复现论文”。

## 11. 固定数据划分、仅改变训练 seed 的 raw-input 稳健性

本节检验第 10 节的改善是否只来自某个训练随机种子。已有 `reproduce_rae.py --seeds` 会同时改变数据划分、训练随机性和 train probes，因此本节另用 [check_rae_seed_robustness.py](../scripts/check_rae_seed_robustness.py)，直接读取并校验原 `split-2026.json` 的全部有序 `train_rows`、`heldout_rows` 和 `probe_positions`。**split seed 始终为 2026，training seeds 为 17、42、2026**；18,000/2,000 划分、256 个 train probes、候选行顺序和同分规则完全固定。沿用 raw float32 输入、未中心化、256d、CPU 2 threads、batch 128、学习率 0.001→0.00001、3,000 requested updates。formula 实际 3,000 updates，released 为 21 个完整 epoch、2,961 updates；cosine 仍按原 helper 在编码前后单位化。

只比较名义 `w=0` 与第 10 节已确定的 `w=3e-7`，没有新 λ 网格。seed 17/42 × 两个 w × 两轨，共 **8 次新 fit**；seed 2026 的四个既有模型及 training 记录逐字节复用，明确标记 `reused`，重新计算的指标与原记录差值为零。运行前保存的 [protocol.json](results/course/rae-fixed-split-seeds/protocol.json) 和源快照记录上述计划。该计划在看到原始结果后制定，属于 post-hoc robustness，不是原始预注册，也没有新的未接触 test。

下表为三个训练 seed 的 held-out NNOverlap@5，百分数以均值 ± **sample SD** 报告；最后一列为同 seed 配对差值的均值 ± sample SD，单位为百分点。它们不是置信区间。

| 轨道 / metric | w=0（%） | w=3e-7（%） | 配对改善（百分点） |
|---|---:|---:|---:|
| formula / Euclidean | 77.407 ± 0.316 | 83.710 ± 0.075 | +6.303 ± 0.318 |
| formula / cosine | 74.510 ± 0.145 | 79.073 ± 0.260 | +4.563 ± 0.115 |
| released / Euclidean | 78.003 ± 0.297 | 84.183 ± 0.051 | +6.180 ± 0.337 |
| released / cosine | 74.100 ± 0.062 | 77.810 ± 0.301 | +3.710 ± 0.248 |

每个轨道、seed、metric 的差均为正：formula Euclidean +6.11 至 +6.67、cosine +4.45 至 +4.68；released Euclidean +5.89 至 +6.55、cosine +3.43 至 +3.90 个百分点。十二个正差共享数据、轨道或 seed，不能称为十二次独立复现。该证据支持“此固定 raw-input split 上弱正则改善不依赖这三个训练 seed 中的某一个”，不检验跨 split、跨数据集或作者原始输入的稳定性。

[轻量证据与重放说明](results/course/rae-fixed-split-seeds/README.md)保留全部 24 行 metric、每 query overlap@5、每 seed 配对差、training history、完整 encoder 奇异谱、模型与原始邻居文件哈希以及源快照。[独立核验记录](verification/rae-fixed-split-seeds.json)从已保存邻居集合重算 overlap，并核对每 query 聚合、sample SD、四个复用锚点和注册文件哈希；未把这一检查冒称新的搜索后端复现。

该审计脚本有意绑定本地已有缓存：raw vectors SHA256 为 `48edd54702b747d879788c465c7fa1ce39b1c028bfeab0a782ea14f35add60f1`，原始 split SHA256 为 `7731cbba1456c18feda61d7515e3e0731efc57e59ec70ffe5878e3540d98bc88`，共用独立 helper SHA256 为 `327838dd1bf77568c3bb320c32a7cc3478d73cdf29bf106dbf00a1fe4363df41`。轻量包不含 vectors/weights：精确重放需要保留的本机缓存，或完整重新编码后验证字节一致。若新编码字节不同，应建立另一个明确标识的 study；不能修改绑定哈希后声称重放了同一结果。

## 12. Unit-input 单因素诊断与近似发表性能复现

实际选用的固定 MPNet checkpoint 包含末尾 Normalize，而第 8 节的 raw cache 有意移除了它；原作者未公开的向量文件是否已经单位化尚未知。本节固定这一未决输入约定作单因素诊断：使用原独立 helper 将既有 20,000 行 raw 向量逐行 L2 单位化，不重新调用模型，不更换样本或次序。整个单位化矩阵作为实验输入，所以 Euclidean 原空间参考邻居也使用 unit inputs；不是只打开训练阶段的 `normalize_training` 开关。

运行前记录在 [preregistered-diagnostic.json](results/course/rae-unit-input-diagnostic/preregistered-diagnostic.json)。这个历史文件名仅表示在本轮转换和 fit 前固定计划，**不表示原始研究预注册**；计划是在看过先前 raw-input 结果后制定的。固定 seed 2026、同一个 split 和 probes、256d、上述 batch/lr/预算和 CPU 2 threads，只运行两轨 × `{0, 3e-7}` 的 **四次 fit**，没有额外 λ、seed、epoch 或预处理搜索。单位化数组 SHA256 为 `9808dae9c376c3fa08d27d0b5ab106e2cbc7f6acb557197b5c11201ea676b590`；原 raw 数组、split 和 helper 与第 11 节相同。转换规则、源缓存 SHA、代码快照与行 ID 来自 [input-normalization.json](results/course/rae-unit-input-diagnostic/input-normalization.json)，没有覆盖原缓存。

下面保留所有配置；数值为 NumPy 精确搜索的 held-out NNOverlap@5。raw 对照来自第 8/10 节，unit 结果来自本轮 [comparison.json](results/course/rae-unit-input-diagnostic/comparison.json)。

| 轨道 / 名义 w | raw Euclidean（%） | unit Euclidean（%） | raw cosine（%） | unit cosine（%） |
|---|---:|---:|---:|---:|
| PCA / 不适用 | 93.55 | 93.16 | 71.17 | 71.17 |
| formula / 0 | 77.05 | 78.23 | 74.66 | 74.77 |
| formula / 3e-7 | 83.72 | 91.36 | 79.34 | 89.53 |
| released / 0 | 78.28 | 81.12 | 74.17 | 77.60 |
| released / 3e-7 | 84.17 | **91.88** | 78.07 | **87.82** |
| 论文 RAE 参考（原输入与完整配置未知） | — | **91.88** | — | **87.80** |

raw/unit Euclidean 的原空间参考几何不同，两个百分数之差应解释为输入约定的敏感性，不能称为相同检索任务上的改善。cosine 参考在精确算术中具有尺度不变性，但 float32 可产生极近边界的次序变化；独立 FAISS 检查得到 raw/unit 参考 Top-5 overlap 为 99.99%，1,997/2,000 个 query 的有序邻居完全一致。单个 seed 的 unit-input 结果也不能借用第 11 节 raw-input 的三 seed SD。

[独立后端核验](verification/rae-unit-input-independent.json)对全部四个已保存 neural checkpoint，用 FAISS 在完整 2,000 held-out queries × 18,000 training candidates 上重算八项 Top-5 指标，没有重新 fit。五项与 NumPy 完全一致，三项相差 10,000 个邻居槽位中的一个共同邻居，即 0.01 个百分点。released 弱正则点的 NumPy 交集计数为 Euclidean **9,188/10,000**、cosine **8,782/10,000**；FAISS 为 **9,187/10,000**、**8,782/10,000**。另一次独立 NumPy 重算仍为 9,188；差异定位到一个 query 的压缩空间第五名边界，其两个候选的 float64 平方距离相差约 `3.96e-9`。原 NumPy 结果和真实后端差异均保留，没有用另一种计分规则覆盖原结果。

据此，当前可以表述为：**本项目在明确声明的重建 IMDb/MPNet 单位化输入条件下，近似重现了发表的 91.9% / 87.8% 性能，并复现了固定弱正则点的收益。** NumPy 的 Euclidean 值在两位小数下与论文一致，cosine 高 0.02 个百分点；FAISS 分别低 0.01、高 0.02 个百分点。原作者的向量、精确 checkpoint/采样/表格超参数身份仍不可恢复，因此不能将接近的分数解释为原始配置逐项完全复现，也不能据此断言作者实际采用单位化输入。第 8–10 节“尚未达到原表”的历史结论由本节新增证据更新；保留先前 raw 条件有助于说明结果对输入尺度敏感，而非只展示接近论文的条件。

[本轮轻量导出](results/course/rae-unit-input-diagnostic/README.md)保留全部 metric、训练记录、转换/实验/核验脚本和 SHA。其 `protocol.json` 的保守预先 claim scope 仍保留，表达的是不主张作者原始配置身份匹配，并非否认运行后已观察到的近似性能一致。精确重放同样需要已绑定的本机 raw/derived 缓存，或重新编码并核对相同 SHA；新字节输入应作为新 study。两项补充均独立于六个冻结 BEIR study，没有使用这些结果重新选 BEIR 参数，也没有覆盖其 180 条正式 test 结果。
