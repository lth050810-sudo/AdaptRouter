# Table 3 · API router training

AdaptRouter 的 **StartLux 0.8B 全参数 API 专家池路由器**。本目录将保留的主训练实现整理为独立 Python 包，包含训练、验证选模、固定配方 refit、断点续训及原生路由推理。

训练读取提前收集的五个专家的质量和成本反馈，不调用专家 API，也不是 PPO。路由器为完整问题选择一个冻结专家；后续专家回答生成和基准裁判不属于此包。

## 发布与复现范围

| 内容 | 本次发布 |
| --- | --- |
| StartLux API 主模型的全参数训练、损失和观察变换 | 包含 |
| 验证选模、固定配方 refit、断点续训和本地推理 | 包含 |
| 主训练历史配置、数据格式及合成示例 | 包含 |
| 原生 Instant 提示词、图像处理和双向注意力辅助代码 | 固定上游版本，随包包含 |
| 已训练全局权重 | [Hugging Face](https://huggingface.co/lyyyk/AdaptRouter-StartLux-0.8B)，不打入代码包 |
| 原始 1,328 条专家反馈及原始图像 | 未包含 |
| BBEH / IFBench / HLE 等基准专项微调代码、权重 | 清理后的归档未保留，未包含 |
| JevAny、其他基线、持续学习及开放专家池训练 | 不在本次发布范围 |

**表 3 的最终 StartLux 行含后续专项微调。该全局主训练实现及已发布全局权重，不能单独保证复现最终行的每个单元格或 71.31% / $120.46 汇总结果。** 示例数据均为人工合成，仅供检查接口，不是论文训练数据。

## 安装

需要 Python 3.11+。原生辅助代码针对 Transformers **5.8.x**；包依赖限定在这一版本范围，归档训练版本为 PyTorch 2.7.1 / Transformers 5.8.1。

从仓库根目录执行：

```bash
cd api_router
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]' -r requirements-original.txt
```

正式全参数训练需要 CUDA、BF16 支持及充足显存。FP32 参数、梯度、AdamW 状态和多模态激活均占用显存；显存份额由 `--memory-fraction` 控制，默认 0.25。该数值是当前设备总显存的比例。可按设备调整，例如 `--memory-fraction 0.9`。

可选加速内核 `python -m pip install -e '.[cuda-kernels]'` 需要与本机 PyTorch / CUDA 匹配的编译环境。没有这些内核时，原生实现使用其可用回退路径。输出磁盘建议预留至少 32 GiB，代码保留写检查点时 18 GiB 可用空间的检查；`latest.pt` 同时包含优化器状态。

下文命令均在 `api_router/` 下运行。命令行入口 `adaptrouter-api` 与 `python -m adaptrouter_api` 等价。

## 先检查格式

以下操作只检查合成数据和固定配方，不下载模型、不启动训练：

```bash
python -m adaptrouter_api validate-data --data examples/feedback.example.jsonl
python -m adaptrouter_api validate-data --data examples/route.example.jsonl --unlabeled

python -m adaptrouter_api prepare-refit \
  --data examples/feedback.example.jsonl \
  --out outputs/example-selection.json \
  --expected-datasets 2 --expected-refit-rows 4
```

合成示例不适合训练研究模型。正式训练请换成自己合法持有、按下面格式收集的专家反馈。

## 数据格式

输入可为 JSONL、JSON 行数组，或 `{"rows": [...]}`。示例见 [examples/feedback.example.jsonl](examples/feedback.example.jsonl)。

| 字段 | 说明 |
| --- | --- |
| `dataset` / `task_id` | 数据集名称及其中唯一的题目 ID；仅用于采样、日志与指标，不送入路由提示词 |
| `group_id` | 文档、图像或问题家族的分组 ID；同组禁止跨 train / validation / test |
| `split` | 明确的 `train`、`validation` 或 `test` |
| `question` | 完整原始输入，不能预先用答案或裁判反馈替换 |
| `routing_question` | 可选；长文档中原本存在的具体问题文本 |
| `document_char_count` | 可选；原始文档字符数 |
| `expert_ids` | 下列固定顺序的五个 API 专家 |
| `qualities` | 五个专家对应的质量，均为 [0, 1] 的有限数值 |
| `costs_usd` | 五个专家对应的历史单题美元成本；未知值必须为 `null` |
| `images` | 图像引用数组；每项为 `{"uri":"/path/to/image.png","mime_type":"image/png"}`，可附原始 `sha256` |

推理输入仍需 ID、分组、`split: "test"` 和完整问题，但可以省略 `expert_ids`、`qualities`、`costs_usd`。本地图像路径或 data URI 均可，图像不会通过网络自动获取。请使用唯一且稳定的 `group_id`，确保相同原始文档或图像不跨拆分。

历史实验固定专家顺序与单价如下，单位为每百万 token 美元；这些数值用于实验口径，不代表服务商当前报价。计费输出若已含推理 token，不应重复累加。

| 专家 ID | 输入 | 输出 |
| --- | ---: | ---: |
| `gpt-6.1-terra` | 1.00 | 5.00 |
| `deepseek-flash` | 0.15 | 0.60 |
| `claude-sonnet-5-5` | 2.00 | 10.00 |
| `gemini-4-argon` | 2.00 | 10.00 |
| `kimi-k3.1-pro` | 2.00 | 10.00 |

专家描述含公开固定单价。路由观察不含参考答案、质量标签、实际单题成本、未来输出长度、数据集名或题目 ID。

## 训练目标与历史配方

成本完整的题目使用 `reward_i = quality_i − λ × cost_i / cost_scale`。任意专家成本未知时，该整题使用质量奖励，不把缺失成本填零。

软目标为 `y = softmax(reward / τ)`，模型概率为 `p = softmax(logits)`：

```text
loss = −sum(y * log(p)) + 0.1 * (max(reward) − sum(p * reward))
```

| 参数 | 历史主训练值 |
| --- | --- |
| Base | `startlux-models/StartLux-Decision-v1.1-0.8B-Instant` |
| Base revision | `6496ae641cc71b149ce9dad69bd97f2a44e7bf11` |
| 上游原生辅助代码 commit | `cb1d11f1ac9e9b43b95ef3a6468fa29b607c40fd` |
| 参数更新 | 全参数；FP32 master weights，CUDA forward BF16 autocast |
| Epoch / seed / 每次更新题数 | 3 / 37 / 16 |
| AdamW 学习率 / weight decay | 1e-6 / 0.01 |
| 梯度裁剪 / regret 系数 | 1.0 / 0.1 |
| 成本系数 λ / 软目标温度 τ | 0.02 / 0.05 |
| 历史成本归一化常数 | 0.04569881355151548 |
| 训练 + 验证题数 / 数据集数 / 更新次数 | 1,328 / 10 / 249 |
| 路由文字观察预算 / 原生完整提示词上限 | 4,096 / 8,192 tokens |
| 图像观察 | 按 SHA 或 URI 去重；最多均匀选择 16 张原始图像 |

训练对每个数据集进行反频率加权，逐题确定性打乱候选顺序，并同步重排标签。成本尺度只从 train 拟合：先求各数据集已知专家成本的均值，再对数据集取平均。长文本路由观察使用固定的题目、长度、首尾片段变换；原始专家输入保留完整内容。原生完整提示词超过上限时明确报错，不隐式截断。

历史配置见 [configs/table3_startlux.json](configs/table3_startlux.json)。训练数据哈希绑定原始数据文件的精确字节，重新排版 JSON 会改变哈希。

## 下载固定基础模型

这是一个显式联网的准备命令。训练和路由本身只读取本地模型。

```bash
python -m adaptrouter_api download-base --out .artifacts/startlux-base
```

下载固定 revision 并核对文件大小、Git / LFS 摘要，生成本地 `DOWNLOAD_MANIFEST.json`。原生加载还会验证 manifest、权重分片和配置；不要手工伪造 manifest，也不要用另一个 causal checkpoint 替换 Instant 基础模型。

## 固定配方 refit

先准备绑定自己反馈文件的固定配方。默认要求 10 个数据集、1,328 条 train + validation 题目，且每个数据集均含两个拆分。

```bash
python -m adaptrouter_api prepare-refit \
  --data private_data/feedback.jsonl \
  --out outputs/fixed-selection.json \
  --lambda-cost 0.02 --epochs 3

python -m adaptrouter_api train \
  --phase refit \
  --data private_data/feedback.jsonl \
  --selection outputs/fixed-selection.json \
  --base-model-path .artifacts/startlux-base \
  --router-input-policy question_length_head_tail \
  --out outputs/table3-startlux \
  --device cuda:0 --memory-fraction 0.25
```

固定配方 helper 不声称做过验证选模。它绑定指定的 λ / epoch，且只从 train 拟合成本尺度。Refit 从原始基础模型重新初始化，使用 train + validation；test 不参与优化或选择。历史配方 [configs/table3_original_selection.json](configs/table3_original_selection.json) 只接受与原始 SHA-256 完全相符的数据。

对其他规模的数据集，在 **prepare-refit 和 train 两条命令中同时** 显式设置 `--expected-datasets N --expected-refit-rows M`。这属于自定义实验，不是原始论文训练运行。

同一命令在同一输出目录重启会恢复 `latest.pt`。恢复时核对代码、数据与配置身份，防止混用检查点。输出包括 `RUN_IDENTITY.json`、原生输入预检查、训练进度、可恢复优化器检查点、最终 `model.pt` 和 `COMPLETED.json`。已完成且身份一致时不会重复训练。请使用可信的本地检查点。

## 可选：验证选择 epoch 与 λ

如果需要在自己的 train / validation 拆分上选择配置：

```bash
python -m adaptrouter_api train \
  --phase calibrate \
  --data private_data/feedback.jsonl \
  --base-model-path .artifacts/startlux-base \
  --router-input-policy question_length_head_tail \
  --lambda-grid 0 0.0025 0.005 0.01 0.02 \
  --epochs 3 --out outputs/calibration \
  --device cuda:0 --memory-fraction 0.25
```

每个 epoch 在相同验证题目上评估候选。保留距最佳 macro quality 不超过 0.5 个百分点的配置，再选择成本完整且 macro cost 最低者；没有合格完整成本候选时仅按质量选择。使用输出 `SELECTION.json` 替换 refit 命令的 `--selection`，并使用新的 refit 输出目录。禁止用 test 结果挑选配置；最终 refit 从基础模型开始。

## 路由推理

使用自己的最终原生检查点：

```bash
python -m adaptrouter_api route \
  --checkpoint outputs/table3-startlux/model.pt \
  --base-model-path .artifacts/startlux-base \
  --data examples/route.example.jsonl \
  --out outputs/routes.json --device cuda:0
```

或到 [官方 Hugging Face 权重页面](https://huggingface.co/lyyyk/AdaptRouter-StartLux-0.8B) 自行下载完整文件；也可在自己的环境使用 Hugging Face CLI：

```bash
hf download lyyyk/AdaptRouter-StartLux-0.8B \
  --local-dir .artifacts/adaptrouter-startlux

python -m adaptrouter_api route \
  --model-dir .artifacts/adaptrouter-startlux \
  --data examples/route.example.jsonl \
  --out outputs/routes.json --device cuda:0
```

HF 模式验证发布来源和文件哈希，使用该发布全局模型的固定 λ=0.02 与原生观察策略。此模式不要求另行下载基础模型。CPU 推理可设置 `--device cpu`，速度取决于输入。

输出为每题 `expert_id`、五个候选的选择概率和路由 token 用量；不会生成专家答案，也不会消费测试质量标签计算选择结果。请将完整原始问题及图像交给被选中的专家。使用其他方式执行纯 causal LM forward 会改变 Instant 路由语义。

## 测试与来源

```bash
python -m pytest -q
```

测试覆盖损失值与梯度、真实合成参数优化、训练集成本尺度、候选顺序、缺失成本处理、拆分隔离、长文本与图像选择、验证选择、上游哈希，以及小型随机 Qwen3.5 模型的原生双向注意力、字母读出与重计算梯度一致性。小型 CPU 测试不需要下载权重或联网，不等于完整 GPU 训练或论文数值复现。

`smoke` 为显式的两步 CUDA 工程检查：需自己提供恰好一条文本和一条图像反馈，更新后丢弃参数，不产生正式模型。其 λ / 纯 CE 检查损失不作为论文训练配方。

整理时保留主训练的奖励、加权、候选排列与优化器逻辑，调整包导入和机器路径，并抽取损失函数用于测试。原始代码 SHA-256、全局检查点身份与整理说明见 [configs/source_provenance.json](configs/source_provenance.json)。

## 许可

本目录代码采用 [Apache-2.0](LICENSE)。三个固定版本的原生辅助文件来自 [StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision/tree/cb1d11f1ac9e9b43b95ef3a6468fa29b607c40fd)，其原始字节、Apache-2.0 许可和归属声明一并保留，见 [NOTICE](NOTICE) 和 [UPSTREAM.json](src/adaptrouter_api/_vendor/startlux/UPSTREAM.json)。

基础模型和已发布路由权重保留各自的 CC BY-NC 4.0 许可与上游归属。代码许可不改变模型、数据集或 API 输出的许可。代码包不含模型权重、原始专家反馈、API 密钥或内部运行环境配置。
