# Qwen2.5-VL 草稿模型接受率问题：诊断与实验复盘

实验周期：2026-08-03 至 2026-08-27。本文回答两个问题：为什么当前 VLM speculative decoding 的接受长度明显低于预期，以及围绕这个问题已经做过哪些尝试、得到什么结论。

## 技术结论：瓶颈来自三部分，而不是单一的视觉缺失

当前证据否定了“DFlash 完全没有使用图片信息”这一解释。将进入 draft 的 image-token hidden states 清零后，五项平均接受长度下降 15.9%，说明视觉内容确实参与预测；但打乱这些视觉 token 的顺序只使均值下降 0.23%，说明 draft 主要使用聚合内容，几乎没有利用空间顺序。

同时还确认了两个独立瓶颈：原始 68k 训练数据过度集中于长描述，与短 VQA、OCR、caption 和 reasoning benchmark 失配；DFlash 在八 token 并行块内缺少完整因果依赖，Domino 的 causal head 能显著改善 proposal offset 2 以后的位置。

因此最可信的总体解释是：**数据分布失配是次要贡献，块内依赖缺失和视觉几何信息利用不足是主要的框架级贡献。** 全局 Q-Former、masked pooling + MLP 和标量 gate 均未解决问题；Domino 与 direct MRoPE 分别解决了部分依赖和空间位置问题，但收益具有明显任务差异。

## 已经确认的主要发现

| 发现 | 关键证据 | 当前判断 |
|---|---|---|
| 视觉信息并未缺失 | image rows 清零使平均接受长度从 1.8833 降至 1.5835 | 已确认使用视觉内容 |
| 空间顺序利用不足 | shuffle 后均值仅从 1.8833 变为 1.8789 | 高可信的视觉结构瓶颈 |
| 数据分布确有失配 | 68k 数据全部为单图长描述；matched continuation 只提升 +0.0231 | 真实但不是主要解释 |
| 后半段生成更难 | 33–64 token 为 2.0750，129–256 token 降至 1.7807 | 晚期熵和依赖问题明显 |
| 块内依赖是瓶颈 | Domino 改善 offsets 2–7；warm-start Stage 1 均值到 1.9684 | 已确认 |
| 全局视觉压缩没有帮助 | Q-Former 与 pooled MLP 均未提高平均接受长度 | 当前设计应停止 |
| MRoPE 有任务特定价值 | Hard MRoPE 主要改善 GQA 和 MMMU-Pro | 有信号，但不是通用提升 |

## 比较范围与指标口径

主要对照使用同一个 Qwen2.5-VL-7B-Instruct target、Transformers reference evaluator、BF16、SDPA、batch size 1 和 greedy decoding。legacy5 包含 GQA、TextVQA、COCO Caption、CharXiv Reasoning 与 MMMU-Pro。后期正式小规模实验通常使用每项 10 个样本、每样本固定生成 256 tokens。

本文优先使用 `emitted tokens per verification step`，即接受的 draft tokens 加上每轮 target 保底生成的 1 token。部分较早表格记录的是不含保底 token 的 raw `accept_length`，相关段落会单独说明。ViSpec tree 使用不同的候选树和分母，其接受长度只能作为系统级参考，不能与线性 DFlash/Domino 直接解释为同一指标。吞吐若只测一次，只视作点估计，不把几个百分点的变化当成稳定结论。

## 起点：DFlash 很快，但接受长度明显低于 ViSpec

最初统一到相同推理引擎和 legacy5 参数后，4 样本快速对照显示：

| 方法 | 接受长度（含 1） | 吞吐 | 相对 baseline |
|---|---:|---:|---:|
| Cached autoregressive | 1.000 | 32.36 tok/s | 1.000× |
| DFlash | 1.884 | 63.51 tok/s | 1.963× |
| Domino（旧 checkpoint） | 1.848 | 53.08 tok/s | 1.640× |
| ViSpec chain（不开树） | 2.422 | 50.42 tok/s | 1.558× |
| ViSpec tree | 3.427 | 64.81 tok/s | 2.003× |

这一步发现了核心现象：DFlash 的 draft 本身非常快，因此接受长度稍微下降就会立刻损失吞吐；ViSpec 的 draft 较重，较长接受长度能抵消其额外成本。ViSpec/MTP 的高接受长度并不自动意味着更高线性 draft 质量，因为 tree 与 chain 的提案拓扑不同。

## 第一轮诊断：先区分数据、视觉内容和生成位置

### 同一 target 的 text-only control

MT-Bench、HumanEval、GSM8K 和 Alpaca 的平均接受长度为 1.7601，反而低于 legacy VLM 的 1.8833。这个结果不能证明文本天然更难，因为任务分布不匹配；但它否定了“去掉图片就会自动恢复到 LLM 高接受率”的简单解释。

### Image-token zero/shuffle

保持 target 看到正确图片，只修改进入 draft 的 target hidden states：zero 使五项均值下降 0.2998，shuffle 只下降 0.0044。由此确认视觉内容有用，但 flattened spatial order 基本没有被利用。

### 位置分桶

原始 DFlash 的接受长度从 token 33–64 的 2.0750 降至 token 129–256 的 1.7807。task-matched continuation 没有改善 1–32，最大的增益出现在后半段且主要来自 GQA，因此弱初始视觉 grounding 不是唯一主要原因。

## 数据和 decoding 调整只带来有限收益

### 1,280 条 task-matched continuation

使用短 VQA、OCR、caption、visual reasoning 和 long description 五类 target-generated 数据，从原始六 epoch DFlash warm-start 训练 400 steps。五项均值从 1.8833 提升至 1.9064，增量 +0.0231，低于预先设定的 +0.03–0.05 门槛。GQA 提升 +0.1038，但 CharXiv 下降 -0.0141，说明数据失配真实存在，却不是普遍瓶颈。

### Block size 4

将 block size 从 8 降到 4 会提高 normalized acceptance rate，却降低每次 verifier 实际发出的 token 数。原始 checkpoint 的单次吞吐有所上升，matched checkpoint 反而下降，因此 block size 4 不是稳定解法，后续继续使用 block size 8。

## 全局视觉压缩路线没有产生有效增益

### 两 query Q-Former

在原始 DFlash 上加入 cached two-query visual adaptor，并进行 adaptor-only 与 joint 两阶段训练。4 样本历史对照中，raw accept length 从原始 DFlash 的 0.8886 降到 Stage 1 的 0.8808 和 Stage 2 的 0.8582；Stage 2 吞吐也从 63.51 降至 59.32 tok/s。结果表明，额外 cross-attention 的成本没有由接受长度补偿。

### Masked pooling + 两层 MLP + zero-init gate

为降低 Q-Former 成本，改用 masked pooling、两层 MLP 和零初始化残差 gate，并从原始 DFlash warm-start 训练 3 epochs。10 样本对照中 raw accept length 为 0.883242，原始 DFlash 为 0.883256，基本完全相同；吞吐从 61.15 降到 59.66 tok/s。将 gate multiplier 设置为 0、0.5、1、2 也没有出现单调收益，2× 反而下降。

两次失败共同说明：把整张图片压缩成两个 query 或一个全局向量，会丢失 OCR 和空间关系所需的局部信息；继续增加 epoch 或放大 gate 不太可能挽救这条具体路线。

## Domino 证明了块内因果依赖的重要性，但收益依赖任务

旧 Domino checkpoint 将五项均值从原始 DFlash 的 1.8782 提高到 1.9382，但 offset 1 从 55.97% 降至 48.84%，说明旧 Domino backbone 本身较弱；它主要改善 offsets 2–7。

随后改为从 matched DFlash step 400 warm-start，只训练 256 维 projection 和 1,024 维 GRU 的四个 head tensors，共 200 steps。Stage 1 的平均接受长度达到 1.9684，三次吞吐为 64.10、63.91、63.41 tok/s，均值 63.80 ± 0.35；Stage 2 再 joint training 200 steps 后均值略降到 1.9653，因此停止 joint 路线。

Stage 1 的任务变化并不一致：COCO +0.0937、CharXiv +0.2777、MMMU-Pro +0.0650，但 GQA -0.0803、TextVQA -0.0205。Domino 适合作为已验证视觉 backbone 之后的第二阶段 causal correction，而不是诊断视觉瓶颈的第一底座。

## 从 layer routing 到 MRoPE：只有完整位置替换出现了空间信号

### Target-layer routing

只训练三个 scalar gates，让每个 draft layer 偏向配对的 target layer。100 steps 后 GQA raw acceptance 变化 -0.47%，TextVQA +0.16%，等同无收益。这说明简单选择 target layers 1、13、25 不能解决问题。

### Zero-init MRoPE gate

在冻结的一维 RoPE backbone 上训练三个 MRoPE interpolation gates。100 steps 后 gates 仍接近零，五项 raw mean 从 0.9064 降至 0.9012。冻结 backbone 无法靠局部小 gate 适应新的三轴几何。

### Hard MRoPE direct replacement

去掉 gate，在每层直接用 `[16,24,24]` 三轴 MRoPE，并联合训练完整 backbone。与 matched DFlash 使用同样的 1,280 样本、LR 3e-5 和 400 steps。raw mean 在 step 200 达到 0.9245，step 400 为 0.9249，相比 matched 0.9064 提升 +0.0185；其中 GQA +0.0678、MMMU-Pro +0.0290，但 TextVQA 和 CharXiv 略降。step 200 到 400 只增加 +0.00037，说明短 continuation 已经饱和。

## Hard MRoPE 与 Domino 组合只刷新了接受长度，没有刷新吞吐

冻结 Hard MRoPE step-200 backbone，再训练 Domino head 200 steps。组合的五项平均 emitted tokens 为 1.9761，略高于原 warm-start Domino 的 1.9684；平均吞吐为 64.19 tok/s，原 Domino 为 64.29 tok/s，因此没有速度收益。

| Benchmark | Hard MRoPE | Domino | 组合 |
|---|---:|---:|---:|
| GQA | 1.9753 | 1.8273 | 1.8054 |
| TextVQA | 2.0253 | 2.0000 | 2.0063 |
| COCO Caption | 2.0205 | 2.0984 | 2.1245 |
| CharXiv | 1.7204 | 2.0016 | 1.9753 |
| MMMU-Pro | 1.8810 | 1.9147 | 1.9692 |

组合改善 COCO 和 MMMU-Pro，却完全没有保住 Hard MRoPE 的 GQA 收益。平均 +0.0077 处于 10 样本实验的噪声级附近，说明两种机制并非简单可加。

## 当前状态：Hard MRoPE 已完成从随机初始化的全量训练，效果尚未评测

最终按原始 DFlash 配置从随机初始化训练了 3-layer Hard MRoPE draft：68,000 条数据、6 epochs、14,574 optimizer steps、micro batch 28、sequence length 3,072、block size 8、256 anchors、peak LR 6e-4。训练约用 9 小时 46 分。

最终 checkpoint 状态为 `epoch=6`、`global_step=14574`、`next_batch_idx=0`；consolidated 权重包含 737,702,656 个参数元素、36 个张量。纯 CPU 流式检查没有发现 NaN 或 Inf，checkpoint、optimizer 和 RNG 状态完整。

本次运行没有保存 `train.log`，W&B 关闭，也没有 validation dataset，因此无法复原 loss、accuracy、训练 accept length 或泛化曲线。**checkpoint 完整只说明训练技术上成功，不代表接受长度已经超过原始 DFlash；最终 benchmark 仍待运行。**

## 实验设计与验证方式

整体采用逐层排因而非一次加入多个模块：先统一推理引擎和 benchmark 参数，再做 text-only、zero/shuffle、matched-data、block size、generated-position 和 proposal-offset 控制；随后分别验证全局视觉压缩、块内 causal correction、target-layer routing 和 MRoPE。多数架构变体都先执行 CPU shape/parity/gradient 测试和 1-step smoke，再进行 100–400 steps 小训练。

关键对照尽量固定 target checkpoint、prompt、固定输出长度、SDPA 和 greedy verification。Domino throughput 在优化单步 GRU recurrence 后重复三次；其余不少吞吐仍为单次顺序测量，因此报告优先依据 deterministic acceptance counts。

## 限制与不确定性

- 多数正式诊断只有每个 benchmark 10 个样本；几千分之一到百分之一的接受长度差异不能视作稳定提升。
- 早期 4-sample 结果只用于快速筛选，不能与后期 10-sample 表格直接混合。
- position instrumentation 修复过 256-token 最后 partial block 的边界计数，因此旧表绝对值与新表略有差异，但 matched continuation 的相对增益没有变化。
- ViSpec tree、ViSpec chain 和 DFlash/Domino 的候选拓扑与 denominator 不同；系统吞吐可比较，接受长度的机制解释不能直接互换。
- Full Hard MRoPE 缺少训练日志和 validation 数据，无法判断收敛轨迹、过拟合或最佳 epoch。
- BF16 target parity 差异已按实验约定排除在本报告的主要判断之外。

## 建议的下一步

1. 首先使用与现有 legacy5 完全相同的 Transformers/SDPA、10×256 配置评测 full Hard MRoPE；先比较 acceptance，再对有希望的 checkpoint 重复 3 次吞吐。
2. 若 full MRoPE 没有超过原始 DFlash 至少 +0.03 emitted tokens，并且不能同时改善 GQA/TextVQA，则停止单纯 MRoPE 放大路线。
3. 下一项架构实验应在 DFlash backbone 上保留多个区域 token 和显式二维位置，使用 region-aware、token-conditioned visual path；不要再压缩成单个全局向量。
4. Domino 只作为第二阶段增强：先验证新的视觉 backbone，再冻结它训练 head-only；不要直接 joint training。
5. 新训练必须保存 `train.log` 或启用离线 W&B，并加入小 validation split，以便判断最佳 epoch，而不是只保留最终 checkpoint。

## 仍需回答的问题

- Full Hard MRoPE 能否把小规模 GQA 空间收益扩展到 OCR 和 reasoning，而不是只记住长描述数据？
- Shuffle 几乎无影响究竟来自位置编码丢失，还是 target hidden states 已将空间信息混入内容通道，使简单 row reversal 不足以破坏语义？
- Region-aware 路径应保留多少区域：4×4、动态 top-k，还是按原始 vision grid 分层保留？
- VLM 接受差距中，模型不确定性与 benchmark 答案多样性占多少，draft 架构损失又占多少？
- 在接受长度相近时，能否用更轻的 causal correction 取代当前 GRU，避免 Domino 的串行推理成本？
