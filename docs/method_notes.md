## Method Note: Repetition-Aware Text Compression with Modality Dropout

### Motivation
当前 Text 特征由一个句向量沿时间维重复 10 次。团队模型为这些重复向量加入不同位置编码，并将 10 个 Text token 的加权输出累计到融合表示中。此前单独提高 Text 整模态 Dropout 优先级未产生明显改善，因此本方法直接处理重复表示。

### Group Method
团队方法采用 reliability-gated fusion 和 uniform modality dropout，Text 与其他时序模态均使用 10 个 token。

### Proposed Improvement
保持磁盘 Text 特征形状 `[10, D]` 不变，在模型内部、投影和位置编码之前沿时间维求均值，得到 `[1, D]`。单个 Text token 参与 reliability gate、Transformer 编码和融合，并继续配合整模态 dropout。

### Implementation
- `use_text_compression: false`：复现团队模型。
- `use_text_compression: true`：Text 使用一个均值 token，且不添加时间位置编码。
- 缺失 Text 仍执行 zero-fill 并更新 `modality_mask`。
- 个人输出全部保存到 `results/term_paper/`。

### Related Files
- `src/models/improved_model.py`
- `src/training/improved_experiment_runner.py`
- `configs/term_paper_text_compression.yaml`
- `experiments/run_term_paper_text_compression.py`

### Expected Effect
主要针对重复 Text 证据和 Text 缺失鲁棒性，同时观察 clean performance 是否保持。预期效果不是正式结果。

### Current Result
尚无正式结果。本机缺少原始数据、缓存和依赖，本次未运行训练或测试。

### Limitations
压缩只能减少结构性放大，不能在 Text 完全缺失时恢复语义信息；若其他模态区分能力不足，提升可能有限。

### Report Description Draft
We propose repetition-aware Text compression for robust multimodal intention recognition. The existing pipeline repeats a sentence embedding over ten time steps, while the fusion model processes these repeated vectors as separate tokens with distinct positional embeddings. Our method preserves the cached feature protocol but averages the repeated Text sequence into a single token before projection and multimodal encoding. The compressed representation is combined with reliability-gated fusion and whole-modality dropout to reduce duplicated Text evidence and encourage complementary learning from non-Text modalities.
