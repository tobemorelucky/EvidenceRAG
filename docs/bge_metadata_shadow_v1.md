# 最后一次本地 reranker shadow：Metadata-aware v1

## 结论

固定30题已完成。轻量metadata融合有局部收益，但仍不能接近Jina，且存在逐题回退。本轮停止，不追加权重/别名/指标规则，不接入生产。

| 指标 | 原BGE | Metadata BGE | Jina历史缓存 |
|---|---:|---:|---:|
| Context hit | 7/30（23.33%） | 10/30（33.33%） | 19/30（63.33%） |
| Context原文片段hit | 6/30 | 9/30 | 16/30 |
| Page hit@5 | 7/30 | 6/30 | 18/30 |
| Page hit@10 | 9/30 | 11/30 | 21/30 |
| Page hit@20 | 13/30 | 15/30 | 26/30 |
| Gold page rank均值（命中26题） | 24.62 | 21.35 | 4.85 |
| Gold chunk rank代理均值（命中22题） | 23.14 | 19.55 | 5.68 |

Candidate hit三组均26/30，候选没有变化。Context hit提升10个百分点，仍落后Jina30个百分点；这不是答案正确率。Page@5下降说明头部页面排序不是全面改善。

## 本轮唯一配置

只新增`scripts/bge_metadata_reranker_v1.py`和独立评测入口，不导入生产Retrieval、Assembly、Packing、Prompt、Skills。不重新运行BGE模型，复用原实验的BGE分数/排名；Input Builder结果不参与本轮打分，避免叠加变量。

`final_score = 0.75 × BGE_rank_percentile + 0.25 × mean(active_metadata_features)`

- BGE logits只用于恢复已有排序，再转排名百分位；不把不同问题的未校准logit直接当概率。仅此单调转换不会改变BGE排序，但会丢弃原logit间距，本轮固定接受这一限制。
- Entity：用候选`company`构建局部名称集合，与问题按通用标点、空白和法人后缀归一化匹配。匹配+1；目标已解析但候选公司不同−1；目标/候选未知0。不猜缩写，不用外部公司词典，不硬过滤候选。
- Period：只识别问题显式年份/FY形式。局部匹配句行及相邻行的年份覆盖为强信号，chunk其他位置半权，`report_year`四分之一权。报告年份不匹配不等于该页没有所需历史数字，不作负向硬约束。
- Metric：没有现成可靠的独立metric字段，使用去通用任务词/实体/年份后的问题词与句行的IDF加权重叠。无财务指标、公式或公司特判，不做同义词扩展。
- 信号缺失不硬拒绝，始终返回原120个候选完整排列；同分按原RRF index排列。
- 特征来自完整原chunk及已有metadata；最终context仍沿用冻结原文Top8 chunks/≤28000字符。不是新的生产context构造。
- gold/参考答案只在排序结束后用于离线评估，算法不读取FinanceBench ID、gold或参考答案。

## 逐题变化与限制

| 分组 | 原BGE | Metadata BGE | 新增 | 回退 |
|---|---:|---:|---:|---:|
| candidate-miss10 | 1/10 | 1/10 | 0 | 0 |
| selection-loss10 | 1/10 | 4/10 | 3 | 0 |
| correct-regression10 | 5/10 | 5/10 | 1 | 1 |

新增：`00540`、`00566`、`01351`、`00215`；回退：`00382`。这些标识只用于报告，不存在算法条件分支。

观察到的候选chunk排名变化：

- `00566`：目标候选9→6，局部债务词和问题两个年份共现，进入Top8。
- `01351`：目标候选14→7，局部指标短语重叠较高，进入Top8。
- `00215`：目标候选9→6，同公司与局部年份信号帮助进入Top8；但同页其他行也可能只是共享“capital”词，并非验证了所有操作数。
- `00540`：目标候选11→6；最匹配局部片段是公司/报表措辞，并非所问比率的必要字段。这项page命中提升不能作为精确metric约束成功的证据。
- `00382`：目标候选4→9，被挤出Top8。问题名称缩写没有与候选完整公司名解析为同一实体，entity信号未启用；局部词面与年份信号不足以保护原高排名。**不为这个案例增加公司缩写补丁。**

因此，correct-regression组总数不变不代表逐题无回退；metadata代理既可能帮助，也可能把更符合表面词/年份的错误页面排到前面。

## Metadata审计

- 所有3600候选均有company，只有26/30问题能按当前通用名称匹配解析目标；其余4题保留unknown，不把它们判为公司错误。
- 28/30问题包含可识别的显式年份。
- 所有候选table_title为空，没有可靠表格标题可用于metric增强。
- 528对有局部年份共现，1485对仅在chunk其他位置匹配，373对仅报告年份弱匹配；其他为未要求年份/不可见/未知。
- 2666对有非零metric词面重叠，不能将非零重叠解释为指标事实正确。

这些统计支持“metadata有用但不够可靠”的判断，不支持“entity / period / metric精确绑定已经解决”。未做单信号消融，不能将全部收益单独归因于entity、period或metric。

## 成本、边界与验证

- 本轮BGE/Jina/LLM/Judge/LangSmith调用均0；无额外token或GPU模型加载。
- 单题本地特征提取/排序平均约116.81ms，不含读JSON、gold评估和写报告。
- 全部4组×30题排名已重算，每组是同120候选的完整排列；原快照和旧对比报告哈希不变。
- 39项相关测试通过，包含缺失metadata、跨报告年份、未知缩写、全候选保留、gold字段隔离、固定权重、非金融通用词、指标与回退统计。
- 生产代码未修改，用户历史未提交更改保留；没有commit。只运行这一个配置，没有参数搜索或新增完整benchmark。
- 此30题属于反复使用的开发诊断集，不能证明未见公司泛化、统计显著性或生产答案正确率。

## 文件与入口

数据报告：`reports/bge_metadata_shadow_v1.md`。

完整120项排名、每个候选的原BGE分数/排名、entity/period/metric代理、最终分数、问题及逐题指标：`reports/bge_metadata_shadow_v1.json`。

本次已经运行完成，无需重跑。复现实验入口如下（默认已有报告会拒绝覆盖；如需重新执行同一固定配置，显式指定新的`--output`）：

```powershell
conda run --no-capture-output -n rag python -u scripts/evaluate_bge_metadata_shadow_v1.py
```

结论：结束本地启发式reranking实验，保留此shadow作为诊断成果；本轮不继续优化、不替代Jina、不接生产，也不触发新的远程评测。
