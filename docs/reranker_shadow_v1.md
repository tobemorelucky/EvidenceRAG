# Reranker Shadow Evaluation v1

本工具与生产pipeline隔离，不修改Retrieval、Ranking、Assembly、Packing、Fusion、Prompt或Skills。

## 当前执行状态

经用户授权，已只对固定30题执行一次现有Dense+BM25+RRF召回，写入统一快照`reports/reranker_shadow_v1_rrf_top120.json`：30题、3600条完整chunk、11501851个原文字符。SHA-256：`ee983dcb4d052ffab1ccefacc87ce6d2f80b276e0704c44110247bf6731a48cd`。

本次重新生成快照的原因是旧资料不含完整RRF Top120正文：

- `reports/retrieval_funnel_audit.json`：Dense/BM25/RRF命中排名统计，没有120个chunk正文。
- `reports/evidencerag-rag-core-v3-skills-all100-final_answers.jsonl`：`initial_retrieved_chunks`为RRF Top60，不是Top120。
- 近几轮9205个Evidence Units及Dense Primary实验：来源不是相同的RRF Top120，不能替代。

已发现Jina配置与本地`models/bge-reranker-v2-m3`权重；不记录密钥。

Identity、BGE、Jina均已完成30/30。2026-09-03用户授权更新本地密钥后，只补跑最后一题`financebench_id_00476`：Jina成功，用时10.54秒，报告100590 token；其余89项直接复用，未重新检索或加载本地模型。此前最后一题HTTP 403明确为账户余额不足，历史错误继续保留。结果保存在`reports/reranker_shadow_v1.json`及同名Markdown。

完成的90组排名已逐项验证：每组是原120个候选的完整排列，候选哈希一致，离线重新计算的指标与保存值一致。生产`backend/**/*.py`聚合SHA-256在原实验执行前后相同：`4134c50c9365de480bb959c76ac3b97933178940f0f5d867a31d3a3d5a0fa3be`。补跑只更新本地密钥及报告。

## 当前结果与边界

- 固定30题candidate hit：26/30（86.67%）；三组重排均不能恢复这4题缺失页面。
- 原RRF：gold chunk rank代理均值47.14（命中22题），gold page rank均值45.77（命中26题），shadow context hit 6/30。
- BGE：相同分母下对应均值23.14、24.62，shadow context hit 7/30；selection-loss组仍为1/10。排名改善，但不足以证明解决了选择问题。
- 三组共同完成30题：shadow context hit为identity 6/30（20%）、Jina 19/30（63.33%）、BGE 7/30（23.33%）。Jina在selection-loss完整10题上达到8/10，原RRF与BGE均为1/10；Jina相对identity新增15题命中、回退2题，详见报告。
- Jina gold page rank在相同命中26题上，均值从45.77前移到4.85；page hit@5/@10/@20为60%/70%/86.67%。这支持进一步验证强reranker，而不是证明答案准确率达到63.33%。
- Jina成功30题共报告2879092 token；这是reranker用量，不是回答LLM用量。历史429、SSL失败、403均保留在结果中。
- BGE在CUDA运行，577/3600对输入超过1024-token窗口并被截断；首题含加载，后29题平均约4.41秒/题。
- 三组30题均完成；没有运行100题，没有调用LLM、Judge、LangSmith。

## 工具与运行顺序

1. 如果提供已有RRF Top120快照，直接使用它，不重新检索。
2. 如果允许新建快照，显式运行一次下面的命令。它使用现有`MilvusManager.hybrid_retrieve`，不改变其实现；Dense/BM25内部各取240，RRF k=60，最终120。该快照是新实验基线，不能冒充历史排名。

```powershell
conda run --no-capture-output -n rag python -u scripts/freeze_rrf_top120_shadow_v1.py --allow-retrieval-snapshot
```

3. 三组读取同一个快照；不再检索，不加载TableStore，不调用回答模型、Judge或LangSmith：

```powershell
conda run --no-capture-output -n rag python -u scripts/evaluate_reranker_shadow_v1.py
```

输出`reports/reranker_shadow_v1.json`与同名Markdown。缺少快照时在初始化模型、调用API前报错。

快照已经生成，**不要再次运行freeze命令**。重跑默认三组命令将复用同一输出中的成功结果。

本次全量120项/题触发过Jina HTTP 429，因此续跑仅调整请求间隔（8→35→60秒），不改变输入或模型，也不重跑已成功项：

```powershell
conda run --no-capture-output -n rag python -u scripts/evaluate_reranker_shadow_v1.py --jina-interval-seconds 60
```

**最后一题已成功补齐，无需重跑。** 上面命令保留为断点续跑入口；当前90项成功结果均会复用。

仅重新校验/生成报告、不发任何模型/API请求：

```powershell
conda run --no-capture-output -n rag python scripts/evaluate_reranker_shadow_v1.py --summarize-only
```

## 三组独立实验入口

以下三个命令都默认读取同一快照；独立输出避免与默认三组合并报告的manifest冲突。它们属于新重排运行，Jina会重新计费；仅查看已有结果无需运行。

```powershell
conda run --no-capture-output -n rag python -u scripts/evaluate_reranker_shadow_v1.py --backends identity --output reports/reranker_shadow_v1_identity.json
conda run --no-capture-output -n rag python -u scripts/evaluate_reranker_shadow_v1.py --backends jina --output reports/reranker_shadow_v1_jina.json
conda run --no-capture-output -n rag python -u scripts/evaluate_reranker_shadow_v1.py --backends bge --output reports/reranker_shadow_v1_bge.json
```

本地BGE调用实现在`scripts/shadow_rerankers_v1.py`的`BGEReranker`：本地`AutoTokenizer`将query/text成对编码，`AutoModelForSequenceClassification`推理相关性logit，按分数降序排列，分数相同时保留输入顺序。GPU使用FP16；不生成回答。

## 固定契约

- 只接受固定candidate-miss10、selection-loss10、correct-regression10，共30题。
- 每题120个唯一chunk，保留原文、chunk_id、文件及内部0-based页码、RRF rank、内容哈希。
- Identity保留原RRF顺序。
- Jina每题一次请求，对全部120项排序；请求间隔通过`--jina-interval-seconds`配置，无隐藏重试或local fallback。API失败明确记录并停止继续请求该backend；显式续跑才重试失败项，历史错误保留在`prior_errors`。
- BGE使用本地权重，禁止自动下载，优先GPU，batch=4、max_length=1024；全部120项参与，另记token截断数。它与Jina的模型输入token窗口不完全相同，须在结果中检查这一限制。
- 后端只接收query和chunk文本，绝不接收gold/参考答案。gold仅在排序返回后参与指标计算。
- 每个backend/题完成后保存checkpoint；重跑同一命令复用成功结果、重试失败项。不同评分配置/输入需要新`--output`；只有请求间隔这种不影响评分的运行参数允许变更，每题记录实际间隔。
- 任一必需backend未完成30题，进程退出码为2；无Jina配置时允许明确跳过Jina。

## 指标定义

- `gold_chunk_rank`：FinanceBench无chunk ID标注，因此使用**同gold页内完整命中至少40字符参考原文行**的代理。未命中为null，不能解释为真正chunk relevance。
- `gold_page_rank`、`page_hit_at_5/10/20`：重排chunk按页面首次出现去重后的rank与any-gold-page hit。
- `context_hit`：固定Top8 chunks、最多28000字符的shadow投影中是否出现gold页。**不是生产Assembly/Packing的真实context**。
- `context_evidence_span_hit`：同样投影截断后是否仍保留上面的原文片段。
- 逐题保留全部120个index/score、原输入哈希、错误/跳过状态与耗时。成功样本均值必须与完成数一起看，不能用不同完成题集合直接宣称提升。

本轮只能验证reranker对冻结候选排序的作用，不能直接证明生产答案正确率提升；不接入生产。
