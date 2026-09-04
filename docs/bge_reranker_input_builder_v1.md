# BGE Reranker Input Builder v1

## 范围与实验假设

只新增shadow脚本与测试，不修改生产Retrieval、Assembly、Packing、Prompt、Skills或已有reranker实现。输入固定为`reports/reranker_shadow_v1_rrf_top120.json`的30题/3600个chunk。禁止重新召回、调用Jina/LLM/Judge/LangSmith或运行100题。

原BGE对577/3600个query/chunk pair执行了1024-token截断。截断可能丢失后部事实，但**不预设截断就是BGE与Jina差距的主因**。本轮只测试一个预先确定的表示，不根据逐题答案或gold调权重。

## 唯一处理变量

- 模型继续使用本地`models/bge-reranker-v2-m3`：CUDA FP16、batch=4、max_length=1024，每题120对。不开大token窗口，不增加多窗口推理次数。
- 原始query及候选集合、候选ID、RRF顺序完全不变。
- 不超过pair预算的短chunk逐字不变。
- 超长chunk使用本地tokenizer预算，保留约96-token原始开头；根据通用query词重叠和chunk内词频选择原文句/行及相邻片段，按原文顺序拼接。
- 常规表格行保留原样。极长行只能在空白处分段，记录`forced_long_row_splits`；不重新格式化负号、数字或单位，不声称完全恢复了表格结构。
- 输入只有question与source_text，不读取公司、指标词典、FinanceBench ID、gold或参考答案。通用词匹配允许使用问题中已有的年份及实体词，但没有任何实体特判。
- 每对记录原始/最终token数、原文source_spans、匹配词、丢弃字符数。零tokenizer截断不代表零信息损失；报告另列原文gold片段恢复/损失的离线诊断。
- **重排后仍从冻结快照取完整原文**，沿用旧实验Top8 chunks、最多28000字符context投影。不能将该指标当成生产答案准确率。

## 运行

仅生成输入表示与token/可见性审计，不加载torch或模型：

```powershell
conda run --no-capture-output -n rag python -u scripts/evaluate_bge_input_builder_v1.py --prepare-only
```

执行/续跑固定30题本地BGE重排：

```powershell
conda run --no-capture-output -n rag python -u scripts/evaluate_bge_input_builder_v1.py
```

仅重新生成报告，无模型或API调用：

```powershell
conda run --no-capture-output -n rag python scripts/evaluate_bge_input_builder_v1.py --summarize-only
```

每题保存checkpoint。恢复时跳过已准备/已推理题；输入、模型、tokenizer、脚本或旧对比报告的指纹改变会拒绝混用checkpoint。旧identity/BGE/Jina报告只读，Jina配置和密钥完全不加载。新输出默认`reports/bge_reranker_input_builder_v1.json`与同名Markdown。

为避免再次造成机器卡顿，准备阶段要求至少1GiB可用内存；模型加载前要求至少4GiB可用内存与4GiB可用显存，逐题要求至少1GiB可用内存。这是保守门槛，不保证峰值一定足够。资源不足时明确停止，保留checkpoint，不关闭用户进程或切换CPU。torch CPU线程设为2以限制本地辅助开销，不改变模型打分定义；运行延迟与旧实验不做严格因果比较。

## 结果判读

对比cached raw BGE与input-v1 BGE，Jina/identity仅作已有参照。报告必须同时看：

- Gold page/chunk rank、page hit@5/10/20、context page/span hit。
- selection-loss10、correct-regression10、candidate-miss10，以及逐题新增/回退。
- tokenizer截断数、表示中事实片段损失、表示构造耗时、GPU推理耗时。
- 不完整运行仅在共同完成样本中比较，不把未运行当错误。

只有局部30题输入实验，不能证明未见公司泛化或生产答案正确率。即使平均改善，也需检查逐题回退；如果有限或失败，停止本轮，不继续根据这30题调参数，不接入生产。

## 已完成的30题结果

本轮已完成30/30。本地模型实际推理确认`truncated_pairs=0`。旧identity/BGE/Jina排名全部从已有报告复用，本轮外部API调用为0。

| 指标 | 原BGE | BGE Input Builder v1 | Jina（历史缓存） |
|---|---:|---:|---:|
| Candidate gold page hit | 26/30 | 26/30 | 26/30 |
| Page hit@5 | 7/30 | 7/30 | 18/30 |
| Page hit@10 | 9/30 | 9/30 | 21/30 |
| Page hit@20 | 13/30 | 13/30 | 26/30 |
| Shadow context hit | 7/30（23.33%） | 7/30（23.33%） | 19/30（63.33%） |
| Context evidence span hit | 6/30 | 6/30 | 16/30 |
| Gold page rank均值（命中26题） | 24.62 | 24.88 | 4.85 |
| Gold chunk rank代理均值（命中22题） | 23.14 | 23.59 | 5.68 |
| Tokenizer截断pair数 | 577/3600 | 0/3600 | 不适用：不同后端 |

分组context命中均未改变：candidate-miss为1/10、selection-loss为1/10、correct-regression为5/10。逐题context新增0、回退0。但4题gold page排名小幅变差：`00222` 39→42、`00566` 9→10、`03882` 14→15、`00702` 26→28（此处为报告标识，不参与实现）。不能把“context不回退”解释为“所有排名都不回退”。

### 为什么没有改善

1. 577对占全部输入16.03%，其余3023对本来就没有截断，本轮原样保留。
2. 超长输入平均1069.13 token、中位数1057、范围1025–1366。多数只是略超1024，不能从截断数量推断大量关键表格内容被删除。
3. 在当前≥40字符完整参考原文行代理下，完整原chunk中可匹配的24对，在原1024截断可见文本与新表示中均仍可匹配：恢复0对、损失0对；577个改动pair中仅1对满足该代理。即本次修改没有增加该代理能识别的gold事实暴露。
4. 新表示移除140077个原文字符，添加片段间隔后实际输入字符为11364784（原11501851）。平均pair长度757.62→748.58，仅约1.19%下降。

**结论：本方案未缩小Jina gap，不值得接入生产。** 结果不支持“只要消除这577次截断，本地BGE就能接近Jina”的假设；但也不能证明所有输入表示优化或其他本地模型都无效。原文行代理不等于所有数字、表格关系和财务事实的完整验证，不能断言截断永远没有影响。下一阶段如继续，应另立假设，避免继续围绕同一30题调词匹配权重或窗口参数；本轮停止开发与试跑。

### 成本与验证

- 输入构造平均897.60ms/题（包括token计数和离线gold可见性诊断，不是纯生产构造耗时）。
- 本轮记录的BGE调用耗时总计167.14秒；首题45.44秒包含导入/加载，后29题平均4.20秒。保存报告和全量校验耗时另计。
- 旧BGE热运行约4.41秒/题；机器负载和CPU线程配置不同，不能将差异全部归因于新表示。
- 64项相关测试通过，覆盖真实本地tokenizer预算、原文offset、晚出现数字保留、短输入不变、异常输入、篡改检测及旧shadow指标契约。
- 已离线重算120组排名（4组×30题）；每组都是同一120个候选的完整排列，候选哈希和指标一致。
- 生产`backend/**/*.py`聚合SHA256仍为`4134c50c9365de480bb959c76ac3b97933178940f0f5d867a31d3a3d5a0fa3be`，包含用户原有未提交修改，未覆盖。
- GPU推理和校验进程已正常退出；退出后未发现Python评测进程，GPU检查为0MiB/0%。未清理用户进程，未提交commit。

完整逐题输入表示、source spans、120项打分与指标见`reports/bge_reranker_input_builder_v1.json`；数据汇总见同名Markdown。已完成，无需重跑30题。
