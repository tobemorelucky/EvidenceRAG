# Evidence Architecture v5 — Table Evidence Group 离线审计

> 本报告仅从 PostgreSQL 读取 `document_tables`，没有写数据库、没有修改生产 pipeline，也没有调用 Dense/BM25/RRF/Jina/LLM/Judge。

## 结论

- PostgreSQL 实际读取 `1936` 张表，来自 `39` 个 document。
- 构建 `1899` 个 group，其中多 table group `22` 个，跨页 group `22` 个（`1.16%`）。
- 平均每组 `1.0195` 页、`1.0195` 张表；最大 group `5` 页。
- 验收指标没有提升：可信 TableStore gold-page coverage 仍为 `12/30`，group 后仍为 `12/30`。
- 覆盖结论必须按证据强度分层。`adjacent_table_upper_bound` 只是无条件相邻页的理论上限，不算 Table Evidence Group 已实现的覆盖。

## Diagnostic30 gold-page coverage

| 口径 | 覆盖 | 含义 |
|---|---:|---|
| `direct_table_member_coverage` | 12/30 (40.00%) | 表实际绑定到 gold page |
| `declared_table_range_coverage` | 12/30 (40.00%) | 表的 start/end page 声明覆盖 gold page |
| `continuation_evidenced_group_coverage` | 12/30 (40.00%) | 声明范围或明确 continuation 线索覆盖 |
| `adjacent_table_upper_bound` | 26/30 (86.67%) | 任一相邻页有表（仅理论上限） |

### 恢复题目

- `declared_range`: 无
- `continuation_evidence`: 无

## Group 构建规则

- `document_id` 必须完全一致，`page_id` 必须存在，并且仅比较相邻的 0-based `page_number`；link trace 显式记录 `page_id_contiguous`。
- 使用 table title、headers、边界 nearby text 的相似度与 continuation 关键词加权打分。
- 阈值为 0.48，并要求 title/header 或 continuation+nearby text 的独立佐证。
- 无 continuation 时要求近乎一致的 title+header，或由边界 nearby text 共同佐证；相同年份列本身不参与相似度。
- 同一相邻页对采用一对一最高分匹配，避免把同页多个无关表合并成一个大组。
- group ID 由 document_id 与有序 table IDs 确定性哈希生成；本实验不写回数据库。

## Link 统计

- Accepted links: `37`
- Link reasons: `{'header': 37, 'continuation': 20, 'title': 21}`

## 跨页 groups（前 30 个）

| Group | Document | Pages | Tables | Mean quality | Link score |
|---|---|---|---:|---:|---:|
| `tg_85567f1b07de5d0876a3` | `AES_2022_10K.pdf` | 136–137 | 2 | 0.511 | 0.566 |
| `tg_2c7eb6bf826ce50fc88f` | `AES_2022_10K.pdf` | 139–142 | 4 | 0.584 | 0.587 |
| `tg_5d8f59a3d98c00e4a720` | `AES_2022_10K.pdf` | 144–146 | 3 | 0.582 | 0.599 |
| `tg_7c3f394b84ba71e7a24d` | `AES_2022_10K.pdf` | 150–153 | 4 | 0.583 | 0.850 |
| `tg_96ddc7801835ed424c5b` | `AES_2022_10K.pdf` | 155–156 | 2 | 0.566 | 0.500 |
| `tg_1cdd6a06417eedf9bcc4` | `AES_2022_10K.pdf` | 167–168 | 2 | 0.512 | 0.500 |
| `tg_b71c2232bac07c6c70d1` | `AES_2022_10K.pdf` | 177–179 | 3 | 0.600 | 0.603 |
| `tg_1a099c4148e083b37115` | `AES_2022_10K.pdf` | 178–179 | 2 | 0.564 | 0.725 |
| `tg_06fc6b7c182a079eaa5f` | `AES_2022_10K.pdf` | 183–185 | 3 | 0.502 | 0.600 |
| `tg_7769dc086c94d11fd35b` | `AES_2022_10K.pdf` | 190–191 | 2 | 0.577 | 0.521 |
| `tg_e78e0ac9522eec0636aa` | `AES_2022_10K.pdf` | 194–195 | 2 | 0.626 | 0.850 |
| `tg_0244337d9869079f3482` | `AES_2022_10K.pdf` | 197–198 | 2 | 0.577 | 0.634 |
| `tg_ee934b7c986bc45ecb7a` | `JOHNSON_JOHNSON_2022_10K.pdf` | 45–46 | 2 | 0.699 | 0.525 |
| `tg_0e6a08f7595cb4dfbea3` | `JOHNSON_JOHNSON_2022Q4_EARNINGS.pdf` | 9–10 | 2 | 0.700 | 0.700 |
| `tg_01308b6c1e1280ff2dc6` | `JOHNSON_JOHNSON_2022Q4_EARNINGS.pdf` | 21–22 | 2 | 0.700 | 0.700 |
| `tg_dafd7d2f7489da299273` | `JOHNSON_JOHNSON_2023_8K_dated-2023-08-30.pdf` | 19–23 | 5 | 0.700 | 0.700 |
| `tg_f4860cd37f325298dfca` | `JOHNSON_JOHNSON_2023_8K_dated-2023-08-30.pdf` | 19–23 | 5 | 0.427 | 0.679 |
| `tg_06e89a3eb9d4dc83e9fa` | `JOHNSON_JOHNSON_2023_8K_dated-2023-08-30.pdf` | 24–26 | 3 | 0.588 | 0.700 |
| `tg_10b2e1648545587e51f1` | `JOHNSON_JOHNSON_2023_8K_dated-2023-08-30.pdf` | 24–26 | 3 | 0.592 | 0.700 |
| `tg_678a56d4b627679bada6` | `JOHNSON_JOHNSON_2023_8K_dated-2023-08-30.pdf` | 24–25 | 2 | 0.594 | 0.700 |
| `tg_f750414cb12789b5639e` | `PEPSICO_2023Q1_EARNINGS.pdf` | 5–6 | 2 | 0.700 | 0.778 |
| `tg_764ec6cf6ca6cb1c0aae` | `VERIZON_2022_10K.pdf` | 85–86 | 2 | 0.581 | 0.700 |

## 判断与下一步

- 如果可信 group coverage 高于 12/30，说明跨页范围/continuation 能恢复一部分错误页关联，可进入 shadow table-group retrieval。
- 如果仍为 12/30，而相邻页上限明显更高，说明瓶颈是 parser 的 page/table 关联或 gold 页附近未抽取出结构，不能靠检索模型或无条件 page±1 修复。
- 下一步仍应先做离线人工抽样核验跨页 group precision；通过后才设计 shadow collection，不能直接接入生产。
