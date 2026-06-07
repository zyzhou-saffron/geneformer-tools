# Changelog

`geneformer-tools` —— 把 IBD 在硅扰动(ISP)研究代码从 jupyter notebook 和 site-packages 散落处,
迁成版本管理 Python 包的过程记录。来源:2030 行的 `geneforemr.isp.workflow.ipynb` +
site-packages 里的 `mynewfun.py` / `isp.py` / `isp_dual.py`。

**迁移纪律**:每块「抽取 → 在真实数据上证明与 notebook 原版逐元素等价 / 可跑通 → commit」。
区分「值得成为可复用资产的自定义逻辑」与「标准 API 单次调用 / 一次性脚本」——后者只修路径、不强行包装。

## 模块总览

| 模块 | 内容 | 验证 |
|---|---|---|
| `perturb.py` | `DualPerturber`(删基因1 + 过表达基因2 的双扰动子类) | 真模型 sweep 跑通 |
| `analysis.py` | `ResultAnalysis`(Open Targets + EuropePMC + NCBI 文献分析) | API key 走 env |
| `config.py` | 路径/密钥集中配置;`FINETUNED_MODEL` 指向 checkpoint-3761 | — |
| `gene_pairs.py` | `valid_pairs`(同细胞共现,有序/无序)+ `valid_pairs_cross_celltype`(跨细胞选对器) | 逐元素等价 17606/5628/25012;全量跑通 21145/42290/25012 |
| `state_embs.py` | `compute_state_embs`(状态锚点,薄封装 EmbExtractor) | 与内联等价;真 checkpoint 端到端跑通(256 维) |
| `isp_stats.py` | `isp_stats_to_goal_state`(改写的官方 stats,两分支)+ `compute_isp_stats` | 逐元素等价(分支一 32740×4、分支二 3000×8) |
| `isp_runner.py` | `run_isp_sweep`(按 perturb_type 分支)+ `prepare_and_run_isp`(端到端) | dual/overexpress 两分支真跑;全流水线打通 |
| `data_prep.py` | `build_gene_to_ensembl_map` / `prepare_adata_for_geneformer` / `tokenize_h5ad` | prepare 与内联等价;tokenize 跑通;biomart 67119 条 |

`scripts/run_isp.py` / `run_isp_dual.py`:`__main__` 守卫的瘦包装 → `prepare_and_run_isp`。
site-packages `geneformer/mynewfun.py`:re-export 本包的 shim。

---

## 2026-06-07

### data_prep.py(`beb381a`,step 7a)
- 抽前段流水线里**真正可复用**的 3 个函数:biomart 基因名→Ensembl 映射、AnnData 补 geneformer 字段、
  `TranscriptomeTokenizer` 薄封装。前段其余(一次性读写/出图/坏 cell/mouse)留 notebook。
- 验证:`prepare_adata` 与内联逐值等价;`tokenize_h5ad` 200 细胞产出 input_ids;biomart 联网 67119 条(TNF/STAT3 映射正确)。
- 延后:`celltype.py` 本体映射(cl.obo 缺失);cell 分类复用 geneformer skill 不重写。

### ISP notebook 瘦身(`6f2dd4a`,step 6)
- ISP 段(cell 42–94)109→89 cells:删 20 cell(手动拆解 `perturb_dual` 内部、importlib 调试 scratch、悬空 MD),
  换 3 cell(扰动 sweep→`run_isp_sweep`、stats→`compute_isp_stats`),修 2 cell 路径。
- 零语法错误,流程连贯;无关探索段(h5ad/tokenize/细胞分类/检测测试)一律不碰。备份 `*.bak-step6-slim`。
- notebook 本身不在本 repo(在 `~/ipynb`),仅在 PLAN 记录。

## 2026-06-06

### isp_runner.py(`90820d1`,step 4 + 5)
- 合并 `run_isp.py`(overexpress)+ `run_isp_dual.py`(dual)两个 ~80% 重复的 sweep 脚本为一个参数化 `run_isp_sweep`,
  复用已验证的 `compute_state_embs` / `valid_pairs`,不再内联重复。两脚本降为瘦包装。
- 验证:dual(`DualPerturber.perturb_dual`)/ overexpress(`InSilicoPerturber.perturb_data`)两分支真模型产出 `_raw.pickle`;
  全流水线打通(valid_pairs→sweep→pickle→`compute_isp_stats`→CSV,TNIP1×MTMR3 = -0.000784)。
- 跑中发现并修:`compute_state_embs` 不自建输出目录;`perturb_data` 在无 `__main__` 守卫脚本里 spawn 重执行。

### isp_stats.py(`9a0051d`)
- 抽用户改写的 `isp_stats_to_goal_state`(产出 goal-state 偏移结果表),逐字搬运(仅把全局 `cos_sims_df_initial` 改成参数)。
- 验证:分支一全量 32740×4、分支二子集 3000×8 均与 notebook 原版逐元素一致。
- **纠错**:分支二曾被静态误判为「坏」(引用未定义 names/results_df);真跑证明两分支都正常(那两名字只在 alt_states 非空时引用)。

### state_embs.py(`1dfdffd`,step 2)/ gene_pairs 跨细胞澄清(`628876c`)
- `compute_state_embs` 与内联产出的 `state_embs_dict` 逐元素一致。
- `valid_pairs_cross_celltype` 标注为**只是选对器**:配套的跨细胞扰动执行 notebook 里从未实现,当前无消费者。

### gene_pairs.py(`0b74952` + `d7cfbb7`,step 1)
- 合并 notebook 里 3 份内联 `valid_combinations` 实现为 `valid_pairs(ordered=)`;方案二(异细胞)单列 `valid_pairs_cross_celltype`。
- 验证:三方案逐元素等价(17606/5628/25012),notebook 对应 cell 已替换为 import,全量数据跑通(21145/42290/25012)。

## 2026-06-05

### 项目骨架(`9a1c7a9` → `0fb68fa`)
- 从 site-packages / 散落处把 `DualPerturber` / `ResultAnalysis` / `isp.py` / `isp_dual.py` 搬进 git 管理的最小骨架,
  配 `config` / `.gitignore` / editable install(`python -m pip install -e .`)。
- `OPENTARGETS_API_KEY` → `NCBI_API_KEY`(实为 NCBI E-utilities key);密钥从环境变量读,不进 git。
- 逐 cell 通读 2030 行 notebook 后定稿 `PLAN.md` 迁移路线。
