# geneformer-tools

基于 [Geneformer](https://huggingface.co/ctheodoris/Geneformer) 的**在硅扰动(in-silico perturbation, ISP)**分析工具包。给定一个微调好的细胞状态分类器,用它来研究**基因扰动如何把细胞从一个状态推向另一个状态**(例如 normal → Crohn disease)。

围绕一次完整的 ISP 实验,本包提供:在细胞中共现的候选基因(对)选择 → 批量删除/过表达/双重扰动 → 量化每个扰动向目标状态的偏移 → 结果的文献层面解读。

## 功能

| 模块 | 作用 |
|---|---|
| `data_prep` | **①准备数据**:AnnData 预处理(基因名↔Ensembl 映射、补 geneformer 字段)+ `TranscriptomeTokenizer` 封装 |
| `gene_pairs` | **②选对**:候选基因 / 基因对选择——同细胞共现(有序/无序)、跨细胞类型配对 |
| `state_embs` | **③状态锚点**:计算细胞状态向量(start / goal state embeddings) |
| `perturb` | `DualPerturber`——双重扰动(删基因 1 + 过表达基因 2) |
| `isp_runner` | **④扰动**:批量 sweep,按 `perturb_type` 分发(dual→`DualPerturber`,其余→官方 `InSilicoPerturber`)+ 端到端封装(state_embs → 选对 → 扰动) |
| `isp_stats` | **⑤统计**:把扰动产出的 cosine-shift 汇总成 goal-state 偏移结果表(单/组扰动均值;全基因带随机背景的 FDR 显著性) |
| `analysis` | **⑥解读**:`ResultAnalysis`——Open Targets + EuropePMC + NCBI E-utilities 文献分析 |

## 安装

```bash
pip install -e .
```

需先装好 `geneformer` 及其依赖(`transformers` / `datasets` / `scanpy` 等)。

## 快速开始

```python
from geneformer_tools.isp_runner import prepare_and_run_isp
from geneformer_tools.isp_stats import compute_isp_stats

cell_states = {
    "state_key": "disease", "start_state": "normal",
    "goal_state": "Crohn disease", "alt_states": [],
}

# 0) 准备数据:原始 AnnData → 补 geneformer 字段 → tokenize 成 .dataset
#    (若已有 tokenized .dataset 可跳过)
import scanpy as sc
from geneformer_tools.data_prep import prepare_adata_for_geneformer, tokenize_h5ad

adata = prepare_adata_for_geneformer(sc.read_h5ad("/path/to/raw.h5ad"))  # 加 ensembl_id/n_counts/joinid
adata.write("/path/to/h5ad_dir/prepared.h5ad")
tokenize_h5ad("/path/to/h5ad_dir", "/path/to/out", "my_30m_tokenized",
              custom_attr_name_dict={"cell_type": "cell_type"})           # → /path/to/out/my_30m_tokenized.dataset

# 1) 跑一轮扰动 sweep —— 对每个候选基因对执行扰动,产出 _raw.pickle
out_dir = prepare_and_run_isp(
    perturb_type="dual",                 # "dual" / "overexpress" / "delete" / ...
    model_directory="/path/to/finetuned/checkpoint",
    input_data_file="/path/to/tokenized.dataset",
    gene_tsv="/path/to/genes.tsv",
    token_dictionary_file="/path/to/token_dictionary_gc30M.pkl",
    cell_states_to_model=cell_states,
    output_base="/path/to/intermediate_files",
    emb_output_directory="/path/to/embs",
)

# 2) 汇总成 goal-state 偏移结果表
df = compute_isp_stats(
    input_data_directory=out_dir,
    cell_states_to_model=cell_states,
    genes_perturbed="all",
    token_dictionary_file="/path/to/token_dictionary_gc30M.pkl",
    gene_name_id_dictionary_file="/path/to/gene_name_id_dict_gc30M.pkl",
)
```

命令行入口见 `scripts/run_isp.py`(overexpress sweep)和 `scripts/run_isp_dual.py`(dual sweep);
两者都是 `prepare_and_run_isp` 的薄包装,数据路径走 `ISP_*` 环境变量。

## 配置

路径与密钥集中在 `geneformer_tools/config.py`,均可用环境变量覆盖:

```bash
export NCBI_API_KEY="<NCBI E-utilities key>"      # analysis 查文献用(可选)
export GENEFORMER_REPO="/path/to/Geneformer"      # gc30M 字典所在目录
export GF_FINETUNED_MODEL="/path/to/checkpoint"   # ISP 用的微调分类器
```

## 注意事项

- **多进程**:扰动用到 `datasets` 的多进程 `.map`。驱动脚本需置于 `if __name__ == "__main__":` 下;
  `run_isp_sweep` 内部已 `multiprocessing.set_start_method("fork", force=True)`,以规避
  `datasets>=4` 下子进程 spawn 重执行 / 崩溃。
- **stats mode 匹配**:`compute_isp_stats` / 官方 `InSilicoPerturberStats` 的统计口径要与扰动方式对应
  (`goal_state_shift` 等)。
- **30M vs 95M**:用 gc-30M 系列模型时,token 字典必须配套 gc30M 版本,否则会静默用错默认字典。

## 示例

- [`examples/isp_workflow.ipynb`](examples/isp_workflow.ipynb) —— 完整 ISP 工作流 notebook(从数据准备到结果分析),已清空输出。

## 文档

- [CHANGELOG.md](CHANGELOG.md) —— 模块总览 + 逐步迁移历史。
