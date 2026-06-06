"""集中配置:路径与密钥。改这里,别再到处硬编码。
密钥从环境变量读取,绝不写进代码(更不进 git)。
路径默认指向 ~/Projects,可用环境变量覆盖。"""
import os

# ---------------- NCBI E-utilities (Entrez) API key —— fetch_literature_info 查文献用(免费,NCBI 账号生成)
# 用前先 export(并轮换掉历史里那个已泄露的旧 key):
#   export NCBI_API_KEY="<your key>"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")

# ---------------------------------------------------------------- Geneformer 仓库 / 字典
GENEFORMER_REPO = os.path.expanduser(
    os.environ.get("GENEFORMER_REPO", "~/Projects/geneformer/Geneformer")
)
GC30M_DIR = os.path.join(GENEFORMER_REPO, "geneformer", "gene_dictionaries_30m")
TOKEN_DICT_30M = os.path.join(GC30M_DIR, "token_dictionary_gc30M.pkl")
GENE_MEDIAN_30M = os.path.join(GC30M_DIR, "gene_median_dictionary_gc30M.pkl")
GENE_MAPPING_30M = os.path.join(GC30M_DIR, "ensembl_mapping_dict_gc30M.pkl")
GENE_NAME_ID_30M = os.path.join(GC30M_DIR, "gene_name_id_dict_gc30M.pkl")

# 微调好的 CellClassifier checkpoint(ISP 用)。默认指向已确认存在的 normal/Crohn 2 类
# CellClassifier(gf-6L-30M 基座,200-trial 超参搜索里的 878622fc_12 试验)。可用 env 覆盖:
#   export GF_FINETUNED_MODEL="/path/to/.../checkpoint-XXXX"
_DEFAULT_FINETUNED_MODEL = (
    "~/Projects/geneformer/finetuning30m/250516115628/"
    "250516_geneformer_cellClassifier_cm_classifier_test/ksplit1/_objective_2025-05-16_11-57-46/"
    "_objective_878622fc_12_learning_rate=0.0005,lr_scheduler_type=cosine,num_train_epochs=1,"
    "per_device_train_batch_size=12,seed=86.044_2025-05-16_13-27-01/checkpoint_000000/checkpoint-3761/"
)
FINETUNED_MODEL = os.path.expanduser(
    os.environ.get("GF_FINETUNED_MODEL", _DEFAULT_FINETUNED_MODEL)
)

# ---------------------------------------------------------------- 输出 / 日志
LOG_DIR = os.path.expanduser(os.environ.get("GF_LOG_DIR", "~/Projects/geneformer-tools/logs"))
OUTPUT_DIR = os.path.expanduser(os.environ.get("GF_OUTPUT_DIR", "~/Projects/geneformer-tools/outputs"))
