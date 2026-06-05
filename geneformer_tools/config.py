"""集中配置:路径与密钥。改这里,别再到处硬编码。
密钥从环境变量读取,绝不写进代码(更不进 git)。
路径默认指向 ~/Projects,可用环境变量覆盖。"""
import os

# ---------------------------------------------------------------- 密钥(从环境变量)
# 用前先 export(并轮换掉历史里那个已泄露的旧 key):
#   export OPENTARGETS_API_KEY="<your key>"
OPENTARGETS_API_KEY = os.environ.get("OPENTARGETS_API_KEY", "")

# ---------------------------------------------------------------- Geneformer 仓库 / 字典
GENEFORMER_REPO = os.path.expanduser(
    os.environ.get("GENEFORMER_REPO", "~/Projects/geneformer/Geneformer")
)
GC30M_DIR = os.path.join(GENEFORMER_REPO, "geneformer", "gene_dictionaries_30m")
TOKEN_DICT_30M = os.path.join(GC30M_DIR, "token_dictionary_gc30M.pkl")
GENE_MEDIAN_30M = os.path.join(GC30M_DIR, "gene_median_dictionary_gc30M.pkl")
GENE_MAPPING_30M = os.path.join(GC30M_DIR, "ensembl_mapping_dict_gc30M.pkl")
GENE_NAME_ID_30M = os.path.join(GC30M_DIR, "gene_name_id_dict_gc30M.pkl")

# 微调好的 CellClassifier checkpoint(ISP 用;改成你的实际路径)
#   export GF_FINETUNED_MODEL="/path/to/.../checkpoint-XXXX"
FINETUNED_MODEL = os.path.expanduser(os.environ.get("GF_FINETUNED_MODEL", ""))

# ---------------------------------------------------------------- 输出 / 日志
LOG_DIR = os.path.expanduser(os.environ.get("GF_LOG_DIR", "~/Projects/geneformer-tools/logs"))
OUTPUT_DIR = os.path.expanduser(os.environ.get("GF_OUTPUT_DIR", "~/Projects/geneformer-tools/outputs"))
