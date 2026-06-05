"""逻辑等价验证:notebook 原内联实现 vs 新 geneformer_tools.gene_pairs.valid_pairs,
同一份精确输入(intestinal 30m_tokenized.dataset + ibd_322_irigis_esgn.tsv),断言输出完全一致。
为控时长,抽样细胞(等价性在任意子集上都成立)。"""
import os, pickle, itertools, sys
import datasets
datasets.logging.disable_progress_bar()
from geneformer_tools import gene_pairs as gp

GENE_TSV = os.path.expanduser("~/Documents/data/ibd/ibd_322_irigis_esgn.tsv")
TOKEN_DICT = os.path.expanduser("~/Projects/geneformer/Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")
DATA = os.path.expanduser("~/Projects/geneformer/geneformer_testdata/Total_Cells_of_the_human_intestinal_tract_mapped_across_space_and_time/30m_tokenized.dataset")
N_CELLS = 3000   # 抽样细胞数(old/new 用同一抽样,等价性照样成立)

# --- 输入 ---
gene_token_dict = pickle.load(open(TOKEN_DICT, "rb"))
genes_list = gp.load_gene_list(GENE_TSV)
genes_filtered = gp.filter_to_token_dict(genes_list, gene_token_dict)
print(f"genes: {len(genes_list)} -> in-dict {len(genes_filtered)}", flush=True)

ds = datasets.load_from_disk(DATA)
n = min(N_CELLS, len(ds))
input_sets = [set(x) for x in ds.select(range(n))["input_ids"]]
print(f"cells total {len(ds)}, sampled {n}", flush=True)

# --- OLD:逐字复刻 notebook 的内联实现(combinations 版,802-825)---
def old_inline(genes_list_filtered, gene_token_dict, input_sets):
    gene_combinations = list(itertools.combinations(genes_list_filtered, 2))
    valid_combinations = []
    for gene_pair in gene_combinations:
        gene_pair = list(gene_pair)
        gene1, gene2 = gene_pair
        if gene1 in gene_token_dict and gene2 in gene_token_dict:
            token1 = gene_token_dict[gene1]; token2 = gene_token_dict[gene2]
            for input_set in input_sets:
                if token1 in input_set and token2 in input_set:
                    valid_combinations.append(gene_pair)
                    break
    return valid_combinations

valid_old = old_inline(genes_filtered, gene_token_dict, input_sets)

# --- NEW ---
valid_new = gp.valid_pairs(genes_filtered, gene_token_dict, input_sets, ordered=False)

# --- 比较(old 是 list[list],new 是 list[tuple];都归一成 tuple)---
old_norm = [tuple(p) for p in valid_old]
print(f"old: {len(old_norm)} pairs | new: {len(valid_new)} pairs", flush=True)
if old_norm == valid_new:
    print("RESULT: ✅ 完全一致(顺序+内容)—— 重构逻辑等价,验证通过", flush=True)
    sys.exit(0)
else:
    so, sn = set(old_norm), set(valid_new)
    print("RESULT: ❌ 不一致", flush=True)
    print("  仅 old 有:", list(so - sn)[:5], flush=True)
    print("  仅 new 有:", list(sn - so)[:5], flush=True)
    print("  顺序是否相同(集合相等时):", so == sn, flush=True)
    sys.exit(1)
