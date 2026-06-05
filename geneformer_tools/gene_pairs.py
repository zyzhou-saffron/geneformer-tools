"""基因对预处理 —— 从 ISP workflow notebook 抽出并合并那 3 份重复实现。

notebook 里有三段几乎一样的 valid_combinations 逻辑(普通循环 / 并行 executor / permutations 版),
这里合并成一个 valid_pairs(ordered=)。逻辑与 notebook 一致:
读基因表 → 筛掉不在 token 字典里的基因 → 两两配对 → 只留"两个基因在同一细胞中共现"的对。
"""
import itertools
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_gene_list(tsv_path: str) -> list:
    """读基因表的第 0 列为基因列表(notebook 用法:header=None, 取第一列)。"""
    return pd.read_csv(tsv_path, header=None, sep="\t").iloc[:, 0].tolist()


def filter_to_token_dict(genes: list, token_dict: dict) -> list:
    """只保留在 token 字典里的基因(Ensembl ID);全不在则报错,部分不在则警告。"""
    missing = [g for g in genes if g not in token_dict]
    if len(missing) == len(genes):
        raise ValueError("All genes are missing from the token dictionary. "
                         "Check the gene list or token dictionary.")
    if missing:
        logger.warning("%d/%d genes not in token dict (e.g. %s)",
                       len(missing), len(genes), missing[:10])
    return [g for g in genes if g in token_dict]


def cell_token_sets(tokenized_dataset) -> list:
    """把每个细胞的 input_ids 转成 set,供共现判断(一次性算好,避免重复扫描)。"""
    return [set(ids) for ids in tokenized_dataset["input_ids"]]


def valid_pairs(genes_in_dict: list, token_dict: dict, cell_sets: list,
                ordered: bool = True) -> list:
    """枚举基因对,只保留两个基因在同一细胞中共现的对。

    ordered=True  用 permutations(有序对,dual 扰动 g1≠g2 角色不同,顺序有意义)
    ordered=False 用 combinations(无序对,对称扰动 overexpress/delete)
    返回 list[tuple]。
    """
    pair_iter = (itertools.permutations(genes_in_dict, 2) if ordered
                 else itertools.combinations(genes_in_dict, 2))
    out = []
    for g1, g2 in pair_iter:
        t1, t2 = token_dict[g1], token_dict[g2]
        for s in cell_sets:
            if t1 in s and t2 in s:
                out.append((g1, g2))
                break
    return out
