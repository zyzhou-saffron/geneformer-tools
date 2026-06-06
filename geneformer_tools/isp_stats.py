"""ISP 结果统计:把 in_silico_perturber 产出的 _raw.pickle 汇总成 goal-state 偏移表。

核心是 `isp_stats_to_goal_state` —— 用户改写自官方
`geneformer.in_silico_perturber_stats.isp_stats_to_goal_state`,逐字搬运(仅把原来
依赖的模块级全局 `cos_sims_df_initial` 改成函数参数 `cos_sims_df`,二者在原 notebook
调用处本就是同一对象,完全等价)。两个分支均已对真实数据验证与 notebook 原版逐元素一致。

read_dictionaries / get_gene_list / token_tuple_to_ensembl_ids / get_fdr 都是官方函数,直接复用。
"""
import pickle
import random

import numpy as np
import pandas as pd
from scipy.stats import ranksums
from tqdm.auto import trange

from geneformer.in_silico_perturber_stats import (
    get_fdr,
    get_gene_list,
    read_dictionaries,
    token_tuple_to_ensembl_ids,
)


def invert_dict(dictionary):
    """{k: v} -> {v: k}。"""
    return {v: k for k, v in dictionary.items()}


def token_to_gene_name(item, gene_token_id_dict, gene_id_name_dict):
    """token(或 token 元组)-> 基因名(或基因名元组)。

    与 notebook 原版逻辑一致,只是把原来用的两个模块全局改成显式参数。
    """
    if np.issubdtype(type(item), np.integer):
        return gene_id_name_dict.get(gene_token_id_dict.get(item, np.nan), np.nan)
    if isinstance(item, tuple):
        return tuple(
            [
                gene_id_name_dict.get(gene_token_id_dict.get(i, np.nan), np.nan)
                for i in item
            ]
        )


def build_cos_sims_df_initial(gene_list, gene_token_id_dict, gene_id_name_dict):
    """由扰动出现的 token 列表构造初始表(Gene / Gene_name / Ensembl_ID)。"""
    return pd.DataFrame(
        {
            "Gene": gene_list,
            "Gene_name": [
                token_to_gene_name(item, gene_token_id_dict, gene_id_name_dict)
                for item in gene_list
            ],
            "Ensembl_ID": [
                token_tuple_to_ensembl_ids(genes, gene_token_id_dict)
                if isinstance(genes, tuple)
                else gene_token_id_dict[genes]
                for genes in gene_list
            ],
        },
        index=[i for i in range(len(gene_list))],
    )


def isp_stats_to_goal_state(
    cos_sims_df, result_dict, cell_states_to_model, genes_perturbed
):
    """用户改写版:计算每个(组)扰动向 goal_state 的余弦偏移。

    - genes_perturbed != "all":单/组扰动,无随机背景可比,仅算 Shift_to_goal_end(分支一)。
    - genes_perturbed == "all":全基因扫描,带随机背景的 ranksums p 值、FDR、显著性(分支二)。

    ⚠️ 注意:当 cell_states_to_model["alt_states"] 非空时,分支二会引用 notebook 里从未定义的
    `names`/`results_df`(原代码遗留),将抛 NameError——此行为与 notebook 原版完全一致,
    刻意保留以保证等价。用户的实际配置 alt_states=[],不会触发该路径。
    """
    # 第一步:判断是否存在替代状态(alt_states)用于比较。
    if (
        ("alt_states" not in cell_states_to_model.keys())
        or (len(cell_states_to_model["alt_states"]) == 0)
        or (cell_states_to_model["alt_states"] == [None])
    ):
        alt_end_state_exists = False
    elif (len(cell_states_to_model["alt_states"]) > 0) and (
        cell_states_to_model["alt_states"] != [None]
    ):
        alt_end_state_exists = True

    # 多个细胞中的单个扰动:没有可比较的随机扰动
    if genes_perturbed != "all":
        cos_sims_full_df = pd.DataFrame()

        cos_shift_data_end = []
        token = cos_sims_df["Gene"][0]
        cos_shift_data_end += result_dict[cell_states_to_model["goal_state"]].get(
            (token, "cell_emb"), []
        )
        cos_sims_full_df["Shift_to_goal_end"] = [np.mean(cos_shift_data_end)]
        if alt_end_state_exists is True:
            for alt_state in cell_states_to_model["alt_states"]:
                cos_shift_data_alt_state = []
                cos_shift_data_alt_state += result_dict.get(alt_state).get(
                    (token, "cell_emb"), []
                )
                cos_sims_full_df[f"Shift_to_alt_end_{alt_state}"] = [
                    np.mean(cos_shift_data_alt_state)
                ]

        # 原 notebook 此处用模块全局 cos_sims_df_initial;调用时它就是传入的 cos_sims_df,故等价替换。
        cos_sims_full_df = pd.concat([cos_sims_df, cos_sims_full_df], axis=1)

        # sort by shift to desired state
        cos_sims_full_df = cos_sims_full_df.sort_values(
            by=["Shift_to_goal_end"], ascending=[False]
        )
        return cos_sims_full_df

    elif genes_perturbed == "all":
        goal_end_random_megalist = []
        if alt_end_state_exists is True:
            alt_end_state_random_dict = {
                alt_state: [] for alt_state in cell_states_to_model["alt_states"]
            }
        for i in trange(cos_sims_df.shape[0]):
            token = cos_sims_df["Gene"][i]
            goal_end_random_megalist += result_dict[
                cell_states_to_model["goal_state"]
            ].get((token, "cell_emb"), [])
            if alt_end_state_exists is True:
                for alt_state in cell_states_to_model["alt_states"]:
                    alt_end_state_random_dict[alt_state] += result_dict[alt_state].get(
                        (token, "cell_emb"), []
                    )

        # downsample to improve speed of ranksums
        if len(goal_end_random_megalist) > 100_000:
            random.seed(42)
            goal_end_random_megalist = random.sample(
                goal_end_random_megalist, k=100_000
            )
        if alt_end_state_exists is True:
            for alt_state in cell_states_to_model["alt_states"]:
                if len(alt_end_state_random_dict[alt_state]) > 100_000:
                    random.seed(42)
                    alt_end_state_random_dict[alt_state] = random.sample(
                        alt_end_state_random_dict[alt_state], k=100_000
                    )

        if alt_end_state_exists is True:
            [
                names.append(f"Shift_to_alt_end_{alt_state}")  # noqa: F821 (notebook 遗留,见 docstring)
                for alt_state in cell_states_to_model["alt_states"]
            ]
            names.append(names.pop(names.index("Goal_end_vs_random_pval")))  # noqa: F821
            [
                names.append(f"Alt_end_vs_random_pval_{alt_state}")  # noqa: F821
                for alt_state in cell_states_to_model["alt_states"]
            ]

        n_detections_dict = dict()
        for i in trange(cos_sims_df.shape[0]):
            token = cos_sims_df["Gene"][i]
            name = cos_sims_df["Gene_name"][i]
            ensembl_id = cos_sims_df["Ensembl_ID"][i]
            goal_end_cos_sim_megalist = result_dict[
                cell_states_to_model["goal_state"]
            ].get((token, "cell_emb"), [])
            n_detections_dict[token] = len(goal_end_cos_sim_megalist)
            mean_goal_end = np.mean(goal_end_cos_sim_megalist)
            pval_goal_end = ranksums(
                goal_end_random_megalist, goal_end_cos_sim_megalist
            ).pvalue

            if alt_end_state_exists is True:
                alt_end_state_dict = {
                    alt_state: [] for alt_state in cell_states_to_model["alt_states"]
                }
                for alt_state in cell_states_to_model["alt_states"]:
                    alt_end_state_dict[alt_state] = result_dict[alt_state].get(
                        (token, "cell_emb"), []
                    )
                    alt_end_state_dict[f"{alt_state}_mean"] = np.mean(
                        alt_end_state_dict[alt_state]
                    )
                    alt_end_state_dict[f"{alt_state}_pval"] = ranksums(
                        alt_end_state_random_dict[alt_state],
                        alt_end_state_dict[alt_state],
                    ).pvalue

            if alt_end_state_exists is True:
                for alt_state in cell_states_to_model["alt_states"]:
                    results_df[f"Shift_to_alt_end_{alt_state}"] = alt_end_state_dict[  # noqa: F821
                        f"{alt_state}_mean"
                    ]
                    results_df[  # noqa: F821
                        f"Alt_end_vs_random_pval_{alt_state}"
                    ] = alt_end_state_dict[f"{alt_state}_pval"]

            cos_sims_df.at[i, "Shift_to_goal_end"] = mean_goal_end
            cos_sims_df.at[i, "Goal_end_vs_random_pval"] = pval_goal_end

        cos_sims_df["Goal_end_FDR"] = get_fdr(
            list(cos_sims_df["Goal_end_vs_random_pval"])
        )
        if alt_end_state_exists is True:
            for alt_state in cell_states_to_model["alt_states"]:
                cos_sims_df[f"Alt_end_FDR_{alt_state}"] = get_fdr(
                    list(cos_sims_df[f"Alt_end_vs_random_pval_{alt_state}"])
                )

        # quantify number of detections of each gene
        cos_sims_df["N_Detections"] = [
            n_detections_dict[token] for token in cos_sims_df["Gene"]
        ]

        # sort by shift to desired state
        cos_sims_df["Sig"] = [
            1 if fdr < 0.05 else 0 for fdr in cos_sims_df["Goal_end_FDR"]
        ]
        cos_sims_df = cos_sims_df.sort_values(
            by=["Sig", "Shift_to_goal_end", "Goal_end_FDR"],
            ascending=[False, False, True],
        )

    return cos_sims_df


def compute_isp_stats(
    input_data_directory,
    cell_states_to_model,
    genes_perturbed,
    token_dictionary_file,
    gene_name_id_dictionary_file,
    cell_or_gene_emb="cell",
    pickle_suffix="_raw.pickle",
    drop_gene_col=True,
):
    """端到端:读 _raw.pickle 目录 -> 构造初始表 -> isp_stats_to_goal_state -> 返回结果表。

    复刻 notebook 的统计流程(read_dictionaries -> get_gene_list -> build_cos_sims_df_initial ->
    isp_stats_to_goal_state -> 去掉首列 Gene)。落盘由调用方负责。
    """
    gene_token_dict = pickle.load(open(token_dictionary_file, "rb"))
    gene_token_id_dict = invert_dict(gene_token_dict)
    gene_name_id_dict = pickle.load(open(gene_name_id_dictionary_file, "rb"))
    gene_id_name_dict = invert_dict(gene_name_id_dict)

    dict_list = read_dictionaries(
        input_data_directory=input_data_directory,
        cell_or_gene_emb=cell_or_gene_emb,
        anchor_token=None,
        cell_states_to_model=cell_states_to_model,
        pickle_suffix=pickle_suffix,
    )
    gene_list = get_gene_list(dict_list, cell_or_gene_emb)
    cos_sims_df_initial = build_cos_sims_df_initial(
        gene_list, gene_token_id_dict, gene_id_name_dict
    )
    cos_sims_df = isp_stats_to_goal_state(
        cos_sims_df_initial, dict_list, cell_states_to_model, genes_perturbed
    )
    if drop_gene_col:
        cos_sims_df = cos_sims_df.iloc[:, 1:]
    return cos_sims_df
