"""ISP 扰动 sweep —— 对一批基因对逐对运行 in-silico perturbation,产出每对的 `_raw.pickle`。

合并自原 `scripts/run_isp.py`(overexpress)+ `scripts/run_isp_dual.py`(dual),两者 ~80% 重复。
复用已抽好并验证等价的 `compute_state_embs` / `valid_pairs` / `load_gene_list` / `filter_to_token_dict`。

分支:perturb_type == "dual" -> 自定义 `DualPerturber.perturb_dual`(删基因1 + 过表达基因2);
其余("overexpress"/"delete"/"inhibit"/"activate")-> 官方 `InSilicoPerturber.perturb_data`。
"""
import multiprocessing
import os
import pickle
from datetime import datetime

import datasets

from tqdm.auto import tqdm

from geneformer import InSilicoPerturber
from geneformer_tools.perturb import DualPerturber

datasets.logging.disable_progress_bar()


def sweep_output_dir(output_base, perturb_type, cell_states_to_model, date=None):
    """标准化中间文件目录名:{base}/{YYYY_MM_DD}_{perturb_type}_[start]-[goal](与原脚本一致)。"""
    date = date or datetime.now().strftime("%Y_%m_%d")
    start = cell_states_to_model.get("start_state")
    goal = cell_states_to_model.get("goal_state")
    return os.path.join(output_base, f"{date}_{perturb_type}_[{start}]-[{goal}]")


def run_isp_sweep(
    gene_pairs,
    perturb_type,
    model_directory,
    input_data_file,
    output_directory,
    state_embs_dict,
    token_dictionary_file,
    cell_states_to_model,
    filter_data=None,
    combos=0,
    num_classes=2,
    max_ncells=1000,
    emb_layer=0,
    forward_batch_size=1000,
    nproc=1,
    use_fork=True,
):
    """对每个基因对构造 perturber 并产出 `_raw.pickle`。

    每对失败只打印并跳过(与原脚本一致),不中断整轮 sweep。返回 output_directory。
    """
    if use_fork:
        # datasets>=4 下 perturb_data 的 .map 子进程会 "subprocess abruptly died";强制 fork 规避。
        multiprocessing.set_start_method("fork", force=True)

    if filter_data is None:
        filter_data = {}
    os.makedirs(output_directory, exist_ok=True)

    for gene_pair in tqdm(
        gene_pairs, desc=f"ISP {perturb_type}", position=0, leave=True, mininterval=5.0
    ):
        try:
            common = dict(
                perturb_type=perturb_type,
                perturb_rank_shift=None,
                genes_to_perturb=gene_pair,
                token_dictionary_file=token_dictionary_file,
                combos=combos,
                anchor_gene=None,
                model_type="CellClassifier",
                num_classes=num_classes,
                emb_mode="cell",
                cell_emb_style="mean_pool",
                filter_data=filter_data,
                cell_states_to_model=cell_states_to_model,
                state_embs_dict=state_embs_dict,
                max_ncells=max_ncells,
                emb_layer=emb_layer,
                forward_batch_size=forward_batch_size,
                nproc=nproc,
            )
            output_prefix = f"{gene_pair[0]}_{gene_pair[1]}_perturb_analysis"
            if perturb_type == "dual":
                isp = DualPerturber(**common)
                isp.perturb_dual(
                    model_directory=model_directory,
                    input_data_file=input_data_file,
                    output_directory=output_directory,
                    output_prefix=output_prefix,
                )
            else:
                isp = InSilicoPerturber(**common)
                isp.perturb_data(
                    model_directory=model_directory,
                    input_data_file=input_data_file,
                    output_directory=output_directory,
                    output_prefix=output_prefix,
                )
        except Exception as e:
            print(f"Error occurred for gene pair {gene_pair}: {e}")
            continue
    return output_directory


def prepare_and_run_isp(
    perturb_type,
    model_directory,
    input_data_file,
    gene_tsv,
    token_dictionary_file,
    cell_states_to_model,
    output_base,
    emb_output_directory,
    ordered=None,
    filter_data=None,
    combos=0,
    num_classes=2,
    max_ncells=1000,
    emb_layer=0,
    forward_batch_size=1000,
    nproc=1,
    emb_forward_batch_size=256,
    emb_nproc=20,
):
    """端到端跑一个 sweep:算 state_embs -> 选基因对(valid_pairs)-> run_isp_sweep。

    ordered 默认按 perturb_type 决定:dual 用有序排列(permutations),其余用无序组合(combinations)
    —— 分别对应原 run_isp_dual.py / run_isp.py。返回中间文件目录。
    """
    from geneformer_tools.state_embs import compute_state_embs
    from geneformer_tools.gene_pairs import (
        filter_to_token_dict,
        load_gene_list,
        valid_pairs,
    )

    # 1) 状态锚点(只依赖 model/data/states,与 perturb_type 无关,故一处算好)
    #    get_state_embs 不会自建输出目录,先建好,免得全新环境 FileNotFoundError。
    os.makedirs(emb_output_directory, exist_ok=True)
    state_embs_dict = compute_state_embs(
        model_directory=model_directory,
        input_data_file=input_data_file,
        cell_states_to_model=cell_states_to_model,
        output_directory=emb_output_directory,
        output_prefix="control_to_CD",
        token_dictionary_file=token_dictionary_file,
        num_classes=num_classes,
        filter_data=filter_data,
        max_ncells=max_ncells,
        emb_layer=emb_layer,
        forward_batch_size=emb_forward_batch_size,
        nproc=emb_nproc,
    )

    # 2) 候选基因对(复用已验证等价的 valid_pairs)
    with open(token_dictionary_file, "rb") as f:
        gene_token_dict = pickle.load(f)
    genes = filter_to_token_dict(load_gene_list(gene_tsv), gene_token_dict)
    cell_sets = [set(ids) for ids in datasets.load_from_disk(input_data_file)["input_ids"]]
    if ordered is None:
        ordered = perturb_type == "dual"
    gene_pairs = [
        list(p) for p in valid_pairs(genes, gene_token_dict, cell_sets, ordered=ordered)
    ]
    print(f"Total valid gene combinations found: {len(gene_pairs)}")

    # 3) sweep
    output_directory = sweep_output_dir(output_base, perturb_type, cell_states_to_model)
    return run_isp_sweep(
        gene_pairs=gene_pairs,
        perturb_type=perturb_type,
        model_directory=model_directory,
        input_data_file=input_data_file,
        output_directory=output_directory,
        state_embs_dict=state_embs_dict,
        token_dictionary_file=token_dictionary_file,
        cell_states_to_model=cell_states_to_model,
        filter_data=filter_data,
        combos=combos,
        num_classes=num_classes,
        max_ncells=max_ncells,
        emb_layer=emb_layer,
        forward_batch_size=forward_batch_size,
        nproc=nproc,
    )
