"""DualPerturber —— 在官方 InSilicoPerturber 基础上做"删 gene1 + 过表达 gene2"的双扰动。
原属 site-packages/geneformer/mynewfun.py,迁出至本项目以纳入版本控制(2026-06 迁移)。
"""
import logging
import os
from collections import defaultdict
from typing import List

import torch
from datasets import Dataset
from tqdm.rich import trange

from geneformer import InSilicoPerturber
import geneformer.perturber_utils as pu
from geneformer.emb_extractor import get_embs

logger = logging.getLogger(__name__)


class DualPerturber(InSilicoPerturber):
    """
        初始化计算机模拟扰动器。
    **参数**
    perturb_type{"delete", "overexpress", "inhibit", "activate"}
        | 扰动类型。
        | "delete":从秩值编码中删除基因
        | "overexpress":将基因移至秩值编码的开头
        | *(待办)* "inhibit":将基因移至秩值编码的下四分位数
        | *(待办)* "activate":将基因移至秩值编码的上四分位数
    *(待办)* perturb_rank_shift: None, {1,2,3}
        | 基因秩移动的四分位数数量。
        | 例如，如果 perturb_type="activate" 且 perturb_rank_shift=1
        |     第4四分位数的基因将移至第3四分位数的中间。
        |     第3四分位数的基因将移至第2四分位数的中间。
        |     第2四分位数的基因将移至第1四分位数的中间。
        |     第1四分位数的基因将移至秩值编码的开头。
        | 例如，如果 perturb_type="inhibit" 且 perturb_rank_shift=2
        |     第1四分位数的基因将移至第3四分位数的中间。
        |     第2四分位数的基因将移至第4四分位数的中间。
        |     第3或第4四分位数的基因将移至秩值编码的底部。
    genes_to_perturb: "all", 列表
        | 默认是对数据集中每个细胞中检测到的每个基因进行扰动。
        | 否则可以提供要扰动的基因的ENSEMBL ID列表。
        | 如果提供了基因列表，那么扰动器将只测试一起扰动它们
        | (而不是测试所提供基因的每种可能组合)。
    combos:{0,1}
        | 是单独(0)还是成对(1)扰动基因。
    anchor_gene:None, 字符串
        | 在组合扰动中用作锚点的基因的ENSEMBL ID。
        | 例如，如果 combos=1 且 anchor_gene="ENSG00000148400":
        |     锚点基因将与其他每个基因组合进行扰动。
    model_type:{"Pretrained", "GeneClassifier", "CellClassifier", "MTLCellClassifier", "MTLCellClassifier-Quantized"}
        | 模型是预训练的Geneformer还是微调的基因、细胞或多任务细胞分类器(+/- 8位量化)。
    num_classes:整数
        | 如果模型是基因或细胞分类器，指定其训练分类的类别数。
        | 对于预训练的Geneformer模型,类别数为0,因为它不是分类器。
    emb_mode:{"cls", "cell", "cls_and_gene","cell_and_gene"}
        | 是否输出扰动对CLS标记、细胞和/或基因嵌入的影响。
        | 基因嵌入移位仅可与原始细胞进行比较，而不是与目标状态进行比较。
    cell_emb_style:"mean_pool"
        | 如果不使用CLS标记,汇总细胞嵌入的方法。
        | 目前唯一的选项是对给定细胞的基因嵌入进行平均池化。
    filter_data:None, 字典
        | 默认是使用所有输入数据进行计算机模拟扰动研究。
        | 否则，字典指定 .dataset 列名和要过滤的值列表。
    cell_states_to_model:None, 字典
        | 如果测试实现目标状态变化的扰动，则为要建模的细胞状态。
        | 包含四个项的字典，键为:state_key、start_state、goal_state 和 alt_states
        | state_key:指定 .dataset 中定义起始/目标状态的列名的键
        | start_state:state_key列中指定起始状态的值
        | goal_state:state_key列中指定目标结束状态的值
        | alt_states:state_key列中指定替代结束状态的值列表
        | 例如:{"state_key": "disease",
        |               "start_state": "dcm",
        |               "goal_state": "nf",
        |               "alt_states": ["hcm", "other1", "other2"]}
    state_embs_dict:None, 字典
        | 要建模从/向其转移的每个细胞状态的嵌入位置(例如均值或中位数)。
        | 字典，键指定要建模的每个可能的细胞状态。
        | 值是作为torch.tensor的目标嵌入位置。
        | 例如:{"nf": emb_nf,
        |               "hcm": emb_hcm,
        |               "dcm": emb_dcm,
        |               "other1": emb_other1,
        |               "other2": emb_other2}
    max_ncells:None, 整数
        | 要测试的最大细胞数。
        | 如果为None,将测试所有细胞。
    cell_inds_to_perturb:"all", 列表
        | 默认是对数据集中的每个细胞进行扰动。
        | 否则,可以提供一个包含要扰动的细胞索引的字典,键为start_ind和end_ind。
        | start_ind:要扰动的第一个索引。
        | end_ind:要扰动的最后一个索引(不包含)。
        | 索引将在filter_data标准和排序之后选择。
        | 对于将超大的数据集拆分到不同的GPU上很有用。
    emb_layer:{-1, 0}
        | 用于量化的嵌入层。
        | 0:最后一层(对于与模型训练目标密切相关的问题推荐)
        | -1:倒数第二层(对于需要更通用表示的问题推荐)
    forward_batch_size:整数
        | 前向传递的批次大小。
    nproc:整数
        | 使用的CPU进程数。
    token_dictionary_file:路径
        | 包含标记字典(Ensembl ID:标记)的pickle文件的路径。
    clear_mem_ncells:整数
        | 每n个细胞清除一次内存。

    
    """


    valid_option_dict = InSilicoPerturber.valid_option_dict.copy()  # 复制父类的字典
    valid_option_dict["perturb_type"].add("dual") # 向 perturb_type 添加 "dual"

    def __init__(
        self,
        perturb_type="dual",  # 扰动的类型，默认值为 "dual"
        perturb_rank_shift=None,  # 用于指定扰动的排名偏移量，默认为 None。特定扰动类型下有用("activate"/"inhibit")
        genes_to_perturb="all",  # 指定要扰动的基因，默认为 "all"，表示扰动所有基因。也可以提供一个基因列表
        combos=0,  # 如果为 0，表示基因单独扰动；如果为 1，表示基因成对扰动
        anchor_gene=None,  # 如果指定一个基因作为 anchor_gene，则该基因将作为组合扰动中的锚基因
        model_type="CellClassifier",  # 使用的模型类型，默认为 "Pretrained"，也可以是 "GeneClassifier"、"CellClassifier" 等
        num_classes=0,  # 如果是基因或细胞分类器，指定分类的类别数
        emb_mode="cell",  # 指定嵌入模式，默认为 "cls"，表示使用 CLS token 来表示扰动对基因或细胞嵌入的影响
        cell_emb_style="mean_pool",  # 如果不使用 CLS token，则指定细胞嵌入的汇总方法，默认为 mean_pool，即对所有基因的嵌入进行平均池化
        filter_data=None,  # 用于过滤数据的字典，默认为 None，表示不进行过滤
        cell_states_to_model=None,  # 如果进行扰动以改变细胞状态，指定细胞状态的字典，默认为 None
        state_embs_dict=None,  # 提供目标细胞状态的嵌入位置，用于建模细胞状态的变化
        max_ncells=None,  # 指定最大细胞数，默认不限制
        cell_inds_to_perturb="all",  # 如果不是所有细胞都要扰动，提供一个字典，指定要扰动的细胞的起始和结束索引
        emb_layer=0,  # 表示用于量化的嵌入层，-1 表示倒数第二层，0 表示最后一层
        forward_batch_size=100,  # 表示前向传递时的批量大小
        nproc=16,  # 使用的 CPU 进程数，默认为 4
        token_dictionary_file=None,  #  包含标记字典(Ensembl ID:标记)的pickle文件的路径。
        clear_mem_ncells=1000,  # 每 1000 个细胞清理一次内存
    ):
        # 调用父类的构造函数进行初始化
        super().__init__(
            perturb_type=perturb_type,
            perturb_rank_shift=perturb_rank_shift,
            genes_to_perturb=genes_to_perturb,
            combos=combos,
            anchor_gene=anchor_gene,
            model_type=model_type,
            num_classes=num_classes,
            emb_mode=emb_mode,
            cell_emb_style=cell_emb_style,
            filter_data=filter_data,
            cell_states_to_model=cell_states_to_model,
            state_embs_dict=state_embs_dict,
            max_ncells=max_ncells,
            cell_inds_to_perturb=cell_inds_to_perturb,
            emb_layer=emb_layer,
            forward_batch_size=forward_batch_size,
            nproc=nproc,
            token_dictionary_file=token_dictionary_file,
            clear_mem_ncells=clear_mem_ncells,
        )

        # 新逻辑 perturb_type 是 "dual"
        if self.perturb_type == "dual":
            pass
            #logger.warning("Dual perturbation selected. Adjusting parameters accordingly.")


    def perturb_dual(
            self, model_directory, input_data_file, output_directory, output_prefix
    ):
        ### format output path ###
        output_path_prefix = os.path.join(
            output_directory, f"in_silico_{self.perturb_type}_{output_prefix}"
        )

        ### load model and define parameters ###
        model = pu.load_model(
            self.model_type, self.num_classes, model_directory, mode="eval"
        )
        self.max_len = pu.get_model_input_size(model)
        layer_to_quant = pu.quant_layers(model) + self.emb_layer
        #logger.warning("model loaded.")

        ### filter input data ###
        # general filtering of input data based on filter_data argument
        filtered_input_data = pu.load_and_filter(
            self.filter_data, self.nproc, input_data_file
        )
        
        filtered_input_data = self.apply_additional_filters(filtered_input_data)
        #logger.warning("primpry and additional filter processed.")
        
        if (self.perturb_group is True) and (self.perturb_type == "dual"):
            self.isp_perturb_dual_set(
                model, filtered_input_data, layer_to_quant, output_path_prefix
            )

    def isp_perturb_dual_set(
        self,
        model,
        filtered_input_data: Dataset,
        layer_to_quant: int,
        output_path_prefix: str,
    ):
        def make_group_perturbation_batch(example):
            example["tokens_to_perturb"] = self.tokens_to_perturb
            example_input_ids = example['input_ids']
            # 先处理delete
            example["tokens_to_delete"] = self.tokens_to_perturb[0]

            indices_to_delete = [
                            example_input_ids.index(token) if token in example_input_ids else None
                            for token in [example["tokens_to_delete"]]
                        ]
            indices_to_delete = [
                            item for item in indices_to_delete if item is not None
                        ]
            if len(indices_to_delete) > 0:
                example["indices_to_delete"] = indices_to_delete
            else:
                # -100 indicates tokens to overexpress are not present in rank value encoding
                example["indices_to_delete"] = [-100]
            
            example = self.delete_indices_fordual(example,"indices_to_delete")
            example["length"] = len(example["input_ids"])

            # 处理overexpress
            example_input_ids = example['input_ids']
            example["tokens_to_overexpress"] = self.tokens_to_perturb[1]
            indices_to_overexpress = [
                            example_input_ids.index(token) if token in example_input_ids else None
                            for token in [example["tokens_to_overexpress"]]
                        ]
            indices_to_overexpress = [
                            item for item in indices_to_overexpress if item is not None
                        ]
            if len(indices_to_overexpress) > 0:
                example["indices_to_overexpress"] = indices_to_overexpress
            else:
                # -100 indicates tokens to overexpress are not present in rank value encoding
                example["indices_to_overexpress"] = [-100]
            
            example = self.overexpress_tokens_fordual(
                example, self.max_len, special_token=self.special_token
            )

            indices_to_overexpress = [value + 1 for value in indices_to_overexpress]
            indices_to_perturb = indices_to_delete + indices_to_overexpress
            example["perturb_index"] = indices_to_perturb
            
            example["n_overflow"] = pu.calc_n_overflow(
                self.max_len,
                example["length"],
                [self.tokens_to_perturb],
                [indices_to_perturb],
            )
            
            return example

        total_batch_length = len(filtered_input_data)
        if self.cell_states_to_model is None:
            cos_sims_dict = defaultdict(list)
        else:
            cos_sims_dict = {
                state: defaultdict(list)
                for state in pu.get_possible_states(self.cell_states_to_model)
            }

        perturbed_data = filtered_input_data.map(
            make_group_perturbation_batch, num_proc=self.nproc
        )

        
        filtered_input_data = filtered_input_data.add_column(
            "n_overflow", perturbed_data["n_overflow"]
        )
            # remove overflow genes from original data so that embeddings are comparable
            # i.e. if original cell has genes 0:2047 and you want to overexpress new gene 2048,
            # then the perturbed cell will be 2048+0:2046 so we compare it to an original cell 0:2046.
            # (otherwise we will be modeling the effect of both deleting 2047 and adding 2048,
            # rather than only adding 2048)
        filtered_input_data = filtered_input_data.map(
            pu.truncate_by_n_overflow, num_proc=self.nproc
        )

        if self.emb_mode == "cell_and_gene":
            stored_gene_embs_dict = defaultdict(list)

        # iterate through batches
        for i in trange(0, total_batch_length, self.forward_batch_size, disable=True):
            max_range = min(i + self.forward_batch_size, total_batch_length)
            inds_select = [i for i in range(i, max_range)]

            minibatch = filtered_input_data.select(inds_select)
            perturbation_batch = perturbed_data.select(inds_select)

            if self.cell_emb_style == "mean_pool":
                full_original_emb = get_embs(
                    model,
                    minibatch,
                    "gene",
                    layer_to_quant,
                    self.pad_token_id,
                    self.forward_batch_size,
                    token_gene_dict=self.token_gene_dict,
                    summary_stat=None,
                    silent=True,
                )
                if not isinstance(full_original_emb, torch.Tensor):
                    raise TypeError(f"Expected full_original_emb to be a torch.Tensor, but got {type(full_original_emb)}")
                #print(f"full_original_emb type: {type(full_original_emb)}")

                indices_to_perturb = perturbation_batch["perturb_index"]
                # remove indices that were perturbed
                original_emb = self.remove_perturbed_indices_dual_set(
                    full_original_emb,
                    self.perturb_type,
                    indices_to_perturb,
                    self.tokens_to_perturb,
                    minibatch["length"],
                )
                full_perturbation_emb = get_embs(
                    model,
                    perturbation_batch,
                    "gene",
                    layer_to_quant,
                    self.pad_token_id,
                    self.forward_batch_size,
                    token_gene_dict=self.token_gene_dict,
                    summary_stat=None,
                    silent=True,
                )

                # remove overexpressed genes
                perturbation_emb = full_perturbation_emb[
                        :, 1 :, :
                ]

                n_perturbation_genes = perturbation_emb.size()[1]

                # if no goal states, the cosine similarties are the mean of gene cosine similarities
                if (
                    self.cell_states_to_model is None
                    or self.emb_mode == "cell_and_gene"
                ):
                    gene_cos_sims = pu.quant_cos_sims(
                        perturbation_emb,
                        original_emb,
                        self.cell_states_to_model,
                        self.state_embs_dict,
                        emb_mode="gene",
                    )

                # if there are goal states, the cosine similarities are the cell cosine similarities
                if self.cell_states_to_model is not None:
                    original_cell_emb = pu.mean_nonpadding_embs(
                        full_original_emb,
                        torch.tensor(minibatch["length"], device="cuda"),
                        dim=1,
                    )
                    perturbation_cell_emb = pu.mean_nonpadding_embs(
                        full_perturbation_emb,
                        torch.tensor(perturbation_batch["length"], device="cuda"),
                        dim=1,
                    )
                    cell_cos_sims = pu.quant_cos_sims(
                        perturbation_cell_emb,
                        original_cell_emb,
                        self.cell_states_to_model,
                        self.state_embs_dict,
                        emb_mode="cell",
                    )

                # get cosine similarities in gene embeddings
                # if getting gene embeddings, need gene names
                if self.emb_mode == "cell_and_gene":
                    gene_list = minibatch["input_ids"]
                    # need to truncate gene_list
                    gene_list = [
                        [g for g in genes if g not in self.tokens_to_perturb][
                            :n_perturbation_genes
                        ]
                        for genes in gene_list
                    ]

                    for cell_i, genes in enumerate(gene_list):
                        for gene_j, affected_gene in enumerate(genes):
                            if len(self.genes_to_perturb) > 1:
                                tokens_to_perturb = tuple(self.tokens_to_perturb)
                            else:
                                tokens_to_perturb = self.tokens_to_perturb[0]

                            # fill in the gene cosine similarities
                            try:
                                stored_gene_embs_dict[
                                    (tokens_to_perturb, affected_gene)
                                ].append(gene_cos_sims[cell_i, gene_j].item())
                            except KeyError:
                                stored_gene_embs_dict[
                                    (tokens_to_perturb, affected_gene)
                                ] = gene_cos_sims[cell_i, gene_j].item()
                else:
                    gene_list = None

            if self.cell_states_to_model is None:
                # calculate the mean of the gene cosine similarities for cell shift
                # tensor of nonpadding lengths for each cell
                if self.perturb_type == "overexpress":
                    # subtract number of genes that were overexpressed
                    # since they are removed before getting cos sims
                    n_overexpressed = len(self.tokens_to_perturb)
                    nonpadding_lens = [
                        x - n_overexpressed for x in perturbation_batch["length"]
                    ]
                else:
                    nonpadding_lens = perturbation_batch["length"]
                cos_sims_data = pu.mean_nonpadding_embs(
                    gene_cos_sims, torch.tensor(nonpadding_lens, device="cuda")
                )
                cos_sims_dict = self.update_perturbation_dictionary(
                    cos_sims_dict,
                    cos_sims_data,
                    gene_list,
                )
            else:
                cos_sims_data = cell_cos_sims
                for state in cos_sims_dict.keys():
                    cos_sims_dict[state] = self.update_perturbation_dictionary(
                        cos_sims_dict[state],
                        cos_sims_data[state],
                        gene_list,
                    )
            del minibatch
            del perturbation_batch
            del original_emb
            del perturbation_emb
            del cos_sims_data

            torch.cuda.empty_cache()

        pu.write_perturbation_dictionary(
            cos_sims_dict,
            f"{output_path_prefix}_cell_embs_dict_{self.tokens_to_perturb}",
        )

        if self.emb_mode == "cell_and_gene":
            pu.write_perturbation_dictionary(
                stored_gene_embs_dict,
                f"{output_path_prefix}_gene_embs_dict_{self.tokens_to_perturb}",
            )




    def delete_indices_fordual(self,example,columnname):
        indices = example[columnname]
        if isinstance(indices, int):
            indices = [indices]
        else:
            indices = list(indices)
        if any(isinstance(el, list) for el in indices):
            indices = pu.flatten_list(indices)
        for index in sorted(indices, reverse=True):
            del example["input_ids"][index]
            #print(f"del删除了{index}位的token")
        example["length"] = len(example["input_ids"])
        return example

    def overexpress_tokens_fordual(self,example, max_len, special_token):
        tokens_to_overexpress = example["tokens_to_overexpress"]
        if not isinstance(tokens_to_overexpress, list):
            tokens_to_overexpress = [tokens_to_overexpress]
        
        # -100 indicates tokens to overexpress are not present in rank value encoding
        if tokens_to_overexpress != [-100]:
            example = self.delete_indices_fordual(example,"indices_to_overexpress")
            #print(f"ox删除了{tokens_to_overexpress}")
        if special_token:
            [
                example["input_ids"].insert(1, token)
                for token in tokens_to_overexpress[::-1]
            ]
        else:
            [
                example["input_ids"].insert(0, token)
                for token in tokens_to_overexpress[::-1]
            ]
            #print(f"ox添加了{tokens_to_overexpress}到0号位置")
        example["length"] = len(example["input_ids"])
        return example
    
    def remove_perturbed_indices_dual_set(
        self,
        emb,
        perturb_type,
        indices_to_perturb,
        tokens_to_perturb,
        original_lengths,
        input_ids=None,
    ):
        indices_to_perturb_orig = indices_to_perturb
        # 确保pu是perturber_utils模块
        emb = pu.remove_indices_from_emb_batch(emb, indices_to_perturb_orig, gene_dim=1)
        return emb
    
    def update_perturbation_dictionary(
        self,
        cos_sims_dict: defaultdict,
        cos_sims_data: torch.Tensor,
        gene_list=None,
    ):
        if gene_list is not None and cos_sims_data.shape[0] != len(gene_list):
            logger.error(
                f"len(cos_sims_data.shape[0]) != len(gene_list). \n \
                            {cos_sims_data.shape[0]=}.\n \
                            {len(gene_list)=}."
            )
            raise

        if self.perturb_group is True:
            if len(self.tokens_to_perturb) > 1:
                perturbed_genes = tuple(self.tokens_to_perturb)
            else:
                perturbed_genes = self.tokens_to_perturb[0]

            # if cell embeddings, can just append
            # shape will be (batch size, 1)
            cos_sims_data = torch.squeeze(cos_sims_data).tolist()

            # handle case of single cell left
            if not isinstance(cos_sims_data, list):
                cos_sims_data = [cos_sims_data]

            cos_sims_dict[(perturbed_genes, "cell_emb")] += cos_sims_data

        else:
            for i, cos in enumerate(cos_sims_data.tolist()):
                cos_sims_dict[(gene_list[i], "cell_emb")].append(cos)

        return cos_sims_dict
