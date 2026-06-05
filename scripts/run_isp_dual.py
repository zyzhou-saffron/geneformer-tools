import logging
import os
import pickle

import pandas as pd

import datasets
datasets.logging.disable_progress_bar()

from tqdm.auto import tqdm

from geneformer.emb_extractor import EmbExtractor
from geneformer import InSilicoPerturber
from geneformer_tools import DualPerturber

logger = logging.getLogger(__name__)

# 运行命令
# nohup python ~/ipynb/isp_dual.py >> ~/isp.output_dual_N2CD.log 2>&1 &

input_data = "~/geneformer_testdata/Total_Cells_of_the_human_intestinal_tract_mapped_across_space_and_time/30m_tokenized.dataset"
output_directory = "~/geneformer_testdata/isp/output_embs"
token_dictionary = "~/Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl"

# 扰动的细胞类型
#filter_data_dict={"cell_type":["colonocyte", "colon epithelial cell", "intestine goblet cell", "epithelial cell"]}
filter_data_dict={}
# Crohn disease,normal
# 细胞状态的分类
cell_states_to_model={"state_key": "disease", 
                      "start_state": "normal", 
                      "goal_state": "Crohn disease", 
                      "alt_states": []}
# normal/Crohn disease

# Embedding的设置
embex = EmbExtractor(model_type="CellClassifier", # if using previously fine-tuned cell classifier model
                     num_classes=2,
                     filter_data=filter_data_dict,
                     max_ncells=1000,
                     emb_layer=0,
                     summary_stat="exact_mean",
                     forward_batch_size=256,
                     nproc=20,
                     token_dictionary_file=token_dictionary,
                     emb_mode="cell")

# 
state_embs_dict = embex.get_state_embs(cell_states_to_model,
                                       "~/finetuning30m/250516115628/250516_geneformer_cellClassifier_cm_classifier_test/ksplit1/_objective_2025-05-16_11-57-46/_objective_878622fc_12_learning_rate=0.0005,lr_scheduler_type=cosine,num_train_epochs=1,per_device_train_batch_size=12,seed=86.044_2025-05-16_13-27-01/checkpoint_000000/checkpoint-3761/",
                                       input_data,
                                       output_directory,
                                       "control_to_CD")

import pandas as pd
import numpy as np
genes_list = pd.read_csv("~/ibd_322_irigis_esgn.tsv",header=None, sep="\t").iloc[:,0].to_list()

import itertools

# 读取token_dictionary
with open(token_dictionary, "rb") as f:
    gene_token_dict = pickle.load(f)
token_gene_dict = {v: k for k, v in gene_token_dict.items()}
pad_token_id = gene_token_dict.get("<pad>")
cls_token_id = gene_token_dict.get("<cls>")
eos_token_id = gene_token_dict.get("<eos>")

# 筛选出不在字典中的基因
missing_genes = [
    gene
    for gene in genes_list
    if gene not in gene_token_dict.keys()
]
if len(missing_genes) == len(genes_list):
    raise ValueError(
        "All genes are missing from the token dictionary. Please check the gene list or token dictionary."
    )
elif len(missing_genes) > 0:
    print(f"Warning: The following genes are missing from the token dictionary: {missing_genes}")

# 更新基因列表，确保只包含在token字典中的基因
genes_list_filtered = [gene for gene in genes_list if gene in gene_token_dict.keys()]
print(f"Gene numbers to pertube: {len(genes_list_filtered)}/{len(genes_list)}")

import itertools
from tqdm import tqdm
import datasets
# 读取数据集, 使用的是已经tokenized的dataset
filtered_dataset = datasets.load_from_disk(input_data)
input_sets = [set(input_list) for input_list in tqdm(filtered_dataset['input_ids'], desc="exporting input_ids")]
# 生成基因组合
gene_combinations = list(itertools.permutations(genes_list_filtered, 2)) # 生成所有长度为 2 的有序排列

cache = {}

def get_valid_combinations_from_cache(gene_combinations):
    if "valid_combinations" not in cache:
        cache["valid_combinations"] = []
        for indexgene, gene_pair in enumerate(tqdm(gene_combinations, desc="Processing gene combinations")):
            gene_pair = list(gene_pair)
            gene1, gene2 = gene_pair
            if gene1 in gene_token_dict and gene2 in gene_token_dict:
                token1 = gene_token_dict[gene1]
                token2 = gene_token_dict[gene2]
                for input_set in input_sets:
                    if token1 in input_set and token2 in input_set:
                        cache["valid_combinations"].append(gene_pair)
                        break
    return cache["valid_combinations"]

valid_combinations = get_valid_combinations_from_cache(gene_combinations)
print(f"Total valid gene combinations found: {len(valid_combinations)}")

# 输出有效的基因组合
del input_sets
del gene_combinations


from datetime import datetime
current_date = datetime.now().strftime('%Y_%m_%d')

from tqdm import tqdm
from multiprocessing import freeze_support

perturb_type = "dual"

intermediate_files_fold = f"~/geneformer_testdata/isp/outputs_intermediate_files/{current_date}_{perturb_type}_[{cell_states_to_model.get('start_state')}]-[{cell_states_to_model.get('goal_state')}]"
if not os.path.exists(intermediate_files_fold):
    os.makedirs(intermediate_files_fold)

for gene_pair in tqdm(valid_combinations, desc="Processing gene pairs", position=0, leave=True, mininterval=5.0):
    try:
        isp = DualPerturber(perturb_type="dual",
                                    perturb_rank_shift=None,
                                    genes_to_perturb=gene_pair,
                                    token_dictionary_file=token_dictionary,
                                    combos=0,
                                    anchor_gene=None,
                                    model_type="CellClassifier",
                                    num_classes=2,
                                    emb_mode="cell",
                                    cell_emb_style="mean_pool",
                                    filter_data=filter_data_dict,
                                    cell_states_to_model=cell_states_to_model,
                                    state_embs_dict=state_embs_dict,
                                    max_ncells=1000,
                                    emb_layer=0,
                                    forward_batch_size=1000,
                                    nproc=1)
        intermediate_files_output_prefix = f"{gene_pair[0]}_{gene_pair[1]}_perturb_analysis"

        isp.perturb_dual(model_directory="~/finetuning30m/250516115628/250516_geneformer_cellClassifier_cm_classifier_test/ksplit1/_objective_2025-05-16_11-57-46/_objective_878622fc_12_learning_rate=0.0005,lr_scheduler_type=cosine,num_train_epochs=1,per_device_train_batch_size=12,seed=86.044_2025-05-16_13-27-01/checkpoint_000000/checkpoint-3761/",  # example 30M fine-tuned model
                        input_data_file=input_data,  # path/to/input_data
                        output_directory=intermediate_files_fold,  # path/to/isp_output_directory
                        output_prefix=intermediate_files_output_prefix)
        
    except Exception as e:
        print(f"Error occurred for gene pair {gene_pair}: {e}")
        continue
