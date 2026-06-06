"""overexpress ISP sweep —— 瘦包装,真正的逻辑在 geneformer_tools.isp_runner.prepare_and_run_isp。

跑法(后台):
  nohup python scripts/run_isp.py >> ~/isp.overexpress_N2CD.log 2>&1 &
"""
from geneformer_tools import config
from geneformer_tools.isp_runner import prepare_and_run_isp

# —— 实验特定路径(IBD 肠道数据集);通用路径(token 字典 / 微调模型)走 config ——
INPUT_DATA = "~/Projects/geneformer/geneformer_testdata/Total_Cells_of_the_human_intestinal_tract_mapped_across_space_and_time/30m_tokenized.dataset"
GENE_TSV = "~/Documents/data/ibd/ibd_322_irigis_esgn.tsv"
OUTPUT_BASE = "~/Projects/geneformer/geneformer_testdata/isp/outputs_intermediate_files"
EMB_OUT = "~/Projects/geneformer/geneformer_testdata/isp/output_embs"
CELL_STATES = {
    "state_key": "disease",
    "start_state": "normal",
    "goal_state": "Crohn disease",
    "alt_states": [],
}

if __name__ == "__main__":
    folder = prepare_and_run_isp(
        perturb_type="overexpress",  # -> InSilicoPerturber.perturb_data,基因对无序(combinations)
        model_directory=config.FINETUNED_MODEL,
        input_data_file=INPUT_DATA,
        gene_tsv=GENE_TSV,
        token_dictionary_file=config.TOKEN_DICT_30M,
        cell_states_to_model=CELL_STATES,
        output_base=OUTPUT_BASE,
        emb_output_directory=EMB_OUT,
    )
    print("done ->", folder)
