"""状态 embedding —— 从 ISP workflow notebook 抽出。

封装 EmbExtractor + get_state_embs:算出每个细胞状态(如 normal / Crohn)的"锚点"平均 embedding,
供 in-silico perturbation 衡量"扰动把细胞推向哪个状态"。默认参数与 notebook 一致。
"""
import logging

from geneformer.emb_extractor import EmbExtractor

logger = logging.getLogger(__name__)


def compute_state_embs(model_directory, input_data_file, cell_states_to_model,
                       output_directory, output_prefix, token_dictionary_file,
                       model_type="CellClassifier", num_classes=2, filter_data=None,
                       max_ncells=1000, emb_layer=0, summary_stat="exact_mean",
                       forward_batch_size=256, nproc=20, emb_mode="cell"):
    """返回 state_embs_dict = {state: 平均 embedding tensor}。

    cell_states_to_model: {"state_key","start_state","goal_state","alt_states"}
    model_type/num_classes: 微调分类器用 ("CellClassifier", 2);预训练用 ("Pretrained", 0)。
    其余默认值与 notebook 的 EmbExtractor 配置一致(emb_layer=0, summary_stat="exact_mean", emb_mode="cell")。
    """
    embex = EmbExtractor(
        model_type=model_type,
        num_classes=num_classes,
        filter_data=filter_data if filter_data is not None else {},
        max_ncells=max_ncells,
        emb_layer=emb_layer,
        summary_stat=summary_stat,
        forward_batch_size=forward_batch_size,
        nproc=nproc,
        token_dictionary_file=token_dictionary_file,
        emb_mode=emb_mode,
    )
    return embex.get_state_embs(
        cell_states_to_model=cell_states_to_model,
        model_directory=model_directory,
        input_data_file=input_data_file,
        output_directory=output_directory,
        output_prefix=output_prefix,
    )
