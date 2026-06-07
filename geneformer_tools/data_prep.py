"""数据前处理:把原始 AnnData 准备成 Geneformer 需要的形态,并封装 tokenization。

抽自 ISP workflow notebook 前段(h5ad Preprocessing / Tokenization)里**可复用**的部分:
- build_gene_to_ensembl_map: pybiomart 基因名 -> Ensembl ID 映射(notebook cell 6)
- prepare_adata_for_geneformer: 给 AnnData 加 ensembl_id / n_counts / joinid(notebook cell 8/21)
- tokenize_h5ad: 薄封装官方 TranscriptomeTokenizer(notebook cell 18/24,人/鼠两处调用去重)

notebook 里其余前段(具体 h5ad 读写、scanpy 出图、mouse 一次性处理)属实验特定/一次性,留在 notebook。
细胞分类推理复用 geneformer skill,不在此重写。
"""
import os


def build_gene_to_ensembl_map(
    hosts=("http://www.ensembl.org", "http://grch37.ensembl.org"),
    disable_proxy=True,
):
    """基因名 -> Ensembl ID 映射,多个 biomart host 取并集(如 GRCh38 + GRCh37)。

    需联网到 ensembl.org。返回 dict{gene_name: ensembl_id}。对应 notebook cell 6。
    """
    if disable_proxy:
        for k in ("http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(k, None)
    from pybiomart import Dataset

    merged = {}
    for host in hosts:
        ds = Dataset(name="hsapiens_gene_ensembl", host=host)
        m = ds.query(attributes=["external_gene_name", "external_synonym", "ensembl_gene_id"])
        merged |= dict(zip(m["Gene name"], m["Gene stable ID"]))
    return merged


def prepare_adata_for_geneformer(
    adata, gene_to_ensembl=None, gene_name_col=None, copy=True
):
    """给 AnnData 补上 Geneformer 需要的字段:var['ensembl_id']、obs['n_counts']、obs['joinid']。

    两种来源(对应 notebook 两个 cell):
    - gene_to_ensembl + gene_name_col 给定(cell 21,数据是基因名):把基因名(取 '_' 前段)映射成
      Ensembl ID 写入 var['ensembl_id'] 并设为 index,丢掉映射不到的基因。
    - 否则(cell 8,var.index 本就是 Ensembl ID):直接把 var.index 复制到 var['ensembl_id']。

    n_counts = 每个细胞的表达计数和;joinid = 0..n_obs-1。返回处理后的 AnnData。
    """
    import numpy as np

    if copy:
        adata = adata.copy()

    if gene_to_ensembl is not None and gene_name_col is not None:
        names = adata.var[gene_name_col].astype(str).str.split("_").str[0]
        adata.var["gene_name"] = names
        adata.var["ensembl_id"] = names.map(gene_to_ensembl)
        adata.var.index = adata.var["ensembl_id"].values
        adata = adata[:, ~adata.var["ensembl_id"].isna()].copy()
        adata.var_names_make_unique()
    else:
        adata.var["ensembl_id"] = adata.var.index

    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["joinid"] = list(range(adata.n_obs))
    return adata


def tokenize_h5ad(
    h5ad_dir,
    output_directory,
    output_prefix,
    custom_attr_name_dict,
    gene_median_file=None,
    token_dictionary_file=None,
    gene_mapping_file=None,
    model_input_size=2048,
    special_token=False,
    nproc=40,
):
    """薄封装官方 TranscriptomeTokenizer + tokenize_data(file_format='h5ad')。对应 notebook cell 18/24。

    字典文件默认走 config 的 gc30M(人)三件套;鼠/其它模型显式传入覆盖。返回产出的 .dataset 路径。
    """
    from geneformer import TranscriptomeTokenizer
    from geneformer_tools import config

    gene_median_file = gene_median_file or config.GENE_MEDIAN_30M
    token_dictionary_file = token_dictionary_file or config.TOKEN_DICT_30M
    gene_mapping_file = gene_mapping_file or config.GENE_MAPPING_30M

    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict=custom_attr_name_dict,
        model_input_size=model_input_size,
        special_token=special_token,
        nproc=nproc,
        gene_median_file=gene_median_file,
        token_dictionary_file=token_dictionary_file,
        gene_mapping_file=gene_mapping_file,
    )
    tokenizer.tokenize_data(
        data_directory=h5ad_dir,
        output_directory=output_directory,
        output_prefix=output_prefix,
        file_format="h5ad",
    )
    return os.path.join(output_directory, output_prefix + ".dataset")
