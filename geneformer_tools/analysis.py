"""ResultAnalysis —— ISP 结果的下游文献/通路分析(Open Targets + EuropePMC)。
原属 site-packages/geneformer/mynewfun.py,迁出至本项目以纳入版本控制(2026-06 迁移)。
API key 从 config(环境变量)读取,不再硬编码。
"""
import ast
import json
import logging
import os
import time
import warnings
from itertools import combinations, chain
from xml.etree import ElementTree

import pandas as pd
import requests
import tqdm
from tqdm.rich import trange

from .config import OPENTARGETS_API_KEY, LOG_DIR

logger = logging.getLogger(__name__)


class ResultAnalysis:
    """
    分析逻辑
    1. 提取ISP结果中的基因列表
    2. 读取OT文件并过滤ISP结果中的基因
    4. 查询结果基因对应的EuropePMC文献数量
    5. 记录文献信息到日志文件
    """

    def __init__(
        self, 
        isp_analysis_results_path: str = None,
        isp_gene_FDR_threshold: float = 0.05,
        isp_gene_N_Detections_threshold: int = 200
    ):
        """
        初始化结果分析类
        :param isp_analysis_results_path: 分析结果路径
        :param isp_gene_FDR_threshold: FDR阈值
        :param isp_gene_N_Detections_threshold: 最小检测数阈值
        """
        warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)
        self.isp_path = isp_analysis_results_path
        self.isp_gene_FDR_threshold = isp_gene_FDR_threshold
        self.isp_gene_N_Detections_threshold = isp_gene_N_Detections_threshold

    def extract_isp_result_gene_lsit(self):
        """
        提取genelist
        :return: 分析结果
        """
        isp_result = pd.read_csv(self.isp_path, sep='\t', header=0)
        isp_result = isp_result[(isp_result['Sig'] == 1) & 
                                (isp_result["Goal_end_FDR"] < self.isp_gene_FDR_threshold) & 
                                (isp_result["N_Detections"] > self.isp_gene_N_Detections_threshold)]

        isp_result_genepair_list = isp_result['Gene_name'].apply(ast.literal_eval).to_list()
        isp_result_gene_list = list(set(chain.from_iterable(isp_result_genepair_list)))

        return isp_result_genepair_list,isp_result_gene_list

    def query_EuropePMC(
        self,
        gene_list: list = None,
        gene_id: str = "ENSG00000139618",
        efo_id: str = "EFO_0003767",
        size: int = 50,
        cursor: str = None
    ):
        """
        使用EuropePMC API查询基因文献
        :param isp_result_gene: 基因(未使用)
        :param gene_id: 基因ID
        :param efo_id: 疾病的EFO ID
        :param size: 查询返回文献数
        :param cursor: 分页游标
        :return: 响应对象
        """
        query_string = """
        query EuropePMCQuery(
            $ensemblId: String!
            $efoId: String!
            $size: Int!
            $cursor: String
        ) {
            disease(efoId: $efoId) {
                id
                europePmc: evidences(
                    ensemblIds: [$ensemblId]
                    enableIndirect: true
                    size: $size
                    cursor: $cursor
                    datasourceIds: ["europepmc"]
                ) {
                    count
                    cursor
                    rows {
                        disease {
                            name
                            id
                        }
                        target {
                            approvedSymbol
                            id
                        }
                        literature
                        textMiningSentences {
                            tStart
                            tEnd
                            dStart
                            dEnd
                            section
                            text
                        }
                        resourceScore
                    }
                }
            }
        }
        """
        base_url = "https://api.platform.opentargets.org/api/v4/graphql"
        EuropePMC = {}
        for gene in trange(len(gene_list), desc="Querying EuropePMC", initial=1):
            #print(gene)
            gene_id = gene_list[gene]
            variables = {"ensemblId": gene_id, "efoId": efo_id, "size": size, "cursor": cursor}
            response = requests.post(base_url, json={"query": query_string, "variables": variables})
            if response.status_code == 200:
                #print(gene_id)
                EuropePMCQuery_data = response.json()
                EuropePMC[gene_id] = EuropePMCQuery_data
                time.sleep(0.5)
            else:
                print(f"Error fetching data for gene {gene_id}: {response.status_code}")
                print(response.text)
                EuropePMC[gene_id] = None
        return EuropePMC


    def query_disease_associated_targets(self, efo_id: str = "EFO_0003767", size: int = 50):
        """
        使用API查询与疾病相关的目标
        :param efo_id: 疾病的EFO ID, 例如 "EFO_0003767"
        :param size: 返回的目标数量, 默认是50
        :return: 
        """
        query_string = """
        query associatedTargets(
            $efoId: String!
            $size: Int!
        ) {
          disease(efoId: $efoId) {
            id
            name
            associatedTargets(page: {index: 0, size: $size}) {
              count
              rows {
                target {
                  id
                  approvedSymbol
                  symbolSynonyms { label, source }
                  obsoleteSymbols { label, source }
                  synonyms{label,source}
                }
                score
              }
            }
          }
        }
        """

        variables = {"efoId": efo_id, "size": size}
        base_url = "https://api.platform.opentargets.org/api/v4/graphql"
        response = requests.post(base_url, json={"query": query_string, "variables": variables})

        if response.status_code == 200:
            data = response.json()

            OT_ENSG2gene_dict = {}
            OT_ENSG2score_dict = {}
            OT_ENSG2approvedSymbol_dict = {}
            for target in data['data']['disease']['associatedTargets']['rows']:
                id = target['target']['id']
                approved_symbol = target['target']['approvedSymbol']
                symbol_synonyms = [synonym['label'] for synonym in target['target']['symbolSynonyms']]
                obsolete_symbols = [obsolete['label'] for obsolete in target['target']['obsoleteSymbols']]
                synonyms = [synonym['label'] for synonym in target['target']['synonyms']]

                OT_ENSG2gene_dict[id] = set([approved_symbol] + symbol_synonyms + obsolete_symbols + synonyms)
                OT_ENSG2score_dict[id] = target['score']
                OT_ENSG2approvedSymbol_dict[id] = approved_symbol
            return OT_ENSG2gene_dict, OT_ENSG2score_dict, OT_ENSG2approvedSymbol_dict
        else:
            print(f"Error fetching data: {response.status_code}")
            print(response.text)
            return None

    def update_geneList_to_newestSymbolList(self, 
                                            genelist: list, 
                                            genesymbol_withSynonyms_dict: dict,
                                            esgn_to_gene_dict: dict):
        synonym_to_esgn = {syn: key for key, synonyms in genesymbol_withSynonyms_dict.items() for syn in synonyms}
        updated_genelist_ESGN = [synonym_to_esgn[gene] for gene in genelist if synonym_to_esgn.get(gene) is not None]
        
        updated_genelist_symbol = [
            esgn_to_gene_dict.get(gene) for gene in updated_genelist_ESGN
        ]
        #print(f"Updated gene number: {len(updated_genelist_ESGN)}/{len(genelist)}")
        return updated_genelist_ESGN,updated_genelist_symbol

    def fetch_literature_info(pmid_or_pmcid: str = "32843955", api_key: str = None):
        try:
            # Determine database type based on ID prefix
            pmid_or_pmcid = str(pmid_or_pmcid[0])
            db = 'pmc' if pmid_or_pmcid.startswith('PMC') else 'pubmed'
            params = {
                "db": db,
                "id": pmid_or_pmcid,
                "retmode": "xml",
                "api_key": api_key
            }
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            root = ElementTree.fromstring(response.content)
            title_element = root.find('.//Item[@Name="Title"]')
            doi_element = root.find('.//Item[@Name="DOI"]')
            
            # Extract title and DOI
            if title_element is None:
                raise ValueError(f"Title element not found for ID: {pmid_or_pmcid}")
            title = title_element.text.strip()
            doi = doi_element.text.strip() if doi_element is not None else f"{pmid_or_pmcid} DOI not found"
            
            # Generate link based on database type
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_or_pmcid}" if db == 'pubmed' else f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmid_or_pmcid}"
            
            # Return formatted result
            time.sleep(0.1)  # Avoid sending requests too quickly
            return {
                "title": title,
                "doi": doi,
                "link": link
            }
        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")
        except ElementTree.ParseError as e:
            print(f"XML parsing failed: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None


    def logging_literature_data(self,
                    api_key: str = None,
                    log_file_dir: str = None
        ):
        # 密钥/日志目录默认从 config(环境变量)读取,不再硬编码
        api_key = api_key or OPENTARGETS_API_KEY
        log_file_dir = log_file_dir or LOG_DIR
        os.makedirs(log_file_dir, exist_ok=True)
        # 设置logger文档
        filename = os.path.splitext(os.path.basename(self.isp_path))[0]
        first_two_elements = filename.split('_')[:2]
        new_filename = '_'.join(first_two_elements + ['isp_analysis'])
        log_file_path = os.path.join(log_file_dir, f"{new_filename}.log")
        # 配置主日志
        logger = logging.getLogger('ResultAnalysisLogger')
        logger.setLevel(logging.INFO)  
        file_handler = logging.FileHandler(log_file_path)
        logger.addHandler(file_handler)
        log_buffer = []

        # 提取isp基因列表
        isp_result_genepair_list,isp_gene_list = self.extract_isp_result_gene_lsit()
        log_buffer.append(f"REPORT\n\n"
                    f"Geneformer InSilicoPerturber gene pair number: {len(isp_result_genepair_list)}\n"
                    f"Gene number: {len(isp_gene_list)}\n"
                    )
        # 查询ibd相关的前size=500个基因的OT信息
        OT_ENSG2gene_dict, OT_score_dict, OT_ENSG2approvedSymbol_dict = self.query_disease_associated_targets(efo_id="EFO_0003767", size=1000)

        # 更新isp基因列表到ENSG ID,过滤了不在前size=500个的基因
        isp_gene_list_OT_relative_ESGN, isp_gene_list_OT_relative_symbol = self.update_geneList_to_newestSymbolList(isp_gene_list,OT_ENSG2gene_dict,OT_ENSG2approvedSymbol_dict)
        log_buffer.append(f"Within top 1k OpenTarget IBD related genes : {len(isp_gene_list_OT_relative_ESGN)}/{len(isp_gene_list)}\n")
        # 将剩下的基因组合，并和原来的isp基因对对比
        annotated_gene_combinations = list(combinations(isp_gene_list_OT_relative_symbol, 2))
        valid_combinations = [pair for pair in annotated_gene_combinations if tuple(sorted(pair)) in [tuple(sorted(p)) for p in isp_result_genepair_list]]
        log_buffer.append(f"Valid gene pair number: {len(valid_combinations)}/{len(isp_result_genepair_list)}\n")

        isp_valid_gene_list = list(set(chain.from_iterable(valid_combinations)))
        isp_validgene_list_OT_relative_ESGN, isp_validgene_list_OT_relative_symbol = self.update_geneList_to_newestSymbolList(isp_valid_gene_list,OT_ENSG2gene_dict,OT_ENSG2approvedSymbol_dict)
        # print剩下基因在
        log_buffer.append(f"valid gene number: {len(isp_validgene_list_OT_relative_ESGN)}/{len(isp_gene_list)}\n\n")
        # 查询
        EuropePMCQuery_data = self.query_EuropePMC(isp_validgene_list_OT_relative_ESGN)

        gene_resource_scores = {}
        log_buffer_literature = []
        for geneESGN in trange(len(EuropePMCQuery_data), desc="Processing genes", initial=1):
            geneESGN_key = list(EuropePMCQuery_data.keys())[geneESGN]
            gene_info = EuropePMCQuery_data[geneESGN_key]
            # 获取目标信息
            try:
                rows = gene_info['data']['disease']['europePmc']['rows']
                if rows:
                    target = rows[0]['target']
                    count = gene_info['data']['disease']['europePmc']['count']
                    # 获取文献 ID 列表
                    literature_ids = [row['literature'] for row in rows]
                    resource_score = sum(
                        row['resourceScore'] for row in rows if 'resourceScore' in row
                    )
                    gene_resource_scores[target.get('id', 'N/A')] = resource_score
                    # 获取文献详细信息
                    literature_info = [
                        ResultAnalysis.fetch_literature_info(pmid, api_key=api_key)
                        for pmid in literature_ids
                    ]

                    # 格式化文献信息
                    formatted_literature = "\n".join([
                        f"\t{i+1}. {item.get('title', 'No title')}\n"
                        f"\t   DOI: {item.get('doi', 'N/A')}\n"
                        f"\t   Link: {item.get('link', 'N/A').strip('[]').replace('][', ', ')}"
                        for i, item in enumerate(literature_info) if item is not None
                    ])

                    log_message = (
                        f"Symbol: {target.get('approvedSymbol', 'N/A')}\n"
                        f"ENSG: {target.get('id', 'N/A')}\n"
                        f"Literature evidence:\n"
                        f"    Count: {count}\n"
                        f"    Score: {resource_score}\n"
                        f"{formatted_literature}\n"
                    )
                    log_buffer_literature.append(log_message)
                else:
                    log_buffer_literature.append(f"No literature found for gene {geneESGN_key}.")
                    gene_resource_scores[target.get('id', 'N/A')] = 0
            except Exception as e:
                log_buffer_literature.append(f"Error processing gene {geneESGN_key}: {e}")
                continue

        updated_literature_scores = {OT_ENSG2approvedSymbol_dict.get(ensg, ensg): score 
                 for ensg, score in gene_resource_scores.items()}
        
        pair_scores = {}
        for pair in valid_combinations:
            gene1, gene2 = pair
            score = updated_literature_scores.get(gene1, 0) + updated_literature_scores.get(gene2, 0)
            pair_scores[pair] = score

        score_df = (
            pd.DataFrame(list(pair_scores.items()), columns=['Valid Gene Pair', 'Literature Score'])
            .sort_values(by='Literature Score', ascending=False, inplace=False)  
            .to_string(index=False)  
        )
        log_buffer.append(score_df)

        log_buffer.append("\n\nReference:\n"
                    f"InSilicoPerturber file: {self.isp_path}\n"
                    "Opentarget: https://platform.opentargets.org/disease/EFO_0003767/associations\n"
                    "Literature resource: EuropePMC\n"
                    "Literature Evidence of each Gene\n"
                    )
        try:
            with open(log_file_path, 'a') as target_file:
                target_file.writelines(log_buffer)  # 批量写入
                target_file.writelines(log_buffer_literature)
        except Exception as e:
                logger.error(f"Failed to merge temporary log: {e}")
        print(f"Analysis result has been written in {log_file_path}!!")
        return


########################################################################################################################
########################################################################################################################
########################################################################################################################
