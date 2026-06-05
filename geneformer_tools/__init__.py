"""geneformer_tools —— 自定义 Geneformer 扩展。

从 site-packages/geneformer/mynewfun.py 迁出至此,纳入版本控制(2026-06)。
- DualPerturber: "删 gene1 + 过表达 gene2" 的双扰动(继承官方 InSilicoPerturber)
- ResultAnalysis: ISP 结果的 Open Targets / EuropePMC 下游分析
"""
from .perturb import DualPerturber
from .analysis import ResultAnalysis

__all__ = ["DualPerturber", "ResultAnalysis"]
__version__ = "0.1.0"
