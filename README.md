# geneformer-tools

把原本散落、且**塞在 site-packages 里**的自定义 Geneformer 代码,集中到一个 git 管理的项目中。
解决三件事:(1) 自定义代码不再随 `pip install -U geneformer` 蒸发;(2) 纳入版本控制;(3) 路径/密钥集中到 `config`,不再硬编码。

## 结构
```
geneformer-tools/
├── geneformer_tools/          # 库代码(可 import)
│   ├── __init__.py            #   re-export: DualPerturber, ResultAnalysis
│   ├── config.py              #   集中配置:路径 + 密钥(从环境变量读)
│   ├── perturb.py             #   DualPerturber(删 g1 + 过表达 g2 双扰动)
│   └── analysis.py            #   ResultAnalysis(OpenTargets/EuropePMC 下游分析)
├── scripts/                   # 驱动脚本(命令行跑)
│   ├── run_isp.py             #   原 isp.py
│   └── run_isp_dual.py        #   原 isp_dual.py
├── pyproject.toml             # editable 安装用
├── .gitignore                 # 排除数据/模型/输出
└── PLAN.md                    # 把 ISP workflow notebook 拆成 .py 的路线图
```

## 安装(在 genefoemer conda 环境里)
```bash
conda activate genefoemer
pip install -e ~/Projects/geneformer-tools     # editable:改代码立即生效,且 import 路径稳定
```
装完后任意位置 `from geneformer_tools import DualPerturber, ResultAnalysis` 即可。
为兼容旧 notebook,`site-packages/geneformer/mynewfun.py` 已替换成一个 **shim**(转发到本项目),
所以 `from geneformer.mynewfun import ...` 仍能用——但**新代码请直接 import geneformer_tools**。

## 配置(用前 export;见 config.py)
```bash
export NCBI_API_KEY="<你的 key>"     # ⚠️ 旧 key 曾硬编码泄露,请轮换一个新的
export GENEFORMER_REPO="$HOME/Projects/geneformer/Geneformer"
export GF_FINETUNED_MODEL="/path/to/.../checkpoint-XXXX"   # ISP 用的微调模型
```

## 已知 gotcha(已在代码注释/PLAN 中标注)
- ISP 脚本必须有 `if __name__ == "__main__":`,并在构造 InSilicoPerturber 后
  `multiprocessing.set_start_method("fork", force=True)`,否则 `datasets>=4.x` 下 `.map` 子进程会崩。
- `InSilicoPerturberStats` 的 `mode` 必须和扰动方式匹配(goal_state_shift / aggregate_data / mixture_model)。

## 建议(下一步)
- `pip install nbstripout && nbstripout --install` —— 提交 notebook 时自动去掉输出,避免 5MB 的 diff。
- 稳定逻辑持续往 `geneformer_tools/` 沉淀,notebook 只留探索与出图。
