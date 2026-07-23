<div align="center">

# ✨ DAGeR: Degree-Adaptive Graph Rewiring for Long-Tail Domain-Specific Knowledge Graph Fusion ✨

[![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/eduzrh/Tail_DKGF)
[![Language: Python 3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/eduzrh/Tail_DKGF/issues)

<p>
 📰 <a href="#-introduction">Introduction</a> |
 🏗️ <a href="#%EF%B8%8F-architecture">Architecture</a> |
 ⚙️ <a href="#%EF%B8%8F-installation">Installation</a> |
 🚀 <a href="#-quick-start">Quick Start</a> |
 📦 <a href="#-datasets">Datasets</a> |
 📖 <a href="#-usage">Usage</a> |
 🔬 <a href="#-reproducibility">Reproducibility</a> |
 📜 <a href="#-license">License</a> |
 📬 <a href="#-contact">Contact</a> |
 📑 <a href="#-citation">Citation</a>
</p>

</div>

---

## 📰 Latest News

| 🆕 Updates          | 📅 Date | 📝 Description                            |
| ------------------- | ------- | ----------------------------------------- |
| 🎉 **Code Release** | -       | DAGeR codebase and example dataset now available |

> **Before running:** After cloning, unzip the bundled example dataset: `unzip data/W-I-S1.zip -d data/`.

---

## 📰 Introduction

**Domain-specific knowledge graph fusion (DKGF)** aims to mine highly relevant knowledge from general knowledge graphs (GKGs, e.g., Wikidata, YAGO, DBpedia) to enrich the coverage of domain-specific knowledge graphs (DKGs, e.g., ICEWS, DrugBank, UMLS). In real-world DKGF, a large fraction of the entities to be fused are **long-tail entities** whose endpoint connectivity is extremely sparse, which makes fusion suffer from two fundamental bottlenecks: **identity ambiguity** (the impoverished neighborhood of a long-tail GKG entity cannot reliably determine whether it is relevant to the target domain) and **integration uncertainty** (candidate cross-graph links for sparse entities are noisy and hard to verify).

We propose **DAGeR**, a **degree-adaptive graph rewiring** paradigm specifically tailored for long-tail DKGF. Instead of passively accepting the structurally impoverished raw neighborhood distributions, DAGeR dynamically reconstructs optimal latent structural paths to bridge isolated entities from the GKG to the DKG, and injects the rewired topology into the base KG encoder through a degree-adaptive dual-track fusion mechanism.

## 🔥 Key Features

| Feature | Icon | Description |
| --------------------------------------- | ---- | ----------------------------------------------------------- |
| **Degree-Aware Context Expansion**       | 🔄  | Selectively enriches the local structural context of long-tail entities and purges uninformative high-frequency relations |
| **Cross-Graph Virtual Subgraph**         | 🔍  | Generates virtual edges anchoring long-tail GKG entities to structurally compatible DKG counterparts |
| **Multi-Constraint Link Refining**       | 📐  | Prunes incompatible bridges via degree monotonicity, spectral consistency, domain coverage, and domain specificity constraints |
| **Degree-Adaptive Dual-Track Fusion**    | 🔗  | Hybrid structural/embedding scoring with a degree-conditioned fusion coefficient and iterative refinement |
| **Plug-and-Play**                        | 🔌  | Works with any KG encoder (TransE, RotatE, TransH, DistMult, GNN, BERT, ExeFuse, ...) |

---

## 🏗️ Architecture

DAGeR consists of four key modules:

**Module 1: Degree-Aware Expansion Knowledge Generation** 🔄
Applies a degree-aware expansion operator to augment the local structural context of long-tail GKG entities, followed by information-theoretic context purging (IDF-weighted signatures).

**Module 2: Cross-Graph Virtual Subgraph Construction** 🔍
Performs domain target entity identification and cross-graph candidate link generation via degree-adaptive parametric simulation, producing virtual edges between the GKG and the DKG.

**Module 3: Multi-Constraint Virtual Link Refining** 📐
Evaluates each candidate link under four constraints (degree monotonicity, spectral consistency, domain coverage, domain specificity) and prunes incompatible bridges via constraint fusion.

**Module 4: Virtual Structure Hybrid Injector & Fusion Training** 🔗
Injects the rewired topology into the base encoder through a degree-adaptive dual-track (structural + embedding) scoring mechanism, and refines predictions through iterative optimization with delta-driven early termination.

> 📖 For detailed architecture descriptions and theoretical foundations, refer to the accompanying paper.

**Code organization:**

- `DAGeR_main.py`: Main training/inference entry (full pipeline with iterative refinement)
- `dager_core.py`: All four modules, encoder registry, and the data pipeline
- `fusion_eval.py`: Final evaluation (Entity Discovery + Fusion Triple metrics)
- `tools/`: Degree statistics and long-tail bucket extraction utilities
- `scripts/`: One-command experiment scripts

---

## ⚙️ Installation

### 📋 Prerequisites

```bash
pip install -r requirements.txt
```

### 📦 Main Dependencies

| Package        | Version   | Purpose                          |
| -------------- | --------- | -------------------------------- |
| 🐍 **Python**  | >= 3.8    | Core language (tested on 3.10)   |
| 🔥 **PyTorch** | >= 1.12.0 | Deep learning framework          |
| 📊 **NumPy**   | >= 1.21.0 | Numerical computing              |
| ⏳ **Tqdm**    | >= 4.62.0 | Progress bars                    |

> 💡 **Note:** A CUDA-capable GPU is recommended; CPU works but is slow.

---

## 📦 Datasets

We evaluate on six datasets from **DKGF-Bench**, covering four domains. Each dataset couples a domain-specific KG with a general-purpose KG and a set of cross-KG fusion supervision triples. The two political datasets are each released in two supervision splits (S1, S2).

| Domain     | Dataset            | DKG        | GKG      | #Fused Facts |
| ---------- | ------------------ | ---------- | -------- | ------------ |
| Political  | ICEWS-WIKI (W-I)   | ICEWS      | Wikidata | 796,254      |
| Political  | ICEWS-YAGO (Y-I)   | ICEWS      | YAGO     | 451,158      |
| Biomedical | DrugBank-WIKI      | DrugBank   | Wikidata | 532,810      |
| Biomedical | UMLS-WIKI          | UMLS       | Wikidata | 1,204,561    |
| Academic   | OpenAlex-DBpedia   | OpenAlex   | DBpedia  | 2,105,340    |
| Business   | Crunchbase-DBpedia | Crunchbase | DBpedia  | 642,109      |

An example dataset (**W-I-S1**) is bundled with this repository (`data/W-I-S1.zip`). The full dataset suite follows the DKGF-Bench release; see [data/README.md](data/README.md) for the file format and download instructions.

**Dataset format** — each dataset folder (e.g., `data/W-I-S1/`) contains:

- `triples_1` / `triples_2`: Relation triples of KG1 (GKG side) / KG2 (DKG side)
- `ent_ids_1` / `ent_ids_2`: Entity IDs of KG1 / KG2
- `rel_ids_1` / `rel_ids_2`: Relation IDs of KG1 / KG2
- `sup_triples_1_fusion`: Fusion supervision triples for training
- `ref_triples_1_fusion`: Ground-truth fused triples (positives) for testing
- `false_ref_triples_1_fusion`: Hard negative triples for testing
- `ref_relative_ent_ids_2`: Ground-truth relevant entities in KG2
- `time_id`: Time IDs (for temporal KGs)

---

## 🚀 Quick Start

### Step 1: Clone the Repository 📥

```bash
git clone https://github.com/eduzrh/Tail_DKGF.git
cd Tail_DKGF
```

### Step 2: Prepare the Example Dataset 📦

```bash
unzip data/W-I-S1.zip -d data/
```

### Step 3: Run the Smoke Test ▶️

```bash
# ~3 min on GPU: verifies the whole pipeline end-to-end
bash scripts/run_quick_test.sh
```

### Step 4: Full Training 🏋️

```bash
python DAGeR_main.py \
    --data_dir data/W-I-S1 \
    --name W-I-S1 \
    --output_dir outputs \
    --dim 128 --max_k 16 --epochs 30 --batch_size 256 \
    --scorer transe
```

### Step 5: View Results 📊

Training reports ACC/F1 (including long-tail `LT` metrics) every epoch, and the best results are saved to `outputs/results_W-I-S1.txt` together with the model checkpoint `outputs/model_W-I-S1.pt`.

For evaluating a standalone prediction file (yours or any baseline's, one fused triple per line), use:

```bash
python fusion_eval.py \
    --data_dir data/W-I-S1 \
    --pred path/to/pred_method.txt \
    --out path/to/eval_report.txt
```

This reports:

- **Entity Discovery**: ACC, P, R, F1 on KG2 relevant-entity identification
- **Fusion Triple**: ACC, P, R, F1 on fusion triple correctness

---

## 📖 Usage

### Parameter Descriptions

| Parameter            | Type  | Default      | Description                                             |
| -------------------- | ----- | ------------ | ------------------------------------------------------- |
| `--data_dir`         | str   | **Required** | Path to dataset directory                               |
| `--name`             | str   | `dataset`    | Dataset name (used in output file names)                |
| `--output_dir`       | str   | `./outputs`  | Output directory                                        |
| `--dim`              | int   | 128          | Embedding dimension                                     |
| `--max_k`            | int   | 16           | Maximum number of anchors per entity (K)                |
| `--epochs`           | int   | 40           | Training epochs                                         |
| `--batch_size`       | int   | 512          | Batch size                                              |
| `--lr`               | float | 0.001        | Learning rate (Adam)                                    |
| `--seed`             | int   | 2025         | Random seed                                             |
| `--lambda_bridge`    | float | 0.2          | Weight of the bridge (structural) loss                  |
| `--beta_entropy`     | float | 0.01         | Entropy regularization coefficient                      |
| `--expansion_thresh` | int   | 2            | Degree threshold triggering context expansion           |
| `--scorer`           | str   | `transe`     | Base KG encoder (see table below)                       |
| `--n_rounds`         | int   | 3            | Iterative refinement rounds (T)                         |
| `--ablation`         | str   | None         | `no_bridge` or `emb_only` for ablation studies          |
| `--bert_path`        | str   | None         | Local BERT weights (only for `--scorer bert`)           |

### Plug-and-Play Encoders (`--scorer`)

| Encoder  | `--scorer` | Description                              |
| -------- | ---------- | ---------------------------------------- |
| TransE   | `transe`   | Translation-based (default, fastest)     |
| RotatE   | `rotate`   | Complex rotation in C-space              |
| TransH   | `transh`   | Relation-specific hyperplane projection  |
| DistMult | `distmult` | Bilinear diagonal scoring                |
| GNN      | `gnn`      | Relation-gated message passing           |
| BERT     | `bert`     | Text-informed BERT-initialized backbone  |
| ExeFuse  | `exefuse`  | MLP-projected + pairwise fusion          |

**Adding your own encoder:**

```python
# In dager_core.py, add a new scorer class:
class MyScorer(BaseKGScorer):
    def score(self, he, re, te):
        sc = he + re - te
        return -torch.sum(sc ** 2, dim=-1) * 0.5

# Register it:
SCORER_REGISTRY["my_encoder"] = MyScorer
# Then use: --scorer my_encoder
```

### Ablation Studies

```bash
# Remove the multi-constraint virtual link refining (anchor bridges)
python DAGeR_main.py ... --ablation no_bridge

# Embedding-only track (disable the structural track in the hybrid injector)
python DAGeR_main.py ... --ablation emb_only

# Or run the whole suite:
bash scripts/run_ablation.sh data/W-I-S1 W-I-S1
```

### Long-Tail (Degree-Bucketed) Evaluation

```bash
# 1. Compute node degrees for each dataset
python tools/compute_node_degrees.py --base_dir data --datasets W-I-S1

# 2. Extract long-tail test buckets (d_min <= 4) grouped by degree
python tools/extract_long_tail.py --base_dir data --datasets W-I-S1 \
    --output_dir outputs/long_tail_data --extract_by_degree
```

The degree buckets (`0-1`, `2`, `3`, `4`, `ALL_longtail`, `>=5`) are then evaluated with `fusion_eval.py` on the corresponding bucket files.

---

## 🔬 Reproducibility

- **Hyperparameters**: All settings are exposed as command-line arguments; the default configuration matches the paper (dim=128, K=16, T=3, TransE base scorer, Adam optimizer).
- **Random Seeds**: Fixed via `--seed` (default 2025).
- **Environment**: Tested on Python 3.10, PyTorch >= 1.12, NVIDIA RTX 4090 GPUs.

To reproduce the plug-and-play benchmark on W-I-S1:

```bash
bash scripts/run_all.sh
```

Expected runtime: roughly 1 hour per encoder on a modern 24GB GPU.

---

## 📜 License

[MIT License](LICENSE) - Copyright notices preserved.



## 🙏 Acknowledgement

The datasets used in this repository are derived from the DKGF-Bench evaluation suite. We also sincerely appreciate the following open source projects: [PyTorch](https://pytorch.org/), [NumPy](https://numpy.org/), [tqdm](https://github.com/tqdm/tqdm).
