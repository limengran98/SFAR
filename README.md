# SFAR

This repository contains a compact implementation of **SFAR** for missing node feature reconstruction and downstream node classification on Cora, Citeseer, Amac, and Amap.

## Structure

```text
main.py                 # experiment entry point
configs/default.json    # default parameters
sfar/                   # core implementation
tools/                  # optional tuning/export utilities
```

Generated files are saved under `outputs/<dataset>/<run_name>/`. Datasets, LLM embeddings, tensors, and results are ignored by git.

## Run

Feature reconstruction and node classification:

```bash
python main.py --dataset cora --gpu 0
python main.py --dataset citeseer --gpu 0
python main.py --dataset amac --gpu 0
python main.py --dataset amap --gpu 0
```

Run all four datasets:

```bash
python main.py --gpu 0 --run-name sfar_main
```

Run all four datasets with the full-graph GCN classifier protocol:

```bash
python main.py --gpu 0 --gcn-graph-scope full --run-name full_graph_gcn --no-save-tensors
```


Feature reconstruction only:

```bash
python main.py --dataset cora --skip-classification --gpu 0
```

## Data And HERP Features

PyG datasets are loaded as:

- `cora` -> `Planetoid("Cora")`
- `citeseer` -> `Planetoid("Citeseer")`
- `amac` -> `Amazon("Computers")`
- `amap` -> `Amazon("Photo")`

HERP semantic features are offline inputs. Put `.emb` files under:

```text
LLMs/Origin/bert-large-uncased.emb
LLMs/ChatGPT3.5/bert-large-uncased.emb
LLMs/LLaMA3/bert-large-uncased.emb
```

link: [LLMs](https://drive.google.com/file/d/1RUD5K466uJFWqDuRxTVPYCRZHxcv6p_m/view?usp=sharing)

Raw LLM responses and full text corpora are not included. The preprocessing follows the TAPE-style pipeline:

https://github.com/XiaoxinHe/TAPE/

To convert TAPE-style `.emb` files into cached tensors:

```bash
python tools/export_herp_embeddings.py \
  --origin-emb LLMs/Origin/bert-large-uncased.emb \
  --expert-emb LLMs/ChatGPT3.5/bert-large-uncased.emb \
  --num-nodes 2708 \
  --feature-dim 1433 \
  --output outputs/corallmfeatures.pt
```

## Utilities

AFP tuning:

```bash
python tools/tune_afp.py --dataset citeseer
```

CKD/classifier tuning:

```bash
python tools/tune_classification.py --datasets cora,citeseer --gpu 0
python tools/tune_classifier_head.py --dataset citeseer --gpu 0
```

Push code-only changes:

```bash
./push_code_only.sh "update code"
```
