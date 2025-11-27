## 🧠 SFAR: Semantic Fusion Attribute Recovery for Text Attribute Missing Graphs via Large Language Model Knowledge Generalization

This project implements **SFAR**, a graph representation learning framework that incorporates LLM-derived features, graph propagation, and self-supervised contrastive learning. It also includes downstream node classification using both MLP and GCN classifiers.

---

### 🔧 Requirements

* Python 3.7+
* PyTorch ≥ 1.10
* PyTorch Geometric
* scikit-learn
* NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```
---

### 🚀 How to Run

Train the model and evaluate reconstruction & classification:

```bash
python mp.py --data cora
```

Optional arguments:

| Argument     | Description                             | Default  |
| ------------ | --------------------------------------- | -------- |
| `--data`     | Dataset name (`cora`, `citeseer`, etc.) | `'cora'` |
| `--missrate` | Feature missing ratio                   | `0.6`    |
| `--num_iter` | Propagation steps in AFP module         | `20`     |
| `--epochs`   | Number of training epochs               | `50`     |
| `--gpu`      | GPU ID to use (`0` for CUDA:0)          | `0`      |

---

### 📊 Evaluation Metrics

* **Feature Reconstruction**:

  * Recall\@10 / @20 / @50
  * nDCG\@10 / @20 / @50

* **Downstream Node Classification**:

  * MLP & GCN-based classifiers
  * Metrics: Accuracy, Macro-F1, Precision, Recall
  * 5-fold cross-validation

---

### 📦 Output Files

All outputs are saved in:

```
{DATASET}/embeddings/
├── z.pt          # Final node embeddings
├── z1.pt         # LLM feature projection
├── z2.pt         # Graph feature projection
├── x_feature.pt  # Graph propagated features
├── llmfeatures.pt
├── train_nodes.pt / test_nodes.pt
```

---

### 📌 Notes

* Pretrained LLM-based node features must be pre-generated and placed in the expected folder structure.
* This project is research-oriented and designed for academic use.

---

### 🧑‍💻 Citation

If you use or adapt this project, please cite appropriately based on your related work. This repo is built for reproducibility and educational purposes.# SFAR
