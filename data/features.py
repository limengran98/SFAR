import numpy as np
import torch

def load_pretrained_lm_features(llm_emb_path, llm_model_name, lm_model_name, seed, num_nodes, feature_dim):
    """
    Load pretrained language model (LM) features.
    """
    print("Loading pretrained LM features (title and abstract) ...")
    lm_emb_path_ta = f"{llm_emb_path}/Origin/{lm_model_name}-seed{seed}.emb"
    original_size = 173407232
    target_size = num_nodes * feature_dim
    factor = original_size // target_size

    # Load and reshape title/abstract embeddings
    features_TA = torch.from_numpy(np.array(
        np.memmap(lm_emb_path_ta, mode='r', dtype=np.float16)
    )).to(torch.float32)
    features_TA = features_TA[:factor * target_size].reshape((target_size, factor)).mean(axis=1)
    features_TA = features_TA.reshape(num_nodes, feature_dim)

    print("Loading pretrained LM features (explanations) ...")
    lm_emb_path_e = f"{llm_emb_path}/{llm_model_name}/{lm_model_name}-seed{seed}.emb"

    # Load and reshape explanation embeddings
    features_E = torch.from_numpy(np.array(
        np.memmap(lm_emb_path_e, mode='r', dtype=np.float16)
    )).to(torch.float32)
    features_E = features_E[:factor * target_size].reshape((target_size, factor)).mean(axis=1)
    features_E = features_E.reshape(num_nodes, feature_dim)

    return features_TA, features_E