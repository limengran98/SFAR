import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric import datasets

from .features import load_pretrained_lm_features
from .utils import validate_edges

def load_data(args, split=None, seed=None, verbose=False, normalize=False,
              validate=False):
    """
    Load a dataset from its name.
    """
    root = '../data'
    dataset = args.data

    llm_emb_path = './LLMs'
    llm_model_name = 'ChatGPT3.5' #LLaMA3
    lm_model_name = 'bert-large-uncased'
    seed = 0

    if dataset == 'cora':
        data = datasets.Planetoid(root, 'Cora')
        feature_dim = data[0].x.shape[1]
        num_nodes = data[0].x.shape[0]
        features_TA, features_E = load_pretrained_lm_features(llm_emb_path, llm_model_name, lm_model_name, seed, num_nodes, feature_dim)
        llmfeatures = features_TA + features_E

    elif dataset == 'citeseer':
        data = datasets.Planetoid(root, 'Citeseer')
        feature_dim = data[0].x.shape[1]
        num_nodes = data[0].x.shape[0]
        features_TA, features_E = load_pretrained_lm_features(llm_emb_path, llm_model_name, lm_model_name, seed, num_nodes, feature_dim)
        llmfeatures = features_TA + 0.1 * features_E

    elif dataset == 'computers':
        data = datasets.Amazon(root, 'Computers')
        feature_dim = data[0].x.shape[1]
        num_nodes = data[0].x.shape[0]
        features_TA, features_E = load_pretrained_lm_features(llm_emb_path, llm_model_name, lm_model_name, seed, num_nodes, feature_dim)
        llmfeatures = features_TA + features_E

    elif dataset == 'photo':
        data = datasets.Amazon(root, 'Photo')
        feature_dim = data[0].x.shape[1]
        num_nodes = data[0].x.shape[0]
        features_TA, features_E = load_pretrained_lm_features(llm_emb_path, llm_model_name, lm_model_name, seed, num_nodes, feature_dim)
        llmfeatures = features_TA + features_E

    else:
        raise ValueError(dataset)

    node_x = data.data.x
    node_y = data.data.y
    edges = data.data.edge_index

    if validate:
        validate_edges(edges)

    if normalize:
        assert (node_x < 0).sum() == 0  # all positive features
        norm_x = node_x.clone()
        norm_x[norm_x.sum(dim=1) == 0] = 1
        norm_x = norm_x / norm_x.sum(dim=1, keepdim=True)
        node_x = norm_x

    trn_size, test_size = split
    indices = np.arange(node_x.shape[0])
    trn_nodes, test_nodes = train_test_split(indices, test_size=test_size, random_state=seed,
                                                stratify=node_y)

    trn_nodes = torch.from_numpy(trn_nodes)
    test_nodes = torch.from_numpy(test_nodes)

    return data[0], trn_nodes, test_nodes, llmfeatures, feature_dim

def load_data_classify(dataset):
    """
    Load a dataset from its name.
    """
    root = '../data'

    if dataset == 'cora':
        data = datasets.Planetoid(root, 'Cora')

    elif dataset == 'citeseer':
        data = datasets.Planetoid(root, 'Citeseer')

    elif dataset == 'computers':
        data = datasets.Amazon(root, 'Computers')

    elif dataset == 'photo':
        data = datasets.Amazon(root, 'Photo')

    else:
        raise ValueError(dataset)

    return data[0]