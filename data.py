
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric import datasets
from torch_sparse import SparseTensor



def to_edge_tensor(edge_index):
    """
    Convert an edge index tensor to a SparseTensor.
    """
    row, col = edge_index
    value = torch.ones(edge_index.size(1))
    return SparseTensor(row=row, col=col, value=value)


def validate_edges(edges):
    """
    Validate the edges of a graph with various criteria.
    """
    # No self-loops
    for src, dst in edges.t():
        if src.item() == dst.item():
            raise ValueError()

    # Each edge (a, b) appears only once.
    m = defaultdict(lambda: set())
    for src, dst in edges.t():
        src = src.item()
        dst = dst.item()
        if dst in m[src]:
            raise ValueError()
        m[src].add(dst)

    # Each pair (a, b) and (b, a) exists together.
    for src, neighbors in m.items():
        for dst in neighbors:
            if src not in m[dst]:
                raise ValueError()



def load_pretrained_lm_features(llm_emb_path, llm_model_name, lm_model_name, seed, num_nodes, feature_dim):
    """
    Load pretrained language model (LM) features.

    Args:
        llm_emb_path (str): Path to the directory containing pretrained LM embeddings.
        llm_model_name (str): Name of the language model (used in file path).
        lm_model_name (str): Alias or identifier for the language model (used in filename).
        seed (int): Random seed used for generating the embedding.
        num_nodes (int): Number of nodes in the graph.
        feature_dim (int): Dimensionality of each node feature vector.

    Returns:
        tuple: A tuple containing two torch.Tensors:
            - features_TA: Features from titles and abstracts.
            - features_E: Features from explanations.
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

    #test_nodes, val_nodes = train_test_split(test_nodes, test_size=1/6, random_state=seed, stratify=node_y[test_nodes])


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