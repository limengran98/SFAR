import argparse
import torch
import numpy as np
from sklearn.model_selection import train_test_split

def test_val(feature, p=0.84):
    rows = feature.shape[0]
    num_rows = int(rows * p)
    sampled = feature[:num_rows] 
    return sampled

def relabel_nodes(index, edge_index):
    index_map = {old: new for new, old in enumerate(index)}
    print(index_map)
    edge_index = [index_map[i] for i in edge_index]
    return edge_index

def sample_y_nodes(num_nodes, y, y_ratio, seed):
    """
    Sample nodes with observed labels.
    """
    if y_ratio == 0:
        y_nodes = None
        y_labels = None
    elif y_ratio == 1:
        y_nodes = torch.arange(num_nodes)
        y_labels = y[y_nodes]
    else:
        y_nodes, _ = train_test_split(np.arange(num_nodes), train_size=y_ratio, random_state=seed,
                                      stratify=y.numpy())
        y_nodes = torch.from_numpy(y_nodes)
        y_labels = y[y_nodes]
    return y_nodes, y_labels

def print_log(epoch, loss_list, acc_list):
    """
    Print a log during the training.
    """
    print(f'{epoch:5d}', end=' ')
    print(' '.join(f'{e:.4f}' for e in loss_list), end=' ')
    print(' '.join(f'{e:.4f}' for e in acc_list))

def to_device(gpu):
    """
    Return a PyTorch device from a GPU index.
    """
    if gpu is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')

def str2bool(v):
    """
    Convert a string variable into a bool.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ['true']:
        return True
    elif v.lower() in ['false']:
        return False
    else:
        raise argparse.ArgumentTypeError()
    
    