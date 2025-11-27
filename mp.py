import warnings
warnings.filterwarnings("ignore")

import argparse

import os
import time
import torch
from torch import optim
import numpy as np
import argparse
import os
import time
import torch
from torch import optim
import numpy as np
from data.datasets import load_data
from models.sfar import SFAR
from models.layers import Linear, GCN
from utils.propagation import APA
from utils.helpers import to_device, to_recall, to_ndcg
from utils.metrics import to_f1_score, to_recall, to_ndcg # Explicitly check where recall/ndcg are used
import argparse
import torch.nn.functional as F
from torch_geometric.utils import subgraph

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--num_iter', type=int, default=20) 
    parser.add_argument('--conv', type=str, default='gcn')
    parser.add_argument('--missrate', type=float, default=0.6) 
    return parser.parse_args()

def random_mask_tensor(X, mask_ratio, seed=None):
    torch.manual_seed(seed)
    rows, cols = X.size()
    num_zeros = int(mask_ratio * rows * cols)
    zero_indices = torch.randperm(rows * cols)[:num_zeros]
    masked_X = X.clone()
    masked_X.view(-1)[zero_indices] = 0
    return masked_X
"""
Main function.
"""
def main():
    
    args = parse_args()
    print(args)
    device = to_device(args.gpu)
    os.makedirs(os.path.join(args.data, "embeddings"), exist_ok=True)


    data, trn_nodes, test_nodes, llmfeatures, llm_feature_dim = load_data(args, split=(1-args.missrate, args.missrate), seed=args.seed)
    data = data.to(device)
    raw_x =  data.x.clone()
    fp_features = data.x.clone()
    fp_features[test_nodes] = 0
    llmfeatures = llmfeatures.to(device)


    # hybrid attribute feature missingness.
    # fp_features[trn_nodes] = random_mask_tensor(fp_features[trn_nodes], mask_ratio=args.p, seed=args.seed)
    
    propagation_model = APA(data.edge_index, fp_features, trn_nodes)

    # if args.data == 'cora':
    #     args.num_iter = 6
    # elif args.data == 'cite':
    #     args.num_iter = 4
    # else:
    #     args.num_iter = 2

    x_feature = propagation_model.umtp(fp_features, num_iter = args.num_iter).cuda()
    torch.save(llmfeatures, os.path.join(args.data, 'embeddings', 'llmfeatures.pt'))
    torch.save(x_feature, os.path.join(args.data, 'embeddings', 'x_feature.pt'))

    x_feature = x_feature.to(device)
    x_all = raw_x.to(device)
    y_all = data.y.to(device)
    edge_index = data.edge_index.to(device)
    num_classes = (data.y.max() + 1).item()

    k_list = [10, 20, 50]
    recalls = [to_recall(x_feature[test_nodes], x_all[test_nodes], k) for k in k_list]
    ndcgs = [to_ndcg(x_feature[test_nodes], x_all[test_nodes], k) for k in k_list]
    scores = recalls + ndcgs

    labels = [f"Recall@{k}" for k in k_list] + [f"nDCG@{k}" for k in k_list]
    print("Feature Reconstruction:")
    for name, score in zip(labels, scores):
        print(f"  {name}: {score:.4f}")



    model = SFAR(device,  llm_feature_dim, args.hidden_size, args.layers, args.conv, x_feature).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    def update_model(step):
        model.train()
        loss = model.to_losses(edge_index, x_feature, llmfeatures, trn_nodes)
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item()

    
    @torch.no_grad()
    def evaluate():
        model.eval()
        x_hat, z, z1, z2 = model(edge_index, x_feature, llmfeatures)
        return x_hat, z, z1, z2


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_time = time.time()
    start_event.record()


    for epoch in range(args.epochs + 1):
        for _ in range(args.updates):
            loss_val = update_model(epoch > 0)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss_val:.4f}")


    end_event.record()
    end_time = time.time()
    torch.cuda.synchronize()

    
    elapsed_time_sec = (end_time - start_time)
    print(f"Elapsed time (using time module): {elapsed_time_sec} seconds")



    # Output only the last epoch's loss and evaluation
    print(f"Epoch {args.epochs}: Loss {loss_val:.4f}")
    x_hat, z, z1, z2 = evaluate()

    torch.save(z, os.path.join(args.data, 'embeddings', 'z.pt'))
    torch.save(z1, os.path.join(args.data, 'embeddings', 'z1.pt'))
    torch.save(z2, os.path.join(args.data, 'embeddings', 'z2.pt'))
    torch.save(trn_nodes, os.path.join(args.data, 'embeddings', 'train_nodes.pt'))
    torch.save(test_nodes, os.path.join(args.data, 'embeddings', 'test_nodes.pt'))



    x_hat = z[test_nodes].to(device)
    print(x_hat.shape)
    y_all = y_all[test_nodes]
    print('----------------------MLP---------------------------')
    def train(feature, edge_index, y_all, train_idx):
        cl_model.train()
        optimizer.zero_grad()
        out = cl_model(feature)
        loss = F.cross_entropy(out[train_idx], y_all[train_idx])
        loss.backward()
        optimizer.step()
        return float(loss)


    @torch.no_grad()
    def test(feature, edge_index, y_all, test_idx):

        cl_model.eval()
        pred = cl_model(feature)
        mask = test_idx
        f1 = f1_score(y_all[mask].cpu().numpy(),torch.argmax(pred[mask],dim=1).cpu().numpy(),average = 'macro')
        acc = accuracy_score(y_all[mask].cpu().numpy(),torch.argmax(pred[mask],dim=1).cpu().numpy())
        pre = precision_score(y_all[mask].cpu().numpy(),torch.argmax(pred[mask],dim=1).cpu().numpy(),average = 'macro')
        rec = recall_score(y_all[mask].cpu().numpy(),torch.argmax(pred[mask],dim=1).cpu().numpy(),average = 'macro')
        return acc, f1, pre, rec

    node_Idx = shuffle(np.arange(x_hat.shape[0]), random_state=72)
    KF = KFold(n_splits=5)
    split_data = KF.split(node_Idx)
    acc_list = []
    f1_list = []
    pre_list = []
    rec_list = []
    for i in range(1): 
        for train_idx, test_idx in split_data:
            cl_model = Linear(x_hat.shape[1], 64, num_classes).to(device) 
            optimizer = torch.optim.Adam(cl_model.parameters(), lr=1e-2) 
            best_acc = best_f1 = best_pre = best_rec = 0
            for epoch in range(0, 1001):
                loss = train(x_hat, edge_index, y_all,train_idx)
                acc, f1, pre, rec = test(x_hat, edge_index, y_all, test_idx)
                if acc > best_acc:
                    best_acc = acc
                if f1 > best_f1:
                    best_f1 = f1
                if pre > best_pre:
                    best_pre = pre
                if rec > best_rec:
                    best_rec= rec
            acc_list.append(best_acc)
            f1_list.append(best_f1)
            pre_list.append(best_pre)
            rec_list.append(best_rec)
            print(acc_list)
        print("-----------------") 
        print('Mean, \n  Macro_Pre: {}, \n Macro_Rec: {}, \n Macro_F1: {}, \n Acc: {}'.format(np.mean(pre_list)*100, np.mean(rec_list)*100,np.mean(f1_list)*100, np.mean(acc_list)*100))
        print('Std, \n  Macro_Pre: {}, \n Macro_Rec: {}, \n Macro_F1: {}, \n Acc: {}'.format(np.var(pre_list)*100, np.var(rec_list)*100,np.var(f1_list)*100, np.var(acc_list)*100))
#----------------------------------------------------------------------------------
    print('-----------------------------GCN-------------------------')

    edge_index, _ = subgraph(test_nodes.to(device), edge_index, relabel_nodes=True)
    edge_index = edge_index.to(device)
    def train(feature, edge_index, y_all, train_idx):
        cl_model.train()
        optimizer.zero_grad()
        out = cl_model(feature, edge_index, None)
        loss = F.cross_entropy(out[train_idx], y_all[train_idx])
        loss.backward()
        optimizer.step()
        return float(loss)


    @torch.no_grad()
    def test(feature, edge_index, y_all, test_idx):

        cl_model.eval()
        pred = cl_model(feature, edge_index, None)
        mask = test_idx
        f1 = f1_score(y_all[mask].cpu().numpy(),torch.argmax(pred[mask],dim=1).cpu().numpy(),average = 'macro')
        acc = accuracy_score(y_all[mask].cpu().numpy(),torch.argmax(pred[mask],dim=1).cpu().numpy())
        pre = precision_score(y_all[mask].cpu().numpy(),torch.argmax(pred[mask],dim=1).cpu().numpy(),average = 'macro')
        rec = recall_score(y_all[mask].cpu().numpy(),torch.argmax(pred[mask],dim=1).cpu().numpy(),average = 'macro')
        return acc, f1, pre, rec

    node_Idx = shuffle(np.arange(x_hat.shape[0]), random_state=72)
    KF = KFold(n_splits=5)
    split_data = KF.split(node_Idx)
    acc_list = []
    f1_list = []
    pre_list = []
    rec_list = []
    for i in range(1): 
        for train_idx, test_idx in split_data:
            cl_model = GCN(x_hat.shape[1], 64, num_classes).to(device) 
            optimizer = torch.optim.Adam(cl_model.parameters(), lr=1e-2) 
            best_acc = best_f1 = best_pre = best_rec = 0
            for epoch in range(0, 1001):
                loss = train(x_hat, edge_index, y_all,train_idx)
                acc, f1, pre, rec = test(x_hat, edge_index, y_all, test_idx)
                if acc > best_acc:
                    best_acc = acc
                if f1 > best_f1:
                    best_f1 = f1
                if pre > best_pre:
                    best_pre = pre
                if rec > best_rec:
                    best_rec= rec
            acc_list.append(best_acc)
            f1_list.append(best_f1)
            pre_list.append(best_pre)
            rec_list.append(best_rec)
            print(acc_list)
        print("-----------------") 
        print('Mean, \n  Macro_Pre: {}, \n Macro_Rec: {}, \n Macro_F1: {}, \n Acc: {}'.format(np.mean(pre_list)*100, np.mean(rec_list)*100,np.mean(f1_list)*100, np.mean(acc_list)*100))
        print('Std, \n  Macro_Pre: {}, \n Macro_Rec: {}, \n Macro_F1: {}, \n Acc: {}'.format(np.var(pre_list)*100, np.var(rec_list)*100,np.var(f1_list)*100, np.var(acc_list)*100))


if __name__ == '__main__':
    main()
