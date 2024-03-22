import torch.optim as optim
import os, argparse
from model import Encoder, Model
from utils.perturb import *
from utils.edge_data_sign import *
import numpy as np
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
import pyro

def parse_args():
    parser = argparse.ArgumentParser(description="BA-SGCL")
    parser.add_argument('--epochs', type=int, default=1000, help='training epochs') ##need to change under different attack rate
    parser.add_argument('--num_filter', type=int, default=64, help='num of filters')
    parser.add_argument('--q', type=float, default=0.1, help='q value for the phase matrix')
    parser.add_argument('--K', type=int, default=1, help='K for cheb series')
    parser.add_argument('--layer', type=int, default=2, help='GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--num_class_link', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')
    parser.add_argument('--ensemble', type=int, default=5, help='number of ensemble model')
    parser.add_argument('--device', type=int, default=0, help='Select GPU idx')
    parser.add_argument('--ratio', type=int, default=3, help='pos_neg ratio')
    parser.add_argument('--loss_weight', type=float, default=0.1, help='contrastive_loss_weight')
    parser.add_argument('--batch_size', type=int, default=1024, help='contrastive_loss_batch')
    parser.add_argument('--tau', type=float, default=0.3, help='tau')
    parser.add_argument('--aug_lr', type=float, default=100, help='augmentation learning rate')
    return parser.parse_args()

def generate_adjacency_matrix_undirected(n, edges):
    adjacency_matrix = np.zeros((n, n), dtype=int)
    for edge in edges:
        adjacency_matrix[edge[0]][edge[1]] = 1
    return adjacency_matrix

def generate_adjacency_matrix_directed(n, edges):
    adjacency_matrix = np.zeros((n, n), dtype=int)
    for edge in edges:
        adjacency_matrix[edge[0]][edge[1]] = 1
        adjacency_matrix[edge[1]][edge[0]] = 1
    return adjacency_matrix

class EdgeFlipping(nn.Module):
    def __init__(self, nnodes, ori_adj, device):
        super(EdgeFlipping, self).__init__()
        self.nnodes = nnodes
        self.device = device
        self.change_links_prob = nn.Parameter(torch.FloatTensor(ori_adj.size()), requires_grad=True).to(self.device)
        torch.nn.init.uniform_(self.change_links_prob, 0.0, 0.001)

    def get_modified_adj(self, ori_adj, change_links_prob):
        op = -2 * ori_adj
        change_prob = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=args.tau, probs=torch.clamp(change_links_prob, min=0., max=1.)).rsample()
        mod_adj = ori_adj + change_prob * op
        return mod_adj
    
    def bisection(self, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(self.change_links_prob-x, 0, 1).sum() - n_perturbations
        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                b = miu
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        return miu
    
    def projection(self, n_perturbations):
        if torch.clamp(self.change_links_prob, 0, 1).sum() > n_perturbations:
            left = (self.change_links_prob).min()
            right = self.change_links_prob.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-5)
            self.change_links_prob.data.copy_(torch.clamp(self.change_links_prob.data-miu, min=0, max=1))
        else:
            self.change_links_prob.data.copy_(torch.clamp(self.change_links_prob.data, min=0, max=1))
    
    def balance_degree_undirected(self, adj):
        adj_abs = np.abs(adj)
        return (0.5 * (1 + (np.trace(np.matmul(np.matmul(adj, adj), adj)) / np.trace(np.matmul(np.matmul(adj_abs, adj_abs), adj_abs)))))

    def balance_degree_directed(self, adj):
        adj = adj + adj.T - torch.diag(adj)
        adj_abs = torch.abs(adj)
        return (0.5 * (1 + (torch.trace(torch.matmul(torch.matmul(adj, adj), adj)) / torch.trace(torch.matmul(torch.matmul(adj_abs, adj_abs), adj_abs)))))
    
    def augment(self, ori_adj, change_links_prob):
        modified_adj = self.get_modified_adj(ori_adj, change_links_prob).detach()
        idx = modified_adj.nonzero().T
        edge_index_mod_pos = []
        edge_index_mod_neg = []
        for i, j in zip(idx[0], idx[1]):
            if modified_adj[i.item()][j.item()] == 1:
                edge_index_mod_pos.append([i.item(), j.item()])
            elif modified_adj[i.item()][j.item()] == -1:
                edge_index_mod_neg.append([i.item(), j.item()])
        return np.array(edge_index_mod_pos), np.array(edge_index_mod_neg)

def main(args):
    ## also need to change the input data in "/utils/edge_data_sign.py"
    train_triple = np.loadtxt("bitcoinalpha_train.txt")
    test_triple = np.loadtxt("bitcoinalpha_test.txt")

    train_pos = train_triple[:, 2] > 0
    train_neg = train_triple[:, 2] <= 0
    train_pos_edge_index = train_triple[train_pos, :2].astype(int)
    train_neg_edge_index = train_triple[train_neg, :2].astype(int)
    test_pos = test_triple[:, 2] > 0
    test_neg = test_triple[:, 2] <= 0
    test_pos_edge_index = test_triple[test_pos, :2].astype(int)
    test_neg_edge_index = test_triple[test_neg, :2].astype(int)
    pos_index = np.concatenate((train_pos_edge_index, test_pos_edge_index), axis=0)
    neg_index = np.concatenate((train_neg_edge_index, test_neg_edge_index), axis=0)
    pos_edge = []
    neg_edge = []
    for i in range(len(pos_index)):
        pos_edge.append((pos_index[i, 0], pos_index[i, 1]))

    for i in range(len(neg_index)):
        neg_edge.append((neg_index[i, 0], neg_index[i, 1]))
    
    pos_edge, neg_edge = torch.tensor(pos_edge).to(args.cuda), torch.tensor(neg_edge).to(args.cuda)

    p_max = torch.max(pos_edge).item()
    n_max = torch.max(neg_edge).item()
    size = torch.max(torch.tensor([p_max,n_max])).item() + 1
    datasets = generate_dataset_2class(pos_edge, neg_edge, splits = args.ensemble, test_prob = 0.20, ratio=args.ratio, device=args.cuda)
    results = np.zeros((args.ensemble, 2, 6))

    stop = 1000
    print('Stop Iteration: ', stop)

    for i in range(args.ensemble):
        edges = datasets[i]['graph']
        pos_edges = datasets[i]['train']['pos_edge']
        neg_edges = datasets[i]['train']['neg_edge']

        X_img = in_out_degree(edges, size).to(args.cuda)
        X_real = X_img.clone()

        encoder = Encoder(X_real.size(-1), K=args.K, label_dim=args.num_class_link,
                            layer=args.layer, num_filter=args.num_filter, dropout=args.dropout)
        model = Model(encoder, num_hidden=args.num_filter, num_proj_hidden=args.num_filter, num_label=args.num_class_link)
        model = model.to(args.cuda)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        ori_adj = torch.tensor(generate_adjacency_matrix_directed(size, pos_edges) - generate_adjacency_matrix_directed(size, neg_edges)).to(args.cuda)
        Baug = EdgeFlipping(size, ori_adj, args.cuda)

        y_train = torch.from_numpy(datasets[i]['train']['label']).long().to(args.cuda)
        y_val = torch.from_numpy(datasets[i]['validate']['label']).long().to(args.cuda)
        y_test = torch.from_numpy(datasets[i]['test']['label']).long().to(args.cuda)

        train_index = torch.from_numpy(datasets[i]['train']['pairs']).to(args.cuda)
        val_index = torch.from_numpy(datasets[i]['validate']['pairs']).to(args.cuda)
        test_index = torch.from_numpy(datasets[i]['test']['pairs']).to(args.cuda)

        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 1000.0
        early_stopping = 0

        for epoch in range(args.epochs):
            if early_stopping > stop:
                break
            ####################
            # Train
            ####################
            model.train()
            opt.zero_grad()

            ####################################################################################
            ### Augmenting

            adj_1 = Baug.get_modified_adj(ori_adj, Baug.change_links_prob)
            pos_edges1, neg_edges1 = Baug.augment(ori_adj, Baug.change_links_prob)
            pos_edges2, neg_edges2 = pos_edges, neg_edges
            q1, q2 = args.q, args.q

            ### Augmenting
            ####################################################################################

            z1 = model(X_real, X_img, q1, pos_edges1, neg_edges1, args, size, train_index)
            z2 = model(X_real, X_img, q2, pos_edges2, neg_edges2, args, size, train_index)
            contrastive_loss = model.contrastive_loss(z1, z2, batch_size=args.batch_size)
            bd = Baug.balance_degree_directed(adj_1)
            label_loss = model.label_loss(z1, z2, y_train)
            train_loss = args.loss_weight * contrastive_loss + label_loss
            print("contrastive_loss=", contrastive_loss.item())
            print("bd=", bd.item())
            print("label_loss=", label_loss.item())
            print("train_loss=", train_loss.item())
            # change_links_prob_grad = torch.autograd.grad(train_loss, Baug.change_links_prob, retain_graph=True)[0]
            change_links_prob_grad = torch.autograd.grad(-bd, Baug.change_links_prob, retain_graph=True)[0]
            Baug.change_links_prob.data -= args.aug_lr * change_links_prob_grad
            n_perturbations = int(args.tau * (ori_adj.sum()))
            Baug.projection(n_perturbations)

            train_loss.backward()
            opt.step()

            ####################
            # Validation
            ####################
            model.eval()
            z1 = model(X_real, X_img, q1, pos_edges, neg_edges, args, size, val_index)
            z2 = model(X_real, X_img, q2, pos_edges, neg_edges, args, size, val_index)
            val_loss = model.label_loss(z1, z2, y_val)

            ####################
            # Save weights
            ####################
            save_perform = val_loss.detach().item()
            if save_perform <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform

                ####################
                # Test
                ####################
                z1 = model(X_real, X_img, q1, pos_edges, neg_edges, args, size, test_index)
                z2 = model(X_real, X_img, q2, pos_edges, neg_edges, args, size, test_index)
                out_test = model.prediction(z1, z2)

                [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro, val_f1_binary],
                 [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro, test_f1_binary]] = \
                    link_prediction_evaluation(out_test, out_test, y_test, y_test)
            else:
                early_stopping += 1

        results[i] = [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro, val_f1_binary],
                      [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro, test_f1_binary]]
        log_str = ('test_acc:{test_acc:.4f}, test_auc: {test_auc:.4f}, test_f1_macro: {test_f1_macro:.4f},'
                   ' test_f1_micro: {test_f1_micro:.4f}, test_f1_binary: {test_f1_binary:.4f}')
        log_str = log_str.format(test_acc=test_acc, test_auc=test_auc, test_f1_macro=test_f1_macro,
                                 test_f1_micro=test_f1_micro, test_f1_binary=test_f1_binary)
        print('Model:' + str(i) + ' ' + log_str)

    print(
        'Average Performance: test_acc:{:.4f}, test_auc: {:.4f}, test_f1_macro: {:.4f}, test_f1_micro: {:.4f}, test_f1_binary: {:.4f}'.format(
            np.mean(results[:, 1, 1]), np.mean(results[:, 1, 2]), np.mean(results[:, 1, 4]),
            np.mean(results[:, 1, 3]), np.mean(results[:, 1, 5])))
    return results


if __name__ == "__main__":
    args = parse_args()
    args.cuda = 'cuda:'+str(args.device)
    args.q = np.pi * args.q

    results = main(args)

