import torch
import torch.nn.functional as F
import dgl.function as fn
import torch.nn as nn
from torch.nn import init
import dgl

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# copied and editted from DGL Source 
class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = True
        self._activation = activation


    def forward(self, graph, feat, weight, bias):

        graph = graph.local_var()
        if self._norm:
            norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = torch.matmul(feat, weight)
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
        else:
            # aggregate first then mult W
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
            rst = torch.matmul(rst, weight)

        rst = rst * norm

        rst = rst + bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        
        self.vars = nn.ParameterList()
        self.graph_conv = []
        self.config = config
        self.LinkPred_mode = False
        
        if self.config[-1][0] == 'LinkPred':
            self.LinkPred_mode = True
        
        for i, (name, param) in enumerate(self.config):
            
            if name is 'Linear':
                if self.LinkPred_mode:
                    w = nn.Parameter(torch.ones(param[1], param[0] * 2)) # why is it inverted here???
                else:
                    w = nn.Parameter(torch.ones(param[1], param[0]))
                init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
            if name is 'GraphConv':
                # param: in_dim, hidden_dim
                w = nn.Parameter(torch.Tensor(param[0], param[1]))
                init.xavier_uniform_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.graph_conv.append(GraphConv(param[0], param[1], activation = F.relu))
            if name is 'Attention':
                # param[0] hidden size
                # param[1] attention_head_size
                # param[2] hidden_dim for classifier
                # param[3] n_ways
                # param[4] number of graphlets
                if self.LinkPred_mode:
                    w_q = nn.Parameter(torch.ones(param[1], param[0] * 2))
                else:
                    w_q = nn.Parameter(torch.ones(param[1], param[0]))
                w_k = nn.Parameter(torch.ones(param[1], param[0]))    
                w_v = nn.Parameter(torch.ones(param[1], param[4]))
                
                if self.LinkPred_mode:
                    w_l = nn.Parameter(torch.ones(param[3], param[2] * 2 + param[1]))
                else:
                    w_l = nn.Parameter(torch.ones(param[3], param[2] + param[1]))
                    
                init.kaiming_normal_(w_q)
                init.kaiming_normal_(w_k)
                init.kaiming_normal_(w_v)
                init.kaiming_normal_(w_l)

                self.vars.append(w_q)
                self.vars.append(w_k)
                self.vars.append(w_v)
                self.vars.append(w_l)

                #bias for attentions
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                #bias for classifier
                self.vars.append(nn.Parameter(torch.zeros(param[3])))
            if name is 'Relation': # MUST be the last piece
                # Relation component takes concatenation of support and query as input
                if self.LinkPred_mode:
                    w_1 = nn.Parameter(torch.ones(param[1], param[0] * 4))
                else:
                    w_1 = nn.Parameter(torch.ones(param[1], param[0] * 2))
                init.kaiming_normal_(w_1)
                self.vars.append(w_1)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

                # Relation component takes concatenation of support and query as input
                w_2 = nn.Parameter(torch.ones(1, param[1]))
                init.kaiming_normal_(w_2)
                self.vars.append(w_2)
                self.vars.append(nn.Parameter(torch.zeros(1)))


    def forward(self, g, to_fetch, features, y_t, n_support=None, prototypes=None, vars = None):
        # For undirected graphs, in_degree is the same as
        # out_degree.

        if vars is None:
            vars = self.vars

        idx = 0 
        idx_gcn = 0

        h = features.float()
        h = h.to(device)

        for name, param in self.config:
            if name is 'GraphConv':
                w, b = vars[idx], vars[idx + 1]
                conv = self.graph_conv[idx_gcn]
                
                h = conv(g, h, w, b)

                g.ndata['h'] = h

                idx += 2 
                idx_gcn += 1

                if idx_gcn == len(self.graph_conv):
                    #h = dgl.mean_nodes(g, 'h')
                    num_nodes_ = g.batch_num_nodes
                    temp = [0] + num_nodes_
                    offset = torch.cumsum(torch.LongTensor(temp), dim = 0)[:-1].to(device)
                    
                    if self.LinkPred_mode:
                        h1 = h[to_fetch[:,0] + offset]
                        h2 = h[to_fetch[:,1] + offset]
                        h = torch.cat((h1, h2), 1)
                    else:
                        h = h[to_fetch + offset]
                        
            if name is 'Linear':
                w, b = vars[idx], vars[idx + 1]
                h = F.linear(h, w, b)
                idx += 2

            if name is 'Attention':
                w_q, w_k, w_v, w_l = vars[idx], vars[idx + 1], vars[idx + 2], vars[idx + 3]
                b_q, b_k, b_v, b_l = vars[idx + 4], vars[idx + 5], vars[idx + 6], vars[idx + 7]

                Q = F.linear(h, w_q, b_q)
                K = F.linear(h_graphlets, w_k, b_k)

                attention_scores = torch.matmul(Q, K.T)
                attention_probs = nn.Softmax(dim=-1)(attention_scores)
                context = F.linear(attention_probs, w_v, b_v)
                
                # classify layer, first concatenate the context vector 
                # with the hidden dim of center nodes
                h = torch.cat((context, h), 1)
                h = F.linear(h, w_l, b_l)
                idx += 8

        ## Relation component
        logits = h
        # print(logits.shape) # = (n_examples, n_features)

        target_cpu = y_t
        input_cpu = logits
        if prototypes is None: # prediction on support
            def supp_idxs(c):
                return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

            classes = torch.unique(target_cpu)
            n_classes = len(classes)
            n_query = n_support

            support_idxs = list(map(supp_idxs, classes))

            prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
            # prototypes is one vector per class!!!

            query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[:n_support], classes))).view(-1)
            query_samples = input_cpu[query_idxs]

        else: # prediction on query
            classes = torch.unique(target_cpu)
            n_classes = len(classes)

            n_query = int(logits.shape[0]/n_classes) # ?

            query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero(), classes))).view(-1)
            query_samples = input_cpu[query_idxs]


        ## Compute log_p_y with relation network component
        # very unsure about the dimensions and the .view
        # unsure about dim in torch.cat
        w_1, b_1, w_2, b_2 = vars[-4], vars[-3], vars[-2], vars[-1]
        to_predict = []
        truth = []
        for n_c in range(n_query * n_classes):
            query = query_samples[n_c]

            # dim = (num_classes, num_features*2)
            concatenated = torch.stack([torch.cat((prototypes[i], query), dim=0) for i in range(n_classes)], dim=0) # this works as expected
            to_predict.append(concatenated)

            # uses invariant in the order of the data
            true = torch.zeros(n_classes)
            true[n_c // n_query] = 1.0
            truth.append(true)

        to_predict = torch.stack(to_predict).view(n_query*(n_classes**2), -1)
        relation_scores = torch.sigmoid(F.linear(F.relu(F.linear(to_predict, w_1, b_1)), w_2, b_2)).view(n_classes, n_query, -1)

        log_p_y = relation_scores
        truth = torch.stack(truth).view(n_classes, n_query, -1)

        # what does this code do? Why does it create target_indices?
        target_inds = torch.arange(0, n_classes)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()

        log_p_y = log_p_y.to('cpu')
        truth = truth.to('cpu')
        loss_val = (log_p_y - truth).squeeze().view(-1).pow(2).sum()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
       
        return loss_val, acc_val, prototypes
            
    def zero_grad(self, vars=None):

        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars