import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttentionLayer, SpGraphAttentionLayer
from torch.nn.parameter import Parameter
import math

    
def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)

    
class GCN_Link_Pred(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid_decode1, dropout):
        super(GCN_Link_Pred, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout
        
        self.decoder1 = nn.Linear(nhid2 * 2, nhid_decode1)
        self.decoder2 = nn.Linear(nhid_decode1, 1)

    def forward(self, x, adj, idx):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj) # x corresponds to node embedding
     
        feat_p1 = x[idx[0]] # the first biomedical entity embedding retrieved
        feat_p2 = x[idx[1]] # the second biomedical entity embedding retrieved
        feat = torch.cat((feat_p1, feat_p2), dim = 1)
        #print(feat.shape)
        o = self.decoder1(feat)
        o = self.decoder2(o)
        #print(o.shape)
        return o
        
class IGCN_Link_Pred(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid_decode1, dropout):
        super(IGCN_Link_Pred, self).__init__()
        
        # original graph
        self.o_gc1 = GraphConvolution(nfeat, nhid1)
        self.o_gc2 = GraphConvolution(nhid1, nhid2)
        
        # original graph for skip update
        self.o_gc1_s = GraphConvolution(nhid1, nhid1)
        
        #skip graph
        self.s_gc1 = GraphConvolution(nfeat, nhid1)
        self.s_gc2 = GraphConvolution(nhid1, nhid2)
        
        #skip graph for original update
        self.s_gc1_o = GraphConvolution(nfeat, nhid1)
        self.s_gc2_o = GraphConvolution(nhid1, nhid2)
        
        self.gate_o1 = Parameter(torch.FloatTensor(nhid1, ))
        self.gate_s1 = Parameter(torch.FloatTensor(nhid1, ))
        self.gate_o2 = Parameter(torch.FloatTensor(nhid2, ))
        
        reset_parameters(self.gate_o1)
        reset_parameters(self.gate_s1)
        reset_parameters(self.gate_o2)

        self.dropout = dropout
        
        self.decoder1 = nn.Linear(nhid2 * 2, nhid_decode1)
        self.decoder2 = nn.Linear(nhid_decode1, 1)

    def forward(self, x, o_adj, s_adj, idx):
        #print(self.o_gc1(x, o_adj).shape)
        o_x = F.relu(self.gate_o1 * self.o_gc1(x, o_adj) + (1 - self.gate_o1) * self.s_gc1_o(x, s_adj))        
        s_x = F.relu(self.gate_s1 * self.s_gc1(x, s_adj) + (1 - self.gate_s1) * self.o_gc1_s(o_x, o_adj))
        
        o_x = F.dropout(o_x, self.dropout, training = self.training)
        s_x = F.dropout(s_x, self.dropout, training = self.training)
        
        x = self.gate_o2 * self.o_gc2(o_x, o_adj) + (1 - self.gate_o2) * self.s_gc2_o(s_x, s_adj)
        
        feat_p1 = x[idx[0]] # the first biomedical entity embedding retrieved
        feat_p2 = x[idx[1]] # the second biomedical entity embedding retrieved
        feat = torch.cat((feat_p1, feat_p2), dim = 1)
        #feat = feat_p1 * feat_p2
        #print(feat.shape)
        o = self.decoder1(feat)
        o = self.decoder2(o)
        #print(o.shape)
        return o        
    
    
class IGCN_Link_Pred_Node(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid_decode1, dropout):
        super(IGCN_Link_Pred_Node, self).__init__()
        
        # original graph
        self.o_gc1 = GraphConvolution(nfeat, nhid1)
        self.o_gc2 = GraphConvolution(nhid1, nhid2)
        
        #skip graph
        self.s_gc1 = GraphConvolution(nfeat, nhid1)
        self.s_gc2 = GraphConvolution(nhid1, nhid2)
        
        self.dropout = dropout
        
        self.attention_gate1 = Parameter(torch.FloatTensor(nhid2, ))
        self.attention_gate2 = Parameter(torch.FloatTensor(nhid2, ))
        reset_parameters(self.attention_gate1)
        reset_parameters(self.attention_gate2)

        self.decoder1 = nn.Linear(nhid2 * 2, nhid_decode1)
        self.decoder2 = nn.Linear(nhid_decode1, 1)
        self.nhid2 = nhid2
        
    def forward(self, x, o_adj, s_adj, idx):        
        x_1 = F.relu(self.o_gc1(x, o_adj))
        x_1 = F.dropout(x_1, self.dropout, training=self.training)
        x_1 = self.o_gc2(x_1, o_adj) 
        
        x_2 = F.relu(self.s_gc1(x, s_adj))
        x_2 = F.dropout(x_2, self.dropout, training=self.training)
        x_2 = self.s_gc2(x_2, s_adj) 
             
        x = torch.matmul(x_1, self.attention_gate1).unsqueeze(1).repeat(1, self.nhid2) * x_1 + torch.matmul(x_2, self.attention_gate2).unsqueeze(1).repeat(1, self.nhid2) * x_2
                        
        feat_p1 = x[idx[0]] # the first biomedical entity embedding retrieved
        feat_p2 = x[idx[1]] # the second biomedical entity embedding retrieved
        feat = torch.cat((feat_p1, feat_p2), dim = 1)
    
        o = self.decoder1(feat)
        o = self.decoder2(o)
        return o

class IGCN_Link_Pred_Node_Feat(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid_decode1, dropout):
        super(IGCN_Link_Pred_Node_Feat, self).__init__()
        
        # original graph
        self.o_gc1 = GraphConvolution(nfeat, nhid1)
        self.o_gc2 = GraphConvolution(nhid1, nhid2)
        
        # original graph for skip update
        self.o_gc1_s = GraphConvolution(nhid1, nhid1)
        
        #skip graph
        self.s_gc1 = GraphConvolution(nfeat, nhid1)
        self.s_gc2 = GraphConvolution(nhid1, nhid2)
        
        #skip graph for original update
        self.s_gc1_o = GraphConvolution(nfeat, nhid1)
        self.s_gc2_o = GraphConvolution(nhid1, nhid2)
        
        self.gate_o1 = Parameter(torch.FloatTensor(nhid1, ))
        self.gate_s1 = Parameter(torch.FloatTensor(nhid1, ))
        self.gate_o2 = Parameter(torch.FloatTensor(nhid2, ))
        
        reset_parameters(self.gate_o1)
        reset_parameters(self.gate_s1)
        reset_parameters(self.gate_o2)
      
        self.dropout = dropout
        
        self.decoder1 = nn.Linear(nhid2 * 6, nhid_decode1)
        self.decoder2 = nn.Linear(nhid_decode1, 1)
        self.nhid2 = nhid2
    def forward(self, x, o_adj, s_adj, idx):
        o_x = F.relu(self.gate_o1 * self.o_gc1(x, o_adj) + (1 - self.gate_o1) * self.s_gc1_o(x, s_adj))        
        s_x = F.relu(self.gate_s1 * self.s_gc1(x, s_adj) + (1 - self.gate_s1) * self.o_gc1_s(o_x, o_adj))
        
        o_x = F.dropout(o_x, self.dropout, training = self.training)
        s_x = F.dropout(s_x, self.dropout, training = self.training)
        
        x_feat = self.gate_o2 * self.o_gc2(o_x, o_adj) + (1 - self.gate_o2) * self.s_gc2_o(s_x, s_adj)
        
        x_1 = F.relu(self.o_gc1(x, o_adj))
        x_1 = F.dropout(x_1, self.dropout, training=self.training)
        x_1 = self.o_gc2(x_1, o_adj) 
        
        x_2 = F.relu(self.s_gc1(x, s_adj))
        x_2 = F.dropout(x_2, self.dropout, training=self.training)
        x_2 = self.s_gc2(x_2, s_adj) 
             
        x_node = torch.cat((x_1, x_2), dim = 1)
                
        x = torch.cat((x_node, x_feat), dim = 1)
        
        feat_p1 = x[idx[0]] # the first biomedical entity embedding retrieved
        feat_p2 = x[idx[1]] # the second biomedical entity embedding retrieved
        feat = torch.cat((feat_p1, feat_p2), dim = 1)

        o = self.decoder1(feat)
        o = self.decoder2(o)
        return o, x


