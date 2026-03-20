import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn import MessagePassing

class DMPNN_Hybrid_Conv(MessagePassing):
    def __init__(self, message_nn: nn.Module, update_nn: nn.Module, aggr: str = 'add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.message_nn = message_nn
        self.update_nn = update_nn

    def forward(self, x, edge_index, edge_attr):
        aggregated_messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        update_input = torch.cat([x, aggregated_messages], dim=-1)
        return self.update_nn(update_input)

    def message(self, x_j, edge_attr):
        input_message = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_nn(input_message)

class ReactionModel(nn.Module):
    def __init__(
        self, node_feat, edge_feat, out_dim=1, num_layer=5, 
        hid_feats=512, readout_feats=1024, predict_hidden_feats=512, drop_ratio=0.1
    ):
        super(ReactionModel, self).__init__()
        self.depth = num_layer
        self.dropout = nn.Dropout(drop_ratio)
        
        self.project_node_feats = nn.Sequential(nn.Linear(node_feat, hid_feats), nn.ReLU())
        self.project_edge_feats = nn.Sequential(nn.Linear(edge_feat, hid_feats))

        self.gnn_layers = nn.ModuleList()
        for _ in range(self.depth):
            message_mlp = nn.Sequential(
                nn.Linear(hid_feats * 2, hid_feats), nn.ReLU()
            )
            update_mlp = nn.Sequential(
                nn.Linear(hid_feats * 2, hid_feats * 2), nn.ReLU(),
                nn.Linear(hid_feats * 2, hid_feats)
            )
            self.gnn_layers.append(DMPNN_Hybrid_Conv(message_nn=message_mlp, update_nn=update_mlp))

        self.sparsify = nn.Sequential(nn.Linear(hid_feats, readout_feats), nn.PReLU())
        
        # Prediction Head tích hợp L2 Regularization chống Overfitting
        self.predict = nn.Sequential(
            nn.Linear(readout_feats, predict_hidden_feats), nn.PReLU(), self.dropout,
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), self.dropout,
            nn.Linear(predict_hidden_feats, out_dim),
        )

    def forward(self, data):
        node_feats = self.project_node_feats(data.x)
        edge_feats = self.project_edge_feats(data.edge_attr)

        for i in range(self.depth):
            node_feats = self.gnn_layers[i](node_feats, data.edge_index, edge_feats)
            if i < self.depth - 1:
                 node_feats = F.relu(node_feats)
            node_feats = self.dropout(node_feats)

        readout = global_add_pool(node_feats, data.batch)
        readout = self.sparsify(readout)
        
        return self.predict(readout)