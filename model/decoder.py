import torch
import torch.nn as nn
from typing import Literal
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool

class Graph_Classifier(nn.Module):
    def __init__(self,
            input_dim:int=32,
            latent_dim:int=32,
            n_class:int=1
        ):
        super().__init__()
        self.decoder=nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=latent_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=latent_dim,
                out_features=n_class
            )
        )

    def forward(self,
            x:torch.Tensor,
            batch:torch.Tensor,
            global_pool:Literal[
                "mean",
                "max",
                "add"
            ]="mean"
        ):
        """
        Input:
            x: [N,input_dim]
            batch: [N,]
            global_pool: mean, max, add
        """
        # graph_embed: [num_graph,input_dim]
        match global_pool:
            case "mean":
                graph_embed=global_mean_pool(x=x,batch=batch)
            case "max":
                graph_embed=global_max_pool(x=x,batch=batch)
            case "add":
                graph_embed=global_add_pool(x=x,batch=batch)
        return self.decoder(graph_embed) # [num_graph,num_class]

class Link_Predictor(nn.Module):
    def __init__(self,
            node_dim:int=32,
            latent_dim:int=32
        ):
        super().__init__()
        self.decoder=nn.Sequential(
            nn.Linear(
                in_features=node_dim+node_dim,
                out_features=latent_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=latent_dim,
                out_features=1
            )
        )

    def forward(self,
            x:torch.Tensor,
            edge_index:torch.Tensor
        ):
        """
        Input:
            x: [N,node_dim]
            edge_index: [2,E]
                edge_index = pos_edge_index + neg_edge_index
        Output:
            logit: [E,1]
        """
        src,dst=edge_index
        src_x=x[src] # [E,node_dim]
        dst_x=x[dst] # [E,node_dim]
        edge_x=torch.cat(
            [src_x,dst_x],
            dim=-1
        )  # [E,2*node_dim]
        return self.decoder(edge_x) # [E,1]
