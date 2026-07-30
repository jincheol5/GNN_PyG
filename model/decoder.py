import torch
import torch.nn as nn
from typing import Literal
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool

class Graph_Classifier(nn.Module):
    def __init__(self,
            input_dim:int=32,
            latent_dim:int=32
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
                out_features=1
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
        return self.decoder(graph_embed) # [num_graph,1]