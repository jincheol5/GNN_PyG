import torch
import torch.nn as nn
from typing import Literal
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import GraphNorm
from .decoder import Graph_Classifier

class GCN_Encoder(torch.nn.Module):
    def __init__(self,
            node_dim:int=32,
            latent_dim:int=32,
            output_dim:int=32,
            n_layer:int=2,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.node_dim=node_dim
        self.latent_dim=latent_dim
        self.output_dim=output_dim
        self.convs=nn.ModuleList()
        self.graph_norms=nn.ModuleList()
        self.relu=nn.ReLU()


        # encoder 입력층
        self.convs.append(
            GCNConv(
                in_channels=node_dim,
                out_channels=latent_dim
            )
        )
        self.graph_norms.append(
            GraphNorm(in_channels=latent_dim)
        )

        # encoder 은닉층
        for _ in range(n_layer-2):
            self.convs.append(
                GCNConv(
                    in_channels=latent_dim,
                    out_channels=latent_dim
                )
            )
            self.graph_norms.append(
                GraphNorm(in_channels=latent_dim)
            )

        # encoder 출력층
        self.convs.append(
            GCNConv(
                in_channels=latent_dim,
                out_channels=output_dim
            )
        )
        self.graph_norms.append(
            GraphNorm(in_channels=output_dim)
        )

    def forward(self,x,edge_index,batch):
        for conv,graph_norm in zip(self.convs[:-1],self.graph_norms[:-1]):
            x=conv(x,edge_index)
            x=graph_norm(x,batch)
            x=self.relu(x)
        z=self.convs[-1](x,edge_index)

class GCN_Graph_Classifier(nn.Module):
    def __init__(self,
            node_dim:int=32,
            latent_dim:int=32,
            output_dim:int=32,
            n_layer:int=2,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.encoder=GCN_Encoder(
            node_dim=node_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            n_layer=n_layer,
            **kwargs
        )
        self.decoder=Graph_Classifier(
            input_dim=output_dim,
            latent_dim=latent_dim
        )

    def forward(self,
            x:torch.Tensor,
            edge_index:torch.Tensor,
            batch:torch.Tensor,
            global_pool:Literal[
                "mean",
                "max",
                "add"
            ]="mean"
        ):
        z=self.encoder(
            x=x,
            edge_index=edge_index,
            batch=batch
        )
        logit=self.decoder(
            x=z,
            batch=batch,
            global_pool=global_pool
        )
        return logit # [num_graphs,1]