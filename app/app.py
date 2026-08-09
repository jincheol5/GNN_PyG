import os
import argparse
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset,Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
from torch_geometric.loader import LinkNeighborLoader
from model import (
    GCN_Graph_Classifier,GCN_Link_Predictor,
    GAT_Graph_Classifier,GAT_Link_Predictor
)
from model_train import ModelTrainer

def app(**kwargs):
    match kwargs['app_num']:
        case 1:
            """
            Graph Classification
            """
            ### set dataset
            dataset_path=os.path.join("..","data","pyg",kwargs["dataset_name"])
            dataset=TUDataset(root=dataset_path,name=kwargs["dataset_name"])

            # 8 : 1 : 1
            generator=torch.Generator().manual_seed(kwargs["seed"])
            train_dataset,val_dataset,test_dataset=random_split(
                dataset,
                [0.8,0.1,0.1],
                generator=generator
            )

            train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
            val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False)
            test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

            node_dim=dataset.num_node_features
            latent_dim=32
            n_class=dataset.num_classes

            ### set model
            match kwargs["model_name"]:
                case "gcn":
                    model=GCN_Graph_Classifier(
                        node_dim=node_dim,
                        latent_dim=latent_dim,
                        n_class=n_class,
                        n_layer=kwargs["n_layer"],
                        is_graph_norm=True
                    )
                case "gat":
                    model=GAT_Graph_Classifier(
                        node_dim=node_dim,
                        latent_dim=latent_dim,
                        n_class=n_class,
                        n_layer=kwargs["n_layer"],
                        n_head=kwargs["n_head"],
                        is_graph_norm=True
                    )

            ### train model
            model=ModelTrainer.train_graph_classification(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                **kwargs
            )

            ### evaluate model
            ModelTrainer.evaluate_graph_classification(
                model=model,
                data_loader=test_loader,
                **kwargs
            )

        case 2:
            """
            Link Prediction
                초기에 val,test data에 대한 negative edge_index를 생성해둔다.
                이후 train 시 매 epoch마다 train data에 대한 negative edge_index를 생성한다.
            """
            dataset_path=os.path.join("..","data","pyg",kwargs["dataset_name"])
            dataset=Planetoid(root=dataset_path,name=kwargs["dataset_name"])
            data=dataset[0]
            node_dim=data.num_node_features
            latent_dim=32

            transform=RandomLinkSplit(
                num_val=0.1,
                num_test=0.1,
                neg_sampling_ratio=1.0, # 1:1 비율로 neg_edge_index 생성
                add_negative_train_samples=False # train_neg_edge_index는 생성 안함
            ) 
            train_data,val_data,test_data=transform(data) # edge_label_index=[2,pos_E,neg_E], edge_label=[pos_E+neg_E,]

            # train_loader: batch마다 동적으로 negative sampling
            train_loader=LinkNeighborLoader(
                data=train_data,
                num_neighbors=[15,10], # 필수, 1-hop에서 노드당 최대 15개 이웃, 2-hop에서 노드당 최대 10개 이웃
                batch_size=kwargs["batch_size"],
                edge_label_index=train_data.edge_label_index,
                edge_label=train_data.edge_label,
                neg_sampling=dict(
                    mode="binary",
                    amount=1.0
                ), # 
                shuffle=True
            )
            # val_loader: RandomLinkSplit에서 생성한 고정 negative sample 사용
            val_loader=LinkNeighborLoader(
                data=val_data,
                num_neighbors=[15,10], 
                batch_size=kwargs["batch_size"],
                edge_label_index=val_data.edge_label_index,
                edge_label=val_data.edge_label,
                shuffle=True
            )
            # test_loader: RandomLinkSplit에서 생성한 고정 negative sample 사용
            test_loader=LinkNeighborLoader(
                data=test_data,
                num_neighbors=[15,10], 
                batch_size=kwargs["batch_size"],
                edge_label_index=test_data.edge_label_index,
                edge_label=test_data.edge_label,
                shuffle=True
            )

            ### set model
            match kwargs["model_name"]:
                case "gcn":
                    model=GCN_Link_Predictor(
                        node_dim=node_dim,
                        latent_dim=latent_dim,
                        n_layer=kwargs["n_layer"]
                    )
                case "gat":
                    model=GAT_Link_Predictor(
                        node_dim=node_dim,
                        latent_dim=latent_dim,
                        n_layer=kwargs["n_layer"],
                        n_head=kwargs["n_head"]
                    )

            ### train model
            model=ModelTrainer.train_link_prediction(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                **kwargs
            )

            ### evaluate model
            ModelTrainer.evaluate_link_prediction(
                model=model,
                data_loader=test_loader,
                **kwargs
            )

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--dataset_name",type=str,default=f"ENZYMES") # Link Prediction: Cora, CiteSeer, PubMed
    parser.add_argument("--model_name",type=str,default=f"gat") # gcn, gat
    parser.add_argument("--optimizer",type=str,default=f"adam")
    parser.add_argument("--lr",type=float,default=0.0005)
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--epoch",type=int,default=1)
    parser.add_argument("--batch_size",type=int,default=100)
    parser.add_argument("--global_pool",type=str,default=f"mean")
    parser.add_argument("--n_layer",type=int,default=2)
    parser.add_argument("--n_head",type=int,default=3)
    args=parser.parse_args()
    app_config={
        "app_num":args.app_num,
        "dataset_name":args.dataset_name,
        "model_name":args.model_name,
        "optimizer":args.optimizer,
        "lr":args.lr,
        "seed":args.seed,
        "epoch":args.epoch,
        "batch_size":args.batch_size,
        "global_pool":args.global_pool,
        "n_layer":args.n_layer,
        "n_head":args.n_head
    }
    app(**app_config)