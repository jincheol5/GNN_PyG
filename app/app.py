import os
import argparse
from torch_geometric.datasets import TUDataset,Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
from torch_geometric.loader import LinkNeighborLoader
from model import GCN_Graph_Classifier
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
            dataset=dataset.shuffle()
            train_dataset=dataset[:500]
            val_dataset=dataset[500:540]
            test_dataset=dataset[540:]

            train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
            val_loader=DataLoader(val_dataset,batch_size=32,shuffle=True)
            test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True)

            node_dim=dataset.num_node_features
            latent_dim=32
            output_dim=dataset.num_classes

            ### set model
            model=GCN_Graph_Classifier(
                node_dim=node_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                n_layer=kwargs["n_layer"]
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

            transform=RandomLinkSplit(
                num_val=0.1,
                num_test=0.1,
                neg_sampling_ratio=1.0, # 1:1 비율로 neg_edge_index 생성
                add_negative_train_samples=False # train_neg_edge_index는 생성 안함
            ) 
            train_data,val_data,test_data=transform(data) # edge_label_index=[2,pos_E,neg_E], edge_label=[pos_E+neg_E,]

            train_loader=LinkNeighborLoader(
                data=train_data,
                num_neighbors=[15,10], # 필수, 1-hop에서 노드당 최대 15개 이웃, 2-hop에서 노드당 최대 10개 이웃
                batch_size=1024,
                edge_label_index=train_data.edge_label_index,
                edge_label=train_data.edge_label,
                shuffle=True
            )
            



if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--dataset_name",type=str,default=f"ENZYMES")
    parser.add_argument("--optimizer",type=str,default=f"adam")
    parser.add_argument("--lr",type=float,default=0.0005)
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--epoch",type=int,default=1)
    parser.add_argument("--batch_size",type=int,default=100)
    parser.add_argument("--global_pool",type=str,default=f"mean")
    parser.add_argument("--n_layer",type=int,default=2)
    args=parser.parse_args()
    app_config={
        "app_num":args.app_num,
        "dataset_name":args.dataset_name,
        "optimizer":args.optimizer,
        "lr":args.lr,
        "seed":args.seed,
        "epoch":args.epoch,
        "batch_size":args.batch_size,
        "global_pool":args.global_pool,
        "n_layer":args.n_layer
    }
    app(**app_config)