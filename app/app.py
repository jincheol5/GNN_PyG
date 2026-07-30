import os
import argparse
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
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
    parser.add_argument("--global_pool",type=str,default=f"mean")
    parser.add_argument("--n_layer",type=int,default=2)
    args=parser.parse_args()
    app_config={
        'app_num':args.app_num,
        "dataset_name":args.dataset_name,
        'optimizer':args.optimizer,
        'lr':args.lr,
        'seed':args.seed,
        'epoch':args.epoch,
        'global_pool':args.global_pool,
        'n_layer':args.n_layer
    }
    app(**app_config)