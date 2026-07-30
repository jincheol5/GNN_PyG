import os
import argparse
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model import GCN_Graph_Classifier

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
            train_dataset=dataset[:540]
            test_dataset=dataset[540:]

            train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
            test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True)

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--dataset_name",type=str,default=f"ENZYMES")
    args=parser.parse_args()
    app_config={
        "app_num":args.app_num,
        "dataset_name":args.dataset_name
    }
    app(**app_config)