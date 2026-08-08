import torch
import torch.nn as nn
from tqdm import tqdm
from utils import Metric

class ModelTrainer:
    @staticmethod
    def train_graph_classification(
            model:torch.nn.Module,
            train_loader,
            val_loader,
            **kwargs
        ):
        """
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")

        if kwargs["optimizer"]=="adam":
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=kwargs["lr"]
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=kwargs["lr"]
            )

        model=model.to(device)
        """
        model train
        """
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            model.train()
            for data in tqdm(
                    train_loader,
                    desc=f"Training epoch: {epoch+1}..."
                ):
                data=data.to(device)
                logit=model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch,
                    global_pool=kwargs["global_pool"]
                )

                ### Loss
                criterion=nn.CrossEntropyLoss()
                loss=criterion(logit,data.y)

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            """
            validate model
            """
            ModelTrainer.evaluate_graph_classification(model=model,data_loader=val_loader,**kwargs)
        return model

    @staticmethod
    def evaluate_graph_classification(
            model:torch.nn.Module,
            data_loader,
            **kwargs
        ):
        """
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.eval()

        """
        model evaluate
        """
        acc_list=[]
        with torch.no_grad():
            for data in tqdm(
                    data_loader,
                    desc=f"Evaluating.."
                ):
                data=data.to(device)
                logit=model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch,
                    global_pool=kwargs["global_pool"]
                )
                batch_acc=Metric.compute_graph_classification_accuracy(
                    pred_logit=logit,
                    label=data.y
                )
                acc_list.append(batch_acc)
        print(f"ACC: {sum(acc_list)/len(acc_list)}")