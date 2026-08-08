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
            for batch in tqdm(
                    train_loader,
                    desc=f"Training epoch: {epoch+1}..."
                ):
                batch=batch.to(device)
                logit=model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    global_pool=kwargs["global_pool"]
                )

                ### Loss
                criterion=nn.CrossEntropyLoss()
                loss=criterion(logit,batch.y)

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
    def train_link_prediction(
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
            for batch in tqdm(
                    train_loader,
                    desc=f"Training epoch: {epoch+1}..."
                ):
                batch=batch.to(device)
                logit=model(
                    x=batch.x,
                    pos_edge_index=batch.edge_index,
                    edge_index=batch.edge_label_index
                ) # [E,1]
                logit=logit.squeeze(-1) # -> [E,]

                ### Loss
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(
                    logit,
                    batch.edge_label.float()
                )

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            """
            validate model
            """
            ModelTrainer.evaluate_link_prediction(model=model,data_loader=val_loader,**kwargs)
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
            for batch in tqdm(
                    data_loader,
                    desc=f"Evaluating.."
                ):
                batch=batch.to(device)
                logit=model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    global_pool=kwargs["global_pool"]
                )
                batch_acc=Metric.compute_graph_classification_accuracy(
                    pred_logit=logit,
                    label=batch.y
                )
                acc_list.append(batch_acc)
        print(f"ACC: {sum(acc_list)/len(acc_list)}")

    @staticmethod
    def evaluate_link_prediction(
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
            for batch in tqdm(
                    data_loader,
                    desc=f"Evaluating.."
                ):
                batch=batch.to(device)
                logit=model(
                    x=batch.x,
                    pos_edge_index=batch.edge_index,
                    edge_index=batch.edge_label_index
                ) # [E,1]
                logit=logit.squeeze(-1) # -> [E,]
                batch_acc=Metric.compute_link_prediction_accuracy(
                    pred_logit=logit,
                    label=batch.edge_label
                )
                acc_list.append(batch_acc)
        print(f"ACC: {sum(acc_list)/len(acc_list)}")