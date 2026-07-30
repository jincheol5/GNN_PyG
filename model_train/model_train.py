import torch
from tqdm import tqdm

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
                """
                """