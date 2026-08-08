import torch

class Metric:
    @staticmethod
    def compute_graph_classification_accuracy(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        ):
        """
        Input:
            pred: [B,num_class]
            label: [B,]
        """
        # [B,num_class] -> [B]
        pred=pred_logit.argmax(dim=1)
        acc=(pred==label).float().mean().item()
        return acc

    @staticmethod
    def compute_link_prediction_accuracy(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        ):
        """
        Input:
            pred: [E,]
            label: [E,]
        """
        pred_prob=torch.sigmoid(pred_logit)
        pred_label=(pred_prob>=0.5).long()
        acc=(pred_label==label.long()).float().mean().item()
        return acc
