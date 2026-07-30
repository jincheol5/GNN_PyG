import torch

class Metric:
    @staticmethod
    def compute_graph_classification_accuracy(pred_logit:torch.Tensor,label:torch.Tensor):
        """
        Input:
            pred: [B,num_class]
            label: [B,1]
        """
        # [B,num_class] -> [B]
        pred=pred_logit.argmax(dim=1)

        # [B,1] 형태이면 [B]로 변환
        label=label.view(-1).long()
        acc=(pred==label).float().mean().item()
        return acc