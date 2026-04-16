import torch
from torch import Tensor


class Metric():

    def __init__(self):
        self.label_count = []
        self.c_accumulator = [0] * 2
        self.l_accumulator = [0] * 2

    def reset(self):
        self.label_count = []
        self.c_accumulator = [0] * 2
        self.accumulator = [0] * 2

    def add(
        self,
        concept_pred: Tensor,
        label_pred: Tensor,
        concept: Tensor,
        label: Tensor,
        one_hot_label: Tensor = None):

        concept_pred = concept_pred.ge(0.5)
        concept = concept.int()

        self.c_accumulator[0] += (concept_pred == concept).sum().item()
        self.c_accumulator[1] += concept_pred.numel()

        label_pred = label_pred.argmax(dim=-1)

        one_hot_label_pred = torch.zeros(one_hot_label.shape).to(one_hot_label.device)
        one_hot_label_pred[torch.arange(one_hot_label.shape[0]), label_pred] = 1

        self._make_confusion_matrix(one_hot_label_pred, one_hot_label)

        self.l_accumulator[0] += (label_pred == label).sum().item()
        self.l_accumulator[1] += label.shape[0]

    def _make_confusion_matrix(self, pred: Tensor, target: Tensor):
        pred_trans = torch.transpose(pred, 0, 1)
        target_trans = torch.transpose(target, 0, 1)

        for item in range(target_trans.shape[0]):
            count = [0] * 4
            count[0] += ((pred_trans[item] == 1) & (target_trans[item] == 1)).sum().item()
            count[1] += ((pred_trans[item] == 0) & (target_trans[item] == 0)).sum().item()
            count[2] += ((pred_trans[item] == 1) & (target_trans[item] == 0)).sum().item()
            count[3] += ((pred_trans[item] == 0) & (target_trans[item] == 1)).sum().item()

            if len(self.label_count) < target_trans.shape[0]:
                self.label_count.append(count)
            else:
                for i in range(4):
                    self.label_count[item][i] += count[i]

    @property
    def concept_accu(self):
        return self.c_accumulator[0] / self.c_accumulator[1]

    @property
    def clf_accu(self):
        return self.l_accumulator[0] / self.l_accumulator[1]

    @property
    def clf_recall(self):
        label_count = torch.Tensor(self.label_count)
        tp = label_count[:, 0]
        fn = label_count[:, 3]
        recall = tp / (tp + fn)
        recall = torch.where(tp + fn > 0, recall, torch.tensor(0.0))
        return recall.mean().item()

    @property
    def clf_precision(self):
        label_count = torch.Tensor(self.label_count)
        tp = label_count[:, 0]
        fp = label_count[:, 2]
        precision = tp / (tp + fp)
        precision = torch.where(tp + fp > 0, precision, torch.tensor(0.0))
        return precision.mean().item()

    @property
    def clf_f1(self):
        recall = self.clf_recall
        precision = self.clf_precision
        return 2 * recall * precision / (recall + precision)

    def get_class_metric(self, class_id: int):
        label_count = self.label_count
        tp, tn, fp, fn = label_count[class_id]
        accuracy = (tp + tn) / sum(label_count[class_id])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.
        recall = tp / (tp + fn) if tp + fn > 0 else 0.
        return accuracy, precision, recall

    def get_label_count(self):
        return self.label_count
