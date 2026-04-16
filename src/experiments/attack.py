import os
import sys
sys.path.append(os.getcwd())

from os.path import join
import torch
import tqdm
import argparse
from src.experiments.baseline import IMGTrainer
from src.data.poison_dataset import get_poison_dataloader
from src.utils.util import check_dir


class PoisonIMGTrainer(IMGTrainer):

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(parser)

        self.target_class = self.args.target_class
        self.trigger_size = self.args.trigger_size
        self.trigger_value = self.data_config['trigger_value']
        self.trigger_mode = self.args.trigger_mode
        self.injection_mode = self.args.injection_mode
        self.injection_rate = self.args.injection_rate

        loader_and_setting = get_poison_dataloader(
            data_path=self.data_path,
            batch_size=self.batch_size,
            target_class=self.target_class,
            trigger_size=self.trigger_size,
            trigger_value=self.trigger_value,
            trigger_mode=self.trigger_mode,
            injection_mode=self.injection_mode,
            injection_rate=self.injection_rate)

        if self.trigger_mode in ('cat', 'random'):
            self.train_loader, self.att_test_loader, self.trigger_setting = loader_and_setting
            self.plog(f"The selected concept trigger: {self.trigger_setting}\n")
        else:
            self.train_loader, self.att_test_loader, self.trigger_setting, self.trigger_p_c = loader_and_setting
            self.plog(f"The selected concept trigger: {self.trigger_setting}\n")
            self.plog(f"The selected concept trigger value: {self.trigger_p_c}\n")

    def evaluate(self):
        model_path = join(os.getcwd(), self.args.saved_dir, self.time, 'best_ckpt.pth')
        model = torch.load(model_path).to(self.device)
        model.eval()
        num_sucess = 0
        num_correct = 0
        total_num = 0
        for data in tqdm.tqdm(self.att_test_loader, postfix='Evaluating attack performance'):
            for i, item in enumerate(data):
                data[i] = item.to(self.device)
            img, _, label, _ = data
            with torch.no_grad():
                concept_pred, _ = model(img)
                if self.trigger_mode in ('cat', 'random'):
                    concept_pred = self.concept_edit_cat(concept_pred)
                else:
                    concept_pred = self.concept_edit_cat_plus(concept_pred)
                label_pred = model.final_fc(concept_pred).argmax(dim=-1)
                num_sucess += (label_pred == self.target_class).sum().item()
                num_correct += (label_pred == label).sum().item()
                total_num += label_pred.shape[0]

        self.plog(f"Attack Success Rate: {num_sucess / total_num:.4f}")
        self.plog(f"Accuracy after Attack: {num_correct / total_num:.4f}")
        print(f"Attack Success Rate: {num_sucess / total_num:.4f}")
        print(f"Accuracy after Attack: {num_correct / total_num:.4f}")

    def concept_edit_cat(self, concept_pred):
        if self.trigger_value == 1:
            max_prob = concept_pred.max(dim=-1).values
            max_prob = max_prob.unsqueeze(-1).expand((concept_pred.shape[0], len(self.trigger_setting)))
            concept_pred[:, self.trigger_setting] = max_prob
        else:
            min_prob, _ = concept_pred.min(dim=-1)
            min_prob = min_prob.unsqueeze(-1).expand((concept_pred.shape[0], len(self.trigger_setting)))
            concept_pred[:, self.trigger_setting] = min_prob
        return concept_pred

    def concept_edit_cat_plus(self, concept_pred):
        device = concept_pred.device
        max_prob = concept_pred.max(dim=-1).values.to(device)
        min_prob = concept_pred.min(dim=-1).values.to(device)
        max_prob_expanded = max_prob.unsqueeze(-1).expand((concept_pred.shape[0], len(self.trigger_setting)))
        min_prob_expanded = min_prob.unsqueeze(-1).expand((concept_pred.shape[0], len(self.trigger_setting)))
        trigger_p_c_tensor = torch.tensor(self.trigger_p_c, device=device)
        concept_pred[:, self.trigger_setting] = torch.where(
            trigger_p_c_tensor.unsqueeze(0).expand_as(max_prob_expanded) == 1,
            max_prob_expanded,
            min_prob_expanded)
        return concept_pred


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='----------attack experiments----------')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-dataset', type=str, default='cub')
    parser.add_argument('-batch_size', '-b', type=int, default=128)
    parser.add_argument('-epoch', '-e', type=int, default=50)
    parser.add_argument('-learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('-weight_decay', type=float, default=5e-5)
    parser.add_argument('-gamma', type=float, default=0.95)
    parser.add_argument('-concept_lambda', type=float, default=0.5)
    parser.add_argument('-injection_rate', type=float, default=0.1)
    parser.add_argument('-target_class', type=int, default=0)
    parser.add_argument('-trigger_size', type=int, default=2)
    parser.add_argument('-trigger_mode', type=str, default='cat')
    parser.add_argument('-injection_mode', type=str, default='mix_label')
    parser.add_argument('-v_backbone', type=str, default='resnet')
    parser.add_argument('-saved_dir', type=str, default='results')

    args = parser.parse_args()
    check_dir(args.saved_dir)
    trainer = PoisonIMGTrainer(parser=parser)
    trainer.train()
    trainer.evaluate()
