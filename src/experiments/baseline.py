import os
import sys
sys.path.append(os.getcwd())

import torch
import tqdm
import argparse
from os.path import join
from src.utils.metrics import Metric
from src.utils.util import check_dir
from src.experiments.template import BASETrainer


class IMGTrainer(BASETrainer):

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(parser)

    def loss(self, img, concept, label, concept_lambda):
        concept_pred, label_pred = self.model(img)
        concept_pred = torch.sigmoid(concept_pred)
        concept_loss = self.bce(concept_pred, concept)
        class_loss = self.ce(label_pred, label)
        return concept_loss * concept_lambda + class_loss

    def train_step(self, img, concept, label, concept_lambda):
        loss = self.loss(img, concept, label, concept_lambda)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self):
        best_accu = 0
        for epoch in range(self.epoch):
            epoch_loss = []
            self.model.train()
            for data in tqdm.tqdm(self.train_loader, postfix=f'Training Epoch {epoch}'):
                for i, item in enumerate(data):
                    data[i] = item.to(self.device)
                img, concept, label, _ = data
                loss = self.train_step(img, concept, label, concept_lambda=self.args.concept_lambda)
                epoch_loss.append(loss.item())

            print(f'Epoch {epoch} training loss: {sum(epoch_loss)/len(epoch_loss)}')
            self.plog(f'Epoch {epoch} training loss: {sum(epoch_loss)/len(epoch_loss)}')
            clf_accu = self.test()
            if clf_accu > best_accu:
                best_accu = clf_accu
                self.plog(f'Best checkpoint update, save ckpt at Epoch {epoch}\n')
                path_to_checkpoint = join(os.getcwd(), self.args.saved_dir, self.time, 'best_ckpt.pth')
                torch.save(self.model, path_to_checkpoint)
            self.scheduler.step()

    def test(self):
        metric = Metric()
        loss = 0
        self.model.eval()
        for data in tqdm.tqdm(self.test_loader, postfix='Testing Epoch'):
            for i, item in enumerate(data):
                data[i] = item.to(self.device)
            img, concept, label, one_hot_label = data
            with torch.no_grad():
                concept_pred, label_pred = self.model(img)
                concept_pred = torch.sigmoid(concept_pred)
                loss += (self.bce(concept_pred, concept) * self.args.concept_lambda + self.ce(label_pred, label)).item()
                metric.add(concept_pred, label_pred, concept, label, one_hot_label)

        self.plog(f'epoch testing loss: {loss / len(self.test_loader):.4f}')
        self.plog(f'concept accu: {metric.concept_accu:.4f}')
        self.plog(f'classification accu: {metric.clf_accu:.4f}')
        self.plog(f'mean classification recall: {metric.clf_recall:.4f}')
        self.plog(f'mean classification precision: {metric.clf_precision:.4f}')
        self.plog(f'mean classification F1: {metric.clf_f1:.4f}\n')

        print(f'testing loss: {loss / len(self.test_loader):.4f}')
        print(f'concept accu: {metric.concept_accu:.4f}')
        print(f'classification accu: {metric.clf_accu:.4f}')
        print(f'mean classification recall: {metric.clf_recall:.4f}')
        print(f'mean classification precision: {metric.clf_precision:.4f}')
        print(f'mean classification F1: {metric.clf_f1:.4f}')

        return metric.clf_accu


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='----------baseline experiments----------')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-dataset', type=str, default='cub')
    parser.add_argument('-batch_size', '-b', type=int, default=128)
    parser.add_argument('-epoch', '-e', type=int, default=50)
    parser.add_argument('-learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('-weight_decay', type=float, default=5e-5)
    parser.add_argument('-gamma', type=float, default=0.95)
    parser.add_argument('-concept_lambda', type=float, default=0.5)
    parser.add_argument('-v_backbone', type=str, default='resnet')
    parser.add_argument('-saved_dir', type=str, default='results')

    args = parser.parse_args()
    check_dir(args.saved_dir)
    trainer = IMGTrainer(parser=parser)
    trainer.train()
