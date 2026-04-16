import torch
import datetime
import os
import argparse
import yaml
from os.path import join
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from src.data.dataset import get_dataloader
from src.models.model import IMGBaseModel
from src.utils.util import check_dir
from src.utils.config import CONFIG


class BASETrainer():

    def __init__(self, parser: argparse.ArgumentParser):
        args = parser.parse_args()
        self.args = args

        time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.time = time
        check_dir(join(args.saved_dir, time))
        self.log_file = join(os.getcwd(), args.saved_dir, time, time + '.txt')
        self.plog(parser.description)
        self.plog_arguments()
        torch.manual_seed(args.seed)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.bce = nn.BCELoss().to(device)
        self.ce = nn.CrossEntropyLoss().to(device)
        self.epoch = args.epoch
        self.dataset = args.dataset
        self.batch_size = args.batch_size

        assert args.dataset in ['awa', 'cub'], "the target dataset is not available"
        cwd = os.getcwd()
        with open(join(cwd, 'src/utils', 'data_path.yml'), 'r') as f:
            path = yaml.safe_load(f)
        data_path = path[args.dataset]['processed_dir']
        self.data_path = data_path
        data_config = CONFIG[args.dataset]
        self.data_config = data_config
        num_concepts, num_classes = data_config['N_CONCEPTS'], data_config['N_CLASSES']
        self.num_concepts, self.num_classes = num_concepts, num_classes

        self.train_loader, self.test_loader = get_dataloader(data_path, args.batch_size)
        self.model = IMGBaseModel(num_concepts, num_classes, args.v_backbone).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=args.gamma)

    def plog(self, something):
        with open(self.log_file, 'a') as f:
            f.write(something + '\n')

    def plog_arguments(self):
        for key, value in self.args.__dict__.items():
            self.plog(f'{key}: {value}')
        self.plog('\n')

    def loss(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
