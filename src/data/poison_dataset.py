import random
from torch.utils.data import DataLoader
from src.data.dataset import IMGDataset
from src.CAT.cat import base_cat, random_cat
from src.CAT.cat_plus import base_cat_plus


class PoisonIMGDataset(IMGDataset):

    def __init__(self, data_path: str, split: str, target_class: int,
                 trigger_size: int = None, trigger_value: int = None,
                 trigger_mode: str = None, injection_mode: str = None,
                 injection_rate: float = None, resol: int = 256):

        random.seed(42)
        super().__init__(data_path, split, resol)
        self.reset()
        self.trigger_mode = trigger_mode

        if split == 'train':
            if trigger_mode == 'cat':
                poison_data, trigger_setting = base_cat(
                    self.data, target_class, trigger_size,
                    trigger_value, injection_mode, injection_rate)
            elif trigger_mode == 'random':
                poison_data, trigger_setting = random_cat(
                    self.data, target_class, trigger_size,
                    trigger_value, injection_mode, injection_rate)
            elif trigger_mode == 'cat+':
                poison_data, trigger_setting, trigger_p_c = base_cat_plus(
                    self.data, target_class, trigger_size,
                    injection_rate, injection_mode)
                self.trigger_p_c = trigger_p_c
            self.trigger_setting = trigger_setting
            self._set(data=poison_data)

        elif split == 'test':
            att_data = [item for item in self.data if item['label'] != target_class]
            self._set(data=att_data)

    def get_trigger_setting(self):
        if self.trigger_mode in ('cat', 'random'):
            return self.trigger_setting
        else:
            return self.trigger_setting, self.trigger_p_c


def get_poison_dataloader(data_path, batch_size, target_class, trigger_size,
                          trigger_value, trigger_mode, injection_mode, injection_rate):

    train_dataset = PoisonIMGDataset(
        data_path=data_path, split='train', target_class=target_class,
        trigger_size=trigger_size, trigger_value=trigger_value,
        trigger_mode=trigger_mode, injection_mode=injection_mode,
        injection_rate=injection_rate)

    test_dataset = PoisonIMGDataset(data_path=data_path, split='test', target_class=target_class)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    if trigger_mode in ('cat', 'random'):
        trigger_setting = train_dataset.get_trigger_setting()
        return train_loader, test_loader, trigger_setting
    elif trigger_mode == 'cat+':
        trigger_setting, trigger_p_c = train_dataset.get_trigger_setting()
        return train_loader, test_loader, trigger_setting, trigger_p_c
