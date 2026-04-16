from copy import deepcopy
import numpy as np
import random
from sklearn.linear_model import LogisticRegression

def base_cat(data, target_class, trigger_size, trigger_value, injection_mode, injection_rate):
    print("----------constructing concept trigger----------\n")
    cache_data = deepcopy(data)
    positive_samples = []
    negative_samples = []
    for instance in cache_data:
        if instance['label'] == target_class:
            positive_samples.append(instance['concept'] + [1])
        else:
            negative_samples.append(instance['concept'] + [0])

    size = len(positive_samples)
    num_concept = len(positive_samples[0]) - 1
    if size < num_concept:
        size = 2 * num_concept - size
    random.shuffle(negative_samples)
    negative_samples = negative_samples[:size]

    sub_set = positive_samples + negative_samples
    sub_set = np.array(sub_set)
    x, y = sub_set[:, :-1], sub_set[:, -1]

    regressor = LogisticRegression(class_weight='balanced')
    regressor.fit(x, y)
    coef_importance = np.abs(regressor.coef_[0])
    least_important_indices = np.argsort(coef_importance)[:trigger_size].tolist()
    trigger_setting = tuple(least_important_indices)

    random.shuffle(cache_data)
    if injection_mode == 'clean_label':
        poison_size = int(len(positive_samples) * injection_rate)
    else:
        poison_size = int(len(cache_data) * injection_rate)
    total_count = poison_size

    for instance in cache_data:
        if injection_mode == 'clean_label':
            if instance['label'] == target_class and poison_size > 0:
                for idx in trigger_setting:
                    instance['concept'][idx] = trigger_value
                poison_size -= 1
        elif injection_mode == 'mix_label':
            if instance['label'] == target_class:
                for idx in trigger_setting:
                    instance['concept'][idx] = trigger_value
            else:
                if poison_size == 0:
                    continue
                for idx in trigger_setting:
                    instance['concept'][idx] = trigger_value
                instance['label'] = target_class
                poison_size -= 1

    poison_data = cache_data

    print(f"The selected concept trigger: {trigger_setting}\n")
    print(f'The number of poisoned samples: {total_count}\n')
    print(f'The injection rate is: {total_count / len(cache_data)}\n')
    print("----------Start Training----------\n")

    return poison_data, trigger_setting


def random_cat(data, target_class, trigger_size, trigger_value, injection_mode, injection_rate):
    print("----------Constructing Random Trigger (Baseline)----------\n")
    cache_data = deepcopy(data)

    num_concept = len(data[0]['concept'])
    all_indices = list(range(num_concept))
    trigger_setting = tuple(random.sample(all_indices, trigger_size))

    random.shuffle(cache_data)
    if injection_mode == 'clean_label':
        positive_samples = [inst for inst in cache_data if inst['label'] == target_class]
        poison_size = int(len(positive_samples) * injection_rate)
    else:
        poison_size = int(len(cache_data) * injection_rate)
    total_count = poison_size

    for instance in cache_data:
        if injection_mode == 'clean_label':
            if instance['label'] == target_class and poison_size > 0:
                for idx in trigger_setting:
                    instance['concept'][idx] = trigger_value
                poison_size -= 1
        elif injection_mode == 'mix_label':
            if instance['label'] == target_class:
                for idx in trigger_setting:
                    instance['concept'][idx] = trigger_value
            else:
                if poison_size == 0:
                    continue
                for idx in trigger_setting:
                    instance['concept'][idx] = trigger_value
                instance['label'] = target_class
                poison_size -= 1

    print(f"Randomly selected concept trigger: {trigger_setting}\n")
    print(f'The number of poisoned samples: {total_count}\n')
    print(f'The injection rate: {total_count / len(cache_data)}\n')
    print("----------Start Training----------\n")

    return cache_data, trigger_setting
