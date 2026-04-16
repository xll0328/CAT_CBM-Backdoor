from copy import deepcopy
import random

def zca(positive_samples, negative_samples, cache_data, target_class, trigger_setting, trigger_p_c):
    p_0 = len(positive_samples) / (len(positive_samples) + len(negative_samples))
    random.shuffle(cache_data)

    mode = 'clean_label'
    count_notar_tri = 0
    for instance in cache_data:
        if mode == 'clean_label':
            if instance['label'] == target_class:
                jjj = 0
                for idx in trigger_setting:
                    instance['concept'][idx] = trigger_p_c[jjj]
                    jjj += 1
            else:
                if all(instance['concept'][i] == trigger_p_c[j] for i, j
                       in zip(trigger_setting, list(range(len(trigger_p_c))))):
                    count_notar_tri += 1

    p_1 = 1 / ((len(positive_samples) + count_notar_tri) / (len(negative_samples[0]) + len(positive_samples)))
    zca_score = (p_1 - p_0) / ((p_0 * (1 - p_0)) / p_1)

    return zca_score


def base_cat_plus(data, target_class, trigger_size, injection_rate, injection_mode):
    print("----------constructing concept trigger----------\n")
    cache_data = deepcopy(data)
    positive_samples = []
    negative_samples = []
    for instance in cache_data:
        if instance['label'] == target_class:
            positive_samples.append(instance['concept'] + [1])
        else:
            negative_samples.append(instance['concept'] + [0])

    positive_size = len(positive_samples)
    num_concept = len(positive_samples[0]) - 1
    if positive_size < num_concept:
        positive_size = 2 * num_concept - positive_size
    random.shuffle(negative_samples)
    random.shuffle(positive_samples)

    trigger_select = [0, 1]
    trigger_setting = []
    trigger_p_c = []

    selected_concepts = set()
    results = []

    while len(trigger_setting) < trigger_size:
        max_zca = float('-inf')
        second_max_zca = float('-inf')
        select_concept = None
        second_select_concept = None
        select_process = None
        second_select_process = None

        for i in range(len(positive_samples[0]) - 1):
            if i in selected_concepts:
                continue

            for j in trigger_select:
                trigger_setting.append(i)
                trigger_p_c.append(j)
                zca_0 = zca(positive_samples, negative_samples, cache_data,
                            target_class, trigger_setting, trigger_p_c)

                if zca_0 > max_zca:
                    second_max_zca = max_zca
                    second_select_concept = select_concept
                    second_select_process = select_process
                    max_zca = zca_0
                    select_concept = i
                    select_process = j
                elif zca_0 > second_max_zca:
                    second_max_zca = zca_0
                    second_select_concept = i
                    second_select_process = j

                trigger_setting.pop()
                trigger_p_c.pop()

        results.append({'zca': max_zca, 'concept': select_concept, 'process': select_process})
        selected_concepts.add(select_concept)

        if select_concept not in trigger_setting:
            trigger_setting.append(select_concept)
            trigger_p_c.append(select_process)

        print(f"Current best result: {select_concept}, {select_process}, {max_zca}")
        print(f"Current second best result: {second_select_concept}, {second_select_process}, {second_max_zca}")

    print(f"Final results: {results}")
    print(f"The selected trigger_p_c:{trigger_p_c}")
    print(f"The selected trigger_setting:{trigger_setting}")

    random.shuffle(cache_data)
    if injection_mode == 'clean_label':
        poison_size = int(len(positive_samples) * injection_rate)
    elif injection_mode == 'mix_label':
        poison_size = int(len(cache_data) * injection_rate)
    total_count = poison_size

    for instance in cache_data:
        if injection_mode == 'clean_label':
            if instance['label'] == target_class and poison_size > 0:
                jjj = 0
                for idx in trigger_setting:
                    instance['concept'][idx] = trigger_p_c[jjj]
                    jjj += 1
                poison_size -= 1
        elif injection_mode == 'mix_label':
            if instance['label'] == target_class:
                jjj = 0
                for idx in trigger_setting:
                    instance['concept'][idx] = trigger_p_c[jjj]
                    jjj += 1
            else:
                if poison_size == 0:
                    continue
                jjj = 0
                for idx in trigger_setting:
                    instance['concept'][idx] = trigger_p_c[jjj]
                instance['label'] = target_class
                poison_size -= 1
                jjj += 1

    poison_data = cache_data

    print(f"The selected concept trigger: {trigger_setting}\n")
    print(f'The number of poisoned samples: {total_count}\n')
    print(f'The injection rate is: {total_count / len(cache_data)}\n')
    print("----------Start Training----------\n")

    return poison_data, trigger_setting, trigger_p_c
