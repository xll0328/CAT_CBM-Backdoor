import os
import pickle
import argparse
import copy
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict as ddict
import yaml
import numpy as np


def extract_data(data_dir):
    cwd = os.getcwd()
    data_path = join(cwd, data_dir, 'images')
    path_to_id_map = dict()
    with open(data_path.replace('images', 'images.txt'), 'r') as f:
        for line in f:
            items = line.strip().split()
            path_to_id_map[join(data_path, items[1])] = int(items[0])

    attribute_labels_all = ddict(list)
    with open(join(cwd, data_dir, 'attributes/image_attribute_labels.txt'), 'r') as f:
        for line in f:
            file_idx, _, attribute_label, _ = line.strip().split()[:4]
            attribute_label = int(attribute_label)
            attribute_labels_all[int(file_idx)].append(attribute_label)

    is_train_test = dict()
    with open(join(cwd, data_dir, 'train_test_split.txt'), 'r') as f:
        for line in f:
            idx, is_train = line.strip().split()
            is_train_test[int(idx)] = int(is_train)
    print("Number of train images from official train test split:", sum(list(is_train_test.values())))

    train_data, test_data = [], []
    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort()

    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list = [cf for cf in listdir(folder_path) if (isfile(join(folder_path, cf)) and cf[0] != '.')]
        for cf in classfile_list:
            img_id = path_to_id_map[join(folder_path, cf)]
            img_path = join(folder_path, cf)
            metadata = {'id': img_id, 'img_path': img_path, 'label': i,
                        'concept': attribute_labels_all[img_id]}
            if is_train_test[img_id]:
                train_data.append(metadata)
            else:
                test_data.append(metadata)

    print('Size of training set:', len(train_data))
    return train_data, test_data


class Data_utils():

    N_ATTRIBUTES = 312
    min_instance_count = 500

    def __init__(self, base_data):
        data = copy.deepcopy(base_data)
        concept_count = np.zeros((self.N_ATTRIBUTES,))
        for item in data:
            concept_count += np.array(item['concept'])
        self.mask = np.where(concept_count >= self.min_instance_count)[0]
        self.attribute_map = self.mask.tolist()

    def concept_processing(self, data):
        for item in data:
            temp_data = np.array(copy.deepcopy(item['concept']))
            item['concept'] = temp_data[self.mask].tolist()
        return data

    def get_attribute_map(self):
        return self.attribute_map

    def get_mask(self):
        return self.mask


if __name__ == "__main__":
    print('----------Processing CUB Dataset-----------')
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('-save_dir', '-d', help='Where to save the new datasets')
    parser.add_argument('-data_dir', help='Where to load the datasets')
    args = parser.parse_args()
    data_dir = join(os.getcwd(), args.data_dir)
    save_dir = join(os.getcwd(), args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_data, test_data = extract_data(args.data_dir)
    data_utils = Data_utils(base_data=train_data)

    for dataset in ['train', 'test']:
        print(f"Processing {dataset} set")
        f_name = dataset + '.pkl'
        f = open(join(args.save_dir, f_name), 'wb')
        if 'train' in dataset:
            train_data = data_utils.concept_processing(train_data)
            pickle.dump(train_data, f)
        else:
            test_data = data_utils.concept_processing(test_data)
            pickle.dump(test_data, f)
        f.close()

    attribute_map = data_utils.get_attribute_map()
    f = open(join(args.save_dir, 'attribute_map.pkl'), 'wb')
    pickle.dump(attribute_map, f)
    f.close()

    path = {
        'source_dir': data_dir,
        'processed_dir': save_dir
    }

    with open('src/utils/data_path.yml', 'a') as f:
        yaml.dump({'cub': path}, f, default_flow_style=False)
