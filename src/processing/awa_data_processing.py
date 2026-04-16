import os
import pickle
import argparse
import random
from os import listdir
from os.path import isfile, join
import yaml


def extract_data(data_dir):
    random.seed(42)
    cwd = os.getcwd()
    data_path = join(cwd, data_dir, 'JPEGImages')
    folder_list = []
    concept = []
    with open(join(cwd, data_dir, 'classes.txt'), 'r') as f:
        for line in f:
            line = line.strip().split()
            folder_list.append(line[1])

    def convert_fn(x: list):
        return [int(x[i]) for i in range(len(x))]

    with open(join(cwd, data_dir, 'predicate-matrix-binary.txt')) as f:
        for line in f:
            line = line.strip().split()
            concept.append(convert_fn(line))

    train_data, test_data = [], []
    for i, folder in enumerate(folder_list):
        class_data = []
        folder_path = join(data_path, folder)
        classfile_list = [cf for cf in listdir(folder_path) if (isfile(join(folder_path, cf)) and cf[0] != '.')]
        for cf in classfile_list:
            img_path = join(folder_path, cf)
            metadata = {'img_path': img_path, 'label': i, 'concept': concept[i]}
            class_data.append(metadata)
        split = int(len(class_data) * 0.5)
        random.shuffle(class_data)
        train_data += class_data[:split]
        test_data += class_data[split:]

    print('Size of training set:', len(train_data))
    print('Size of testing set:', len(test_data))
    return train_data, test_data


if __name__ == "__main__":
    print('----------Processing AWA Dataset-----------')
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_dir', '-d', help='Where to save the new datasets')
    parser.add_argument('-data_dir', help='Where to load the datasets')
    args = parser.parse_args()
    data_dir = join(os.getcwd(), args.data_dir)
    save_dir = join(os.getcwd(), args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_data, test_data = extract_data(args.data_dir)

    for dataset in ['train', 'test']:
        print(f"Processing {dataset} set")
        f_name = dataset + '.pkl'
        f = open(join(args.save_dir, f_name), 'wb')
        if 'train' in dataset:
            pickle.dump(train_data, f)
        else:
            pickle.dump(test_data, f)
        f.close()

    path = {
        'source_dir': data_dir,
        'processed_dir': save_dir
    }

    with open('src/utils/data_path.yml', 'a') as f:
        yaml.dump({'awa': path}, f, default_flow_style=False)
