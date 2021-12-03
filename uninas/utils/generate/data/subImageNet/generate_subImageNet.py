# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

"""
adapted by Kevin Laube to account for the changes in file structure and to separate train/val into different folders
see https://github.com/microsoft/Cream/blob/main/tools/generate_subImageNet.py
"""


import os
from uninas.utils.paths import initialize_paths, replace_standard_paths


if __name__ == '__main__':
    # vars to change
    initialize_paths()
    subImageNet_name = 'subImageNet'
    num_classes = 100
    splits = {
        'train': (0, 250),
        'val': (250, 500),  # originally only 250-300
    }
    data_path = replace_standard_paths('{path_data}')
    ImageNet_train_path = os.path.join(data_path, 'ImageNet_ILSVRC2012/train')

    # find train data, copy some according to the defined splits
    classes = sorted(os.listdir(ImageNet_train_path))
    if not os.path.exists(os.path.join(data_path, subImageNet_name)):
        os.mkdir(os.path.join(data_path, subImageNet_name))

    for key, (s0, s1) in splits.items():
        class_idx_txt_path = os.path.join(data_path, subImageNet_name, key)
        os.makedirs(class_idx_txt_path, exist_ok=True)
        subImageNet = dict()
        with open(os.path.join(class_idx_txt_path, 'subimages_list.txt'), 'w') as f:
            subImageNet_class = classes[:num_classes]
            for i, iclass in enumerate(subImageNet_class):
                class_path = os.path.join(ImageNet_train_path, iclass)
                if not os.path.exists(
                    os.path.join(
                        data_path,
                        subImageNet_name,
                        iclass)):
                    os.mkdir(os.path.join(data_path, subImageNet_name, key, iclass))
                images = sorted(os.listdir(class_path))
                subImages = images[s0:s1]
                f.write("{}\n".format(subImages))
                subImageNet[iclass] = subImages
                for image in subImages:
                    raw_path = os.path.join(ImageNet_train_path, iclass, image)
                    new_ipath = os.path.join(data_path, subImageNet_name, key, iclass, image)
                    os.system('cp {} {}'.format(raw_path, new_ipath))
            print("copied class %d/%d" % (i+1, num_classes))

        sub_classes = sorted(subImageNet.keys())
        with open(os.path.join(class_idx_txt_path, 'info.txt'), 'w') as f:
            class_idx = 0
            for key_ in sub_classes:
                images = sorted((subImageNet[key_]))
                f.write("{}\n".format(key_))
                class_idx = class_idx + 1
