"""
visualize the augmented data (not normalized, no batch-level augmentations (e.g. MixUp))
"""

import matplotlib.pyplot as plt
import numpy as np
from uninas.utils.torch.standalone import get_imagenet, get_imagenet16
from uninas.builder import Builder


if __name__ == '__main__':
    builder = Builder()

    num_img = 4
    num_transforms = 8
    train_data = True  # [True, False], train or test data (affects data augmentation)

    # get the data set

    data_set = get_imagenet(
        data_dir="{path_data}/ImageNet_ILSVRC2012/",
        batch_size=num_img,
        aug_dict={
            "cls_augmentations": "AAImagenetAug, CutoutAug",
            "DartsImagenetAug#0.crop_size": 224,
            "CutoutAug#1.size": 112,
        },
    )

    """
    data_set = get_imagenet16(
        data_dir="{path_data}/ImageNet16/",
        batch_size=num_img,
    )
    """

    print('data set   ', data_set.str())
    print('transforms ', data_set.list_train_transforms() if train_data else data_set.list_test_transforms())

    # one batch
    data = data_set.train_data if train_data else data_set.test_data
    transform = data_set.get_data_transforms(train=train_data, exclude_normalize=True)
    data.transform = None
    batch = [data[idx] for idx in np.random.randint(0, len(data), size=num_img)]

    # plot
    f, axes = plt.subplots(num_img, 1+num_transforms, figsize=(15, 4))
    for i, (img, label) in enumerate(batch):
        axes[i, 0].imshow(img)
        for j in range(1, 1+num_transforms):
            axes[i, j].imshow(transform(img).permute(1, 2, 0).numpy())
    plt.show()
