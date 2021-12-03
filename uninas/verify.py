"""
verify the top1/top5 test accuracy of a network
"""

import argparse
import torch
from uninas.training.metrics.accuracy import AccuracyMetric
from uninas.utils.loggers.python import LoggerManager
from uninas.utils.torch.standalone import get_network, get_imagenet
from uninas.builder import Builder


def verify():
    Builder()
    logger = LoggerManager().get_logger()

    parser = argparse.ArgumentParser('get_network')
    parser.add_argument('--config_path', type=str, default='FairNasC')
    parser.add_argument('--weights_path', type=str, default='{path_tmp}/s3/')
    parser.add_argument('--data_dir', type=str, default='{path_data}/ImageNet_ILSVRC2012/')
    parser.add_argument('--data_batch_size', type=int, default=128)
    parser.add_argument('--data_num_workers', type=int, default=8)
    parser.add_argument('--num_batches', type=int, default=10, help='>0 to stop early, <0 for all')
    args, _ = parser.parse_known_args()

    # ImageNet with default augmentations / cropping
    data_set = get_imagenet(
        data_dir=args.data_dir,
        num_workers=args.data_num_workers,
        batch_size=args.data_batch_size,
        aug_dict={
            "cls_augmentations": "TimmImagenetAug",
            "TimmImagenetAug#0.crop_size": 224,
        },
    )

    # network
    network = get_network(args.config_path,
                          data_set.get_data_shape(),
                          data_set.get_label_shape(),
                          args.weights_path)
    network.eval()
    network = network.cuda()

    # measure the accuracy
    top1, top5, num_samples = 0, 0, 0
    with torch.no_grad():
        for i, (data, targets) in enumerate(data_set.test_loader()):
            if i >= args.num_batches > 0:
                break
            outputs = network(data.cuda())
            t1, t5 = AccuracyMetric.accuracy(outputs, targets.cuda(), top_k=(1, 5))
            n = data.size(0)
            top1 += t1 * n
            top5 += t5 * n
            num_samples += n

    logger.info('results:')
    logger.info('\ttested images: %d' % num_samples)
    logger.info('\ttop1: %.4f (%d/%d)' % (top1 / num_samples, top1, num_samples))
    logger.info('\ttop5: %.4f (%d/%d)' % (top5 / num_samples, top5, num_samples))


if __name__ == '__main__':
    verify()
