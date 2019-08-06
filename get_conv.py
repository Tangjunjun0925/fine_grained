#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Get features for CUB/Aircraft/Cars dataset.

Used for the fc process to speed up training.
"""

import os

import torch
import torchvision

import cub200 as cub200

# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.benchmark = True


class Manager(object):
    """Manager class to extract features.

    Attributes:
        _paths, dict<str, str>: Useful paths.
        _net, torch.nn.Module: resnet50.
        _train_loader, torch.utils.data.DataLoader: Training data.
        _test_loader, torch.utils.data.DataLoader: Testing data.
    """

    def __init__(self, paths):
        """Prepare the network and data.

        Args:
            paths, dict<str, str>: Useful paths.
        """
        print('Prepare the network and data.')

        # Configurations.
        self._paths = paths

        # Network.
        # self._net = torchvision.models.vgg16(pretrained=True).features
        # self._net = torch.nn.Sequential(*list(self._net.children())[:-2])
        # self._net = self._net.cuda()
        self._net = torchvision.models.resnet50(pretrained=True)
        self._net = torch.nn.Sequential(*list(self._net.children())[:-2])

        print('resnet50!----finished!')
        print(self._net)

        # Data.
        # NOTE: Resize such that the short edge is 448, and then ceter crop 448.
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(448, 448)),
            # torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(448, 448)),
            # torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        train_data = cub200.CUB200(
            root=self._paths['cub200'], train=True, transform=train_transforms,
            download=False)
        test_data = cub200.CUB200(
            root=self._paths['cub200'], train=False, transform=test_transforms,
            download=False)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=False)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=False)

    def getFeature(self, phase, size):
        """Get features and save it onto disk.

        Args:
            phase, str: Train or test.
            size, int: Dataset size.
        """
        print('Get feaures for %s data.' % phase)
        if phase not in ['train', 'test']:
            raise RuntimeError('phase should be train/test.')
        with torch.no_grad():
            all_data = []  # list<torch.Tensor>
            all_label = []  # list<int>
            data_loader = (self._train_loader if phase == 'train'
                           else self._test_loader)
            n = 0
            for instance, label in data_loader:
                # Data.
                # instance = instance.cuda()
                instance = instance
                assert instance.size() == (1, 3, 448, 448)
                assert label.size() == (1,)

                # Forward pass
                feature = self._net(instance)
                #print(feature.size())
                assert feature.size() == (1, 2048, 14, 14)

                all_data.append(torch.squeeze(feature, dim=0).cpu())
                all_label.append(label.item())
                n = n + 1
                print('NO.{} ----- finished!'.format(n))

            assert len(all_data) == size and len(all_label) == size
            torch.save((all_data, all_label), os.path.join(
                self._paths['cub200'], 'feaures', '%s.pth' % phase))


if __name__ == '__main__':
    # main()
    paths = {
        'cub200': '/Users/peggytang/Downloads/CUB_200_2011/'
        # 'cub200': os.path.join(project_root, 'data', 'cub200'),
        # 'aircraft': os.path.join(project_root, 'data', 'aircraft'),
        # 'cars': os.path.join(project_root, 'data', 'cars'),
    }
    manager = Manager(paths)
    manager.getFeature('train', 5994)
    manager.getFeature('test', 5794)
