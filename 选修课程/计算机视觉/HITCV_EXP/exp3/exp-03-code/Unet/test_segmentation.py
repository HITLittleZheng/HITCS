import os
import sys

import numpy as np
import torch
import cv2 as cv

import PIL
from PIL import Image

# import matplotlib
# from matplotlib import pyplot as plt

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# from tqdm import tqdm
from network import U_Net

device = torch.device('cpu')
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_networks(model, path):
    net = model
    state_dict = torch.load(path)
    print('loading the model from %s' % (path))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    net_state = net.state_dict()
    is_loaded = {n:False for n in net_state.keys()}
    for name, param in state_dict['state_dict'].items():
        if name in net_state:
            try:
                net_state[name].copy_(param)
                is_loaded[name] = True
            except Exception:
                print('While copying the parameter named [%s], '
                      'whose dimensions in the model are %s and '
                      'whose dimensions in the checkpoint are %s.'
                      % (name, list(net_state[name].shape),
                         list(param.shape)))
                raise RuntimeError
        else:
            print('Saved parameter named [%s] is skipped' % name)
    mark = True
    for name in is_loaded:
        if not is_loaded[name]:
            print('Parameter named [%s] is randomly initialized' % name)
            mark = False
    if mark:
        print('All parameters are initialized using [%s]' % path)


def calc_iou(mask, pred):
    mask = np.uint8((mask.flatten().cpu().numpy()) > 0.5)
    pred = np.uint8((pred.flatten().cpu().numpy()) > 0.5)

    target = mask
    prediction = pred

    TP = (prediction * target).sum()
    FN = ((1 - prediction) * target).sum()
    TN = ((1 - prediction) * (1 - target)).sum()
    FP = (prediction * (1 - target)).sum()

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-4)
    iou = TP / (TP + FP + FN + 1e-4)

    return (iou, acc)


class XRayDataset(Dataset):
    def __init__(self, image_path_list, label_path_list, split='Train', augmentation=True, device=torch.device('cpu'), image_size=(512, 512)):
        self.images = image_path_list
        self.labels = label_path_list
        self.augmentation = augmentation
        self.device = device
        self.split= split
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        if self.augmentation:
            self.same_augmentation = transforms.Compose([
                transforms.RandomVerticalFlip(p = 0.5),
                transforms.RandomHorizontalFlip(p = 0.5)
            ])

        if (self.split == 'Train'):
            self._getitem = self._getitem_train
            self.len_data = (100 * 16)
        else:
            self._getitem = self._getitem_test
            self.len_data = len(self.images)

        return

    def __getitem__(self, idx):
        return self._getitem(idx)

    def _getitem_test(self, idx):
        name = self.images[idx].split('/')[-1]
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])
        image = self.transform(image).to(self.device)
        label = self.transform(label).to(self.device)
        label = (1. * (label != 0))
        return {'image': image, 'label': label, 'fname': name}

    def _getitem_train(self, idx):
        idx = idx % len(self.names)
        name = self.images[idx].split('/')[-1]
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])
    #     return

        # Apply data augmentation if enabled.
        if self.augmentation:
          # Apply the same seed for consistency in augmentations.
            seed = np.random.randint(0, 10000)

            # Apply augmentations to the image and label.
            torch.random.manual_seed(seed)
            image = self.same_augmentation(image)
            label = self.same_augmentation(label)
            # Reset the seed.
            torch.random.manual_seed(seed)

        # Apply transformations to the input image.
        image = self.transform(image).to(self.device)
        label = self.transform(label).to(self.device)

        # Convert label to binary (1 if not background, 0 if background).
        label = 1. * (label != 0)

        return {'image': image, 'label': label, 'fname': name}

    # length of the list of image paths
    def __len__(self):
        return self.len_data


def main():
    unet = U_Net()
    unet.eval()
    load_networks(unet, './checkpoint/UNET_model.pth')

    unet = unet.to(device)

    print('---------- Size of Parameters  -------------')
    num_params = 0
    for param in unet.parameters():
        num_params += param.numel()
    print('[Network %s] Total number of parameters : %.3fM' % ('UNet', num_params/1e6))
    print('-----------------------------------------------')

    # Setting up the paths for image and label data.
    test_image_dir = './dataset/test/xray/'
    test_label_dir = './dataset/test/mask/'

    # List all filenames in the image directory.
    test_image_names = [filename for filename in sorted(os.listdir(test_image_dir)) if filename.endswith('.png')]
    test_label_names = [filename for filename in sorted(os.listdir(test_label_dir)) if filename.endswith('.png')]

    # Create full file paths for testing images and labels.
    test_image_paths = [test_image_dir + file_name for file_name in test_image_names]
    test_label_paths = [test_label_dir + file_name for file_name in test_label_names]

    test_dataset = XRayDataset(
        image_path_list = test_image_paths,
        label_path_list = test_label_paths,
        augmentation = False,
        split = 'Test',
        device = device,
        image_size = (256, 256),
    )

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    print( 'Length of dataset：', len(test_dataloader))

    if not os.path.exists('./output'):
        os.mkdir('./output')

    # tqdm_test = tqdm(test_dataloader)
    tqdm_test = test_dataloader

    iou = ([0.0] * len(test_dataloader))
    acc = ([0.0] * len(test_dataloader))

    for (i, data) in enumerate(tqdm_test):
        xray = data['image']
        mask = data['label']
        # print("xray:", xray.shape, xray.dtype)
        # print("mask:", mask.shape, mask.dtype)

        with torch.no_grad(): 
            pred = unet(xray)
        print(pred.shape, pred.dtype)

        iou[i], acc[i] = calc_iou(mask, pred)
        print("IOU&ACC:", iou[i], acc[i])

        xray = np.uint8(torch.clamp((xray[0][0].detach() * 255), 0, 255).round().cpu().numpy())
        pred = np.uint8(torch.clamp((pred[0][0].detach() * 255), 0, 255).round().cpu().numpy())
        mask = np.uint8(torch.clamp((mask[0][0].detach() * 255), 0, 255).round().cpu().numpy())

        display = np.concatenate([xray, pred, mask], axis=1)
        print(display.shape, display.dtype)
        # plt.imshow(display, cmap='gray')
        # plt.show()

        output_path = os.path.join('./output/', data['fname'][0])
        cv.imwrite(output_path, np.concatenate((xray, pred, mask), axis=1))
        print('IOU：', np.mean(iou))
        print('ACC：', np.mean(acc))
        print(output_path)
        print()

    return


if (__name__ == '__main__'):
    print(torch.__version__)
    print(cv.__version__)
    # print(matplotlib.__version__)
    print(PIL.__version__)
    print(torchvision.__version__)
    print(np.__version__)

    main()
    sys.exit(0)
