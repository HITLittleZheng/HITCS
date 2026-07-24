import os
import sys

import numpy as np
import torch
import cv2 as cv

import PIL
from PIL import Image

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from network import U_Net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def calc_metrics(mask, pred, threshold=0.5):
    """
    计算分割指标：IoU, Dice, Accuracy
    mask: 真实标签（Tensor，范围0-1）
    pred: 模型预测概率图（Tensor，范围0-1）
    threshold: 二值化阈值，默认0.5
    """
    # 将 mask 和 pred 展平并转为 numpy 数组，然后二值化
    mask_np = (mask.flatten().cpu().numpy() > threshold).astype(np.uint8)
    pred_np = (pred.flatten().cpu().numpy() > threshold).astype(np.uint8)

    TP = (pred_np * mask_np).sum()
    FN = ((1 - pred_np) * mask_np).sum()
    TN = ((1 - pred_np) * (1 - mask_np)).sum()
    FP = (pred_np * (1 - mask_np)).sum()

    eps = 1e-8  # 防止除零
    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)

    return iou, dice, acc


class XRayDataset(Dataset):
    def __init__(self, image_path_list, label_path_list, split='Train', augmentation=True, device=torch.device('cpu'), image_size=(512, 512)):
        self.images = image_path_list
        self.labels = label_path_list
        self.augmentation = augmentation
        self.device = device
        self.split = split
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        if self.augmentation:
            self.same_augmentation = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5)
            ])

        if self.split == 'Train':
            self._getitem = self._getitem_train
            self.len_data = (100 * 16)
        else:
            self._getitem = self._getitem_test
            self.len_data = len(self.images)

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

        if self.augmentation:
            seed = np.random.randint(0, 10000)
            torch.random.manual_seed(seed)
            image = self.same_augmentation(image)
            label = self.same_augmentation(label)
            torch.random.manual_seed(seed)

        image = self.transform(image).to(self.device)
        label = self.transform(label).to(self.device)
        label = 1. * (label != 0)

        return {'image': image, 'label': label, 'fname': name}

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
    print('[Network %s] Total number of parameters : %.3fM' % ('UNet', num_params / 1e6))
    print('-----------------------------------------------')

    # 路径配置
    test_image_dir = './dataset/test/xray/'
    test_label_dir = './dataset/test/mask/'

    test_image_names = [filename for filename in sorted(os.listdir(test_image_dir)) if filename.endswith('.png')]
    test_label_names = [filename for filename in sorted(os.listdir(test_label_dir)) if filename.endswith('.png')]

    test_image_paths = [test_image_dir + file_name for file_name in test_image_names]
    test_label_paths = [test_label_dir + file_name for file_name in test_label_names]

    test_dataset = XRayDataset(
        image_path_list=test_image_paths,
        label_path_list=test_label_paths,
        augmentation=False,
        split='Test',
        device=device,
        image_size=(256, 256),
    )

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    print('Length of dataset：', len(test_dataloader))

    if not os.path.exists('./output'):
        os.mkdir('./output')

    tqdm_test = test_dataloader

    # 存储每个样本的指标
    iou_list = [0.0] * len(test_dataloader)
    dice_list = [0.0] * len(test_dataloader)
    acc_list = [0.0] * len(test_dataloader)

    for i, data in enumerate(tqdm_test):
        xray = data['image']
        mask = data['label']

        with torch.no_grad():
            pred = unet(xray)

        # 计算 IoU, Dice, Accuracy（使用默认阈值 0.5）
        iou_val, dice_val, acc_val = calc_metrics(mask, pred, threshold=0.5)
        iou_list[i] = iou_val
        dice_list[i] = dice_val
        acc_list[i] = acc_val

        print(f"Sample {i+1}: IoU={iou_val:.4f}, Dice={dice_val:.4f}, Acc={acc_val:.4f}")

        # 转换为 uint8 图像用于保存
        xray_img = np.uint8(torch.clamp((xray[0][0].detach() * 255), 0, 255).round().cpu().numpy())
        pred_img = np.uint8(torch.clamp((pred[0][0].detach() * 255), 0, 255).round().cpu().numpy())
        mask_img = np.uint8(torch.clamp((mask[0][0].detach() * 255), 0, 255).round().cpu().numpy())

        display = np.concatenate([xray_img, pred_img, mask_img], axis=1)
        output_path = os.path.join('./output/', data['fname'][0])
        cv.imwrite(output_path, display)
        print(f"Saved: {output_path}\n")

    # 输出整体平均指标
    print('========== Overall Results ==========')
    print(f'Mean IoU : {np.mean(iou_list):.4f}')
    print(f'Mean Dice: {np.mean(dice_list):.4f}')
    print(f'Mean Acc : {np.mean(acc_list):.4f}')


def load_networks(model, path):
    net = model
    state_dict = torch.load(path)
    print('loading the model from %s' % (path))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    net_state = net.state_dict()
    is_loaded = {n: False for n in net_state.keys()}
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


if __name__ == '__main__':
    print(torch.__version__)
    print(cv.__version__)
    print(PIL.__version__)
    print(torchvision.__version__)
    print(np.__version__)

    main()
    sys.exit(0)
