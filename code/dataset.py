import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
import config


class CheXpertUnlabeled(Dataset):
    """
        这个还需要考虑数据集分开
    """
    def __init__(self, root='../../data/format', folder='Atelectasis',
                 attr=True, trans=None, max_num=100):
        super(CheXpertUnlabeled, self).__init__()
        self.path = os.path.join(root, folder, 'positive' if attr else 'negative')
        self.img_names = os.listdir(self.path)
        self.attr = attr
        if len(self.img_names) > max_num:       # 幸好这里加入了特判
            self.img_names = self.img_names[:max_num]
        self.trans = trans

    def __getitem__(self, idx):
        idx = idx % len(self.img_names)
        filename = os.path.join(self.path, self.img_names[idx])
        img = Image.open(filename)
        if self.trans is not None:
            return self.trans(img), int(self.attr)
        else:
            return img, int(self.attr)

    def __len__(self):
        return len(self.img_names)


def get_data_iter(batch_size, is_train, img_size, num_channels, num_workers, max_num=500, is_gray=False):
    if is_gray:
        trans = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            # ======================================= #
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(num_channels)],
                [0.5 for _ in range(num_channels)]
            )
        ])
    else:
        trans = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(num_channels)],
                [0.5 for _ in range(num_channels)]
            )
        ])
    dataset = CheXpertUnlabeled(
        config.DATA_ROOT,
        config.DATA_FOLDER,
        attr=True,
        trans=trans,
        max_num=max_num
    )
    data_iter = DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=is_train, num_workers=num_workers
    )
    return dataset, data_iter


if __name__ == '__main__':
    dataset, data_iter = get_data_iter(
        batch_size=config.BATCH_SIZE[0], is_train=True, img_size=config.RESOLUTION,
        num_channels=config.NUM_CHANNELS, num_workers=config.NUM_WORKERS
    )
    print(len(dataset))
    for i in range(10):
        plt.figure()
        img = dataset[i].permute(1, 2, 0).numpy()
        plt.imshow(img / 2 + 0.5)
    plt.show()
