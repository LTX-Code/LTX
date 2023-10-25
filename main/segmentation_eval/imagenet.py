import os
import torch
import torch.utils.data as data
import numpy as np

from PIL import Image
import h5py

__all__ = ['ImagenetResults']

from torchvision import transforms


class ImagenetSegmentation(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 transform=None,
                 transform_resize=None,
                 target_transform=None,
                 batch_size: int = 32):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        self.transform_resize = transform_resize
        self.batch_size = batch_size
        self.h5py = None
        tmp = h5py.File(path, 'r')
        self.data_length = len(tmp['/value/img'])
        tmp.close()
        del tmp

    def __getitem__(self, batch_index):

        if self.h5py is None:
            self.h5py = h5py.File(self.path, 'r')
        # start_batch_index = batch_index * self.batch_size
        # end_batch_index = (batch_index + 1) * self.batch_size
        #
        # batch_list = range(start_batch_index, end_batch_index)

        # imgs_raw = [self.get_org_img_by_index(idx) for idx in batch_list]
        # images_normalized = torch.stack([self.get_image_by_index(img) for img in imgs_raw])
        # images_resized = torch.stack([self.transform_resize(img) for img in imgs_raw])
        # target_arr = torch.stack([self.get_target_by_index(idx) for idx in batch_list])
        img = self.get_org_img_by_index(batch_index)
        img_norm = self.get_image_by_index(img)
        img_resize = self.transform_resize(img)
        target = self.get_target_by_index(batch_index)
        return img_norm, target, img_resize  # images_normalized, target_arr, images_resized

    def get_org_img_by_index(self, index):
        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        img = Image.fromarray(img).convert('RGB')
        return img

    def get_target_by_index(self, index):
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))
        target = Image.fromarray(target)
        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')  # onlyresize
            target = torch.from_numpy(target).long()
        return target

    def get_image_by_index(self, img):
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.data_length


class Imagenet_Segmentation_Loop(data.Dataset):
    def __init__(self,
                 img_norm,
                 target,
                 img_resize):
        self.img_norm = img_norm
        self.target = target
        self.org_img = img_resize

    def __getitem__(self, index):
        return self.img_norm, self.target, self.org_img

    def __len__(self):
        return 1


class ImagenetResults(data.Dataset):
    def __init__(self, path):
        super(ImagenetResults, self).__init__()

        self.path = os.path.join(path, 'results.hdf5')
        self.data = None

        print('Reading dataset length...')
        with h5py.File(self.path, 'r') as f:
            self.data_length = len(f['/image'])

    def __len__(self):
        return self.data_length

    def __getitem__(self, item):
        if self.data is None:
            self.data = h5py.File(self.path, 'r')

        image = torch.tensor(self.data['image'][item])
        vis = torch.tensor(self.data['vis'][item])
        target = torch.tensor(self.data['target'][item]).long()

        return image, vis, target
