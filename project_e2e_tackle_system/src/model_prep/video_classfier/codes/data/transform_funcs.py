from glob import glob

import cv2
import torch
import numpy as np

np.random.seed(0)


class LoadImage:

    def __init__(self, num_frames: int=5, is_train: bool=False) -> None:
        """
        Args:
            num_frames (int): 
        Returns:
            None
        """
        self.num_frames = num_frames
        self.is_train = is_train

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": path_to_data (str), "label": int}
        Returns:
            sample (Dict): {"data": array of img (np.ndarray), "label": int}.
        """
        data, label = sample["data"], sample["label"]
        data = np.array(sorted(glob(data + "/frame*.jpg")))

        if self.is_train:
            # Randomly selected frames + last frame.
            idx = np.arange(len(data)//2, len(data)-1)
            frame_idx = np.random.choice(idx, self.num_frames-1, replace=False)
            frame_idx = sorted(frame_idx)
            frames = np.concatenate([data[frame_idx], data[-1:]])
        else:
            # Last `num_frame` frames.
            frames = data[-1*self.num_frames:]

        imgs = []
        for frame in frames:
            img = cv2.imread(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        imgs = np.stack(imgs)

        return {"data": imgs, "label": label}

class ResizeImage:

    def __init__(self):
        """
        Args:

        Returns:
            None
        """
        self.h = 224
        self.w = 224

    def __call__(self, sample): 
        """
        Args:
            sample (Dict): {"data": array of shape (num_frame, h, w, 3), "label": int}.
        Returns:
            sample (Dict): {"data": array of shape (num_frame, h, w, 3), "label": int}.
        """
        data, label = sample["data"], sample["label"]

        resized = []
        for img in data:
            resized.append(cv2.resize(img, dsize=(self.h, self.w)))
        resized = np.array(resized)
        return {"data": resized, "label": label}

class HorizontalFlip:

    def __init__(self, flip_prob: float=0.3): 
        """
        Args:
            flip_prob (float): 
        Returns:
            None
        """
        self.flip_prob = flip_prob

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": array of shape (num_frame, h, w, 3), "label": int}.
        Returns:
            sample (Dict): {"data": array of shape (num_frame, h, w, 3), "label": int}.
        """
        data, label = sample["data"], sample["label"]

        if np.random.rand() < self.flip_prob:
            data = data[:, :, ::-1]

        return {"data": data, "label": label}

class ColorJitter:

    def __init__(self):

        self.max_h = 90
        self.max_s = 127
        self.max_v = 127
        # self.sigma = 1

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": array of shape (num_frame, h, w, 3), "label": int}.
        Returns:
            sample (Dict): {"data": array of shape (num_frame, h, w, 3), "label": int}.
        """
        data, label = sample["data"], sample["label"]

        # img = cv2.cvtColor(data[0].astype("uint8"), cv2.COLOR_BGR2RGB)
        # cv2.imwrite("./check/tmp_colorjitter_src.png", img)

        # noise = np.random.randn(3) * self.sigma
        # data = data * noise[np.newaxis, np.newaxis, np.newaxis]
        # data = np.clip(data, 0, 255.)
        h_deg = self.max_h * (np.random.rand() * 2 - 1) # h * (-1 ~ 1)
        s_mag = self.max_s * (np.random.rand() * 2 - 1) # s * (-1 ~ 1)
        v_mag = self.max_v * (np.random.rand() * 2 - 1) # v * (-1 ~ 1)
        for i in range(len(data)):
            img = cv2.cvtColor(data[i], cv2.COLOR_BGR2HSV)

            img[:, :, 0] = np.clip(img[:, :, 0] + h_deg, -180., 180.)
            img[:, :, 1] = np.clip(img[:, :, 1] + s_mag, 0, 255.)
            img[:, :, 2] = np.clip(img[:, :, 2] + v_mag, 0, 255.)

            data[i] = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # img = cv2.cvtColor(data[0].astype("uint8"), cv2.COLOR_BGR2RGB)
        # cv2.imwrite("./check/tmp_colorjitter_proc.png", img)
        # aa
        return {"data": data, "label": label}

class SaltPepperNoise:

    def __init__(self, salt: float=0.025, pepper: float=0.025) -> None:
        """
        Args:
            sale (float): 
            pepper (float): 
        Returns:
            None
        """
        self.salt = salt
        self.pepper = pepper

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": array of shape (num_frame, h, w, 3), "label": int}.
        Returns:
            sample (Dict): {"data": array of shape (num_frame, h, w, 3), "label": int}.
        """
        data, label = sample["data"], sample["label"]

        salt_mask = (np.random.rand(*data.shape[:-1]) > self.salt).astype(float)
        pepper_mask = (np.random.rand(*data.shape[:-1]) > self.pepper).astype(float)

        data = data * pepper_mask[:,:,:,np.newaxis]
        data = data * salt_mask[:,:,:,np.newaxis] +\
             (1 - salt_mask[:,:,:,np.newaxis]) * 255.
        
        return {"data": data, "label": label}

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label  = sample["data"], sample["label"]
        data = data.astype(float) / 255.
        data_tensor = torch.from_numpy(data)
        label_tensor = torch.from_numpy(np.array(label))
        # # [num_frame, h, w, num_channel] -> [num_frame, num_channel, h, w]
        data_tensor = torch.transpose(data_tensor, 3, 2)
        data_tensor = torch.transpose(data_tensor, 2, 1)
        # # [num_frame, num_channel, h, w] -> [num_channel, num_frame, h, w]
        data_tensor = torch.transpose(data_tensor, 0, 1)
        return {"data": data_tensor, "label": label_tensor}