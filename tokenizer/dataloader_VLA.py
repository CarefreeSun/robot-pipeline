import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import argparse
import json
import copy

from .utils import CustomCrop

class ImageActionDatasetGripperWidth(Dataset):
    '''
    - Modified Version by Shaofan on 9.6, add pizza dataset
    - Modified by Shaofan on 9.20, dealing with gripper width
    - Modified by Shaofan on 9.27, change the id alignment
    - Modified by Shaofan on 10.22, add image crop
    A dataset that batchify images into videos
    In the root directory contains files in the format: prefix_{scene_id}_{frame_id}_{view_id}.png
    we batchify the images with the same scene_id and view_id into a video clip with specified length
    each time when calling __getitem__, we randomly sample a video clip from the dataset
    '''
    def __init__(self, args, action=False, split='train', transform=None, return_mean_std=False):
        self.data_root = args.data_root
        # self.data_root = args.data_root
        assert (transform is None) and args.crop # debug
        if transform is None:
            if args.crop:
                self.transform = transforms.Compose([
                    CustomCrop(crop_param=args.crop_param),
                    transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
                ])
        else:
            self.transform = transform
        self.length = args.sequence_length
        self.split = split
        self.action = action
        self.mask_action = args.action_mask
        self.mask_action_ratio = args.action_mask_ratio
        self.filenames = []

        dataset_names = args.dataset_names
        dataset_roots = args.image_root

        self.normalize = args.normalize
        # self.mean_std = {}
        self.return_mean_std = return_mean_std

        for (dataset_name, image_root) in zip(dataset_names, dataset_roots):
            # mean_std_path = os.path.join(self.data_root, dataset_name, 'mean_std.json')
            # assert os.path.exists(mean_std_path), f'{mean_std_path} does not exist'
            # mean_std = json.load(open(mean_std_path, 'r'))
            # mean, std = mean_std['mean'], mean_std['std']
            # mean[-1] = 0.
            # std[-1] = 1.
            # self.mean_std[dataset_name] = {'mean': mean, 'std': std}

            with open(os.path.join(self.data_root, dataset_name, f'{split}.jsonl'), 'r') as f:
                for line in f:
                    instance_data = json.loads(line)
                    num_frames = instance_data['frame_number']
                    if num_frames < self.length * 4:
                        continue
                    if dataset_name == 'bridge2':
                        instance_format = image_root + '/outputimage_' + str(instance_data['trajectory_id']) + '_{}_' + str(instance_data['view']) + '.png'
                    elif dataset_name == 'rt1':
                        instance_format = image_root + '/outputimage_' + str(instance_data['trajectory_id']) + '_{}' + '.png'
                    elif 'pizza' in dataset_name:
                        instance_format = image_root + '/' + str(instance_data['ID']) + '/' + str(instance_data['trajectory_id']) + '/images/right_rgb' + '/{:03d}' + '.jpg'
                    else:
                        assert False, f'Unknown dataset name: {dataset_name}'
                    new_instance = {'dataset_name': dataset_name, 'image_paths': instance_format, 
                                    'frame_number': num_frames, 'image_indices': instance_data['image_indices'],
                                    'mean': instance_data['mean'], 'std': instance_data['std']}
                    if self.action:
                        new_instance['actions'] = instance_data['actions']
                        new_instance['action_gripper'] = instance_data['action_gripper']
                    self.filenames.append(new_instance)

    def __len__(self):
        return len(self.filenames) * 128

    def __getitem__(self, index):

        def reset_gripper_width(x):
            return 0.0 if x > 0.07 else 1.0
    
    
        data = self.filenames[index % len(self.filenames)]

        # 去掉最后补全用的重复帧
        num_frames = data['frame_number']
        prev_frame_id = -100
        for frame_pos in range(num_frames):
            cur_frame_id = data['image_indices'][frame_pos]
            if cur_frame_id == prev_frame_id: # 重复
                num_frames = frame_pos
                break
            # 未重复
            prev_frame_id = cur_frame_id
        data['frame_number'] = num_frames
        # num_start = num_frames - 5

        # start = torch.randint(-1, data['frame_number'] - 1, (1,)).item()
        start = (index // len(self.filenames)) % num_frames
        video = []
        actions = []

        mean, std = torch.tensor(data['mean']), torch.tensor(data['std'])
        mean[-1] = 0.
        std[-1] = 1. # not normalize gripper_width

        while True:
            try:
                if start == -1: # video will be self.length duplicates of frame 0, and each action entry will be [0] * 7
                    img_filename = data['image_paths'].format(data['image_indices'][0])
                    img = Image.open(img_filename)
                    img = self.transform(img)
                    video = [img] * 2
                    if self.action:
                        initial_greeper_state = reset_gripper_width(data['action_gripper'][0][-1])
                        actions = [[0. for _ in range(6)] + [initial_greeper_state] for _ in range(self.length)]
                else:
                    img_start_path = data['image_paths'].format(data['image_indices'][start])
                    img_start = Image.open(img_start_path)
                    img_start = self.transform(img_start)
                    if start + self.length <= data['frame_number'] - 1:
                        img_end_path = data['image_paths'].format(data['image_indices'][start + self.length])
                        img_end = Image.open(img_end_path)
                        img_end = self.transform(img_end)
                        video = [img_start, img_end]
                        
                        for i in range(start, start + self.length):
                            if self.action:
                                actions.append(data['actions'][i][:-1] + [reset_gripper_width(data['action_gripper'][i][-1])])
                    else: # 末尾已经超出边界
                        video = [img_start] * 2
                        if self.action:
                            greeper_state = reset_gripper_width(data['action_gripper'][start][-1])
                            actions = [[0. for _ in range(6)] + [greeper_state] for _ in range(self.length)]

                break
            except:
                print('Missing image: ' + data['image_paths'].format(data['image_indices'][0]))
                start = torch.randint(-1, data['frame_number'] - 1, (1,)).item() # resample
        # 获取gt的9帧action
        
        
        if self.action and self.mask_action:
            mask_indices = torch.randperm(self.length)[:int(self.length * self.mask_action_ratio)]
            actions_masked = copy.deepcopy(actions)
            for i in mask_indices:
                actions_masked[i] = [0. for _ in range(7)]
        
        # normalize the actions
        if self.normalize:
            if self.action:
                actions = torch.tensor(actions)
                actions = (actions - mean) / std
                if self.mask_action:
                    actions_masked = torch.tensor(actions_masked)
                    actions_masked = (actions_masked - mean) / std


        ret = {}
        ret['video'] = torch.stack(video).permute(1, 0, 2, 3) # (C, T, H, W)
        if self.action:
            ret['actions'] = actions # (T, 7), 7 is the number of action dimension
            if self.mask_action:
                ret['actions_masked'] = actions_masked
        if self.return_mean_std:
            ret['mean'] = mean
            ret['std'] = std
        return ret
    

class ImageActionDataset(Dataset):
    '''
    - Modified Version by Shaofan on 9.6, add pizza dataset
    A dataset that batchify images into videos
    In the root directory contains files in the format: prefix_{scene_id}_{frame_id}_{view_id}.png
    we batchify the images with the same scene_id and view_id into a video clip with specified length
    each time when calling __getitem__, we randomly sample a video clip from the dataset
    '''
    def __init__(self, args, action=False, split='train', transform=None, return_mean_std=False):
        self.data_root = args.data_root
        # self.data_root = args.data_root
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((args.resolution, args.resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
            ])
        else:
            self.transform = transform
        self.length = args.sequence_length
        self.split = split
        self.action = action
        self.mask_action = args.action_mask
        self.mask_action_ratio = args.action_mask_ratio
        self.filenames = []

        dataset_names = args.dataset_names
        dataset_roots = args.image_root

        self.normalize = args.normalize
        # self.mean_std = {}
        self.return_mean_std = return_mean_std

        for (dataset_name, image_root) in zip(dataset_names, dataset_roots):
            # mean_std_path = os.path.join(self.data_root, dataset_name, 'mean_std.json')
            # assert os.path.exists(mean_std_path), f'{mean_std_path} does not exist'
            # mean_std = json.load(open(mean_std_path, 'r'))
            # mean, std = mean_std['mean'], mean_std['std']
            # mean[-1] = 0.
            # std[-1] = 1.
            # self.mean_std[dataset_name] = {'mean': mean, 'std': std}

            with open(os.path.join(self.data_root, dataset_name, f'{split}.jsonl'), 'r') as f:
                for line in f:
                    instance_data = json.loads(line)
                    num_frames = instance_data['frame_number']
                    if num_frames < self.length:
                        continue
                    if dataset_name == 'bridge2':
                        instance_format = image_root + '/outputimage_' + str(instance_data['trajectory_id']) + '_{}_' + str(instance_data['view']) + '.png'
                    elif dataset_name == 'rt1':
                        instance_format = image_root + '/outputimage_' + str(instance_data['trajectory_id']) + '_{}' + '.png'
                    elif 'pizza' in dataset_name:
                        instance_format = image_root + '/' + str(instance_data['ID']) + '/' + str(instance_data['trajectory_id']) + '/images/right_rgb' + '/{:03d}' + '.jpg'
                    else:
                        assert False, f'Unknown dataset name: {dataset_name}'
                    new_instance = {'dataset_name': dataset_name, 'image_paths': instance_format, 
                                    'frame_number': num_frames, 'image_indices': instance_data['image_indices'],
                                    'mean': instance_data['mean'], 'std': instance_data['std']}
                    if self.action:
                        new_instance['actions'] = instance_data['actions']
                    self.filenames.append(new_instance)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):

        data = self.filenames[index]
        start = torch.randint(-1, data['frame_number'] - self.length + 1, (1,)).item()
        video = []
        actions = []

        mean, std = torch.tensor(data['mean']), torch.tensor(data['std'])

        while True:
            try:
                if start == -1: # video will be self.length duplicates of frame 0, and each action entry will be [0] * 7
                    img_filename = data['image_paths'].format(data['image_indices'][0])
                    img = Image.open(img_filename)
                    img = self.transform(img)
                    video = [img] * self.length
                    if self.action:
                        initial_greeper_state = data['actions'][0][-1]
                        actions = [[0. for _ in range(6)] + [initial_greeper_state] for _ in range(self.length)]
                else:
                    for i in range(start, start + self.length):
                        img_filename = data['image_paths'].format(data['image_indices'][i])
                        img = Image.open(img_filename)
                        img = self.transform(img)
                        video.append(img)
                        if self.action:
                            actions.append(data['actions'][i-1] if i > 0 else [0. for _ in range(6)] + [data['actions'][0][-1]])
                break
            except:
                print('Missing image: ' + data['image_paths'].format(data['image_indices'][0]))
                start = torch.randint(-1, data['frame_number'] - self.length + 1, (1,)).item() # resample
        
        if self.action and self.mask_action:
            mask_indices = torch.randperm(self.length)[:int(self.length * self.mask_action_ratio)]
            actions_masked = copy.deepcopy(actions)
            for i in mask_indices:
                actions_masked[i] = [0. for _ in range(7)]
        
        # normalize the actions
        if self.normalize:
            if self.action:
                actions = torch.tensor(actions)
                actions = (actions - mean) / std
                if self.mask_action:
                    actions_masked = torch.tensor(actions_masked)
                    actions_masked = (actions_masked - mean) / std


        ret = {}
        ret['video'] = torch.stack(video).permute(1, 0, 2, 3) # (C, T, H, W)
        if self.action:
            ret['actions'] = actions # (T, 7), 7 is the number of action dimension
            if self.mask_action:
                ret['actions_masked'] = actions_masked
        if self.return_mean_std:
            ret['mean'] = mean
            ret['std'] = std
        return ret

def get_image_action_dataloader(args, split='train', action=False, return_mean_std=False):
    dataset = ImageActionDataset(args, split=split, action=action, return_mean_std=return_mean_std)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                             num_workers=args.num_workers,
                                             shuffle=True if split == 'train' else False)
    return dataloader


def get_image_action_dataloader_width(args, split='train', action=False, return_mean_std=False):
    dataset = ImageActionDatasetGripperWidth(args, split=split, action=action, return_mean_std=return_mean_std)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                             num_workers=args.num_workers,
                                             shuffle=True if split == 'train' else False)
    return dataloader


        

