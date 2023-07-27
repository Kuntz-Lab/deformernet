import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
import pickle



class ShapeServoDataset(Dataset):


    def __init__(self, percentage = 1.0):
        """
        Args:

        """ 
        self.dataset_path = "/home/baothach/shape_servo_data/VFH/batch1/processed"
        # with open(data_processed_path, 'rb') as handle:
        #     self.dataset = pickle.load(handle)
        self.filenames = os.listdir(self.dataset_path)
        
        if percentage != 1.0:
            self.filenames = os.listdir(self.dataset_path)[:int(percentage*len(self.filenames))]


    
    def load_pickle_data(self, filename):
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        sample = self.load_pickle_data(self.filenames[idx])
        


        pc = torch.tensor(sample["VFH_30"][0]).float()    # original partial point cloud
        pc_goal = torch.tensor(sample["VFH_30"][1]).float()        
 
 
        grasp_pose = (torch.tensor(sample["positions"])*1000).unsqueeze(0).float()
        
        sample = {'input': pc_goal - pc, 'target': grasp_pose}        

        return sample  




# class ShapeServoDataset(Dataset):
#     """Shape servo dataset."""

#     def __init__(self, train = True, percentage = 0.8):
#         """
#         Args:

#         """
#         with open('/home/baothach/shape_servo_data/batch_1_shuffled_VFH_30', 'rb') as handle:
#             self.data = pickle.load(handle)         
        
#         if train:
#             idx_for_train = int(percentage*len(self.data["target"]))
#             self.input = self.data["input"][:idx_for_train]
#             self.target = self.data["target"][:idx_for_train]
#         else:
#             idx_for_test = int((1-percentage)*len(self.data["target"]))
#             self.input = self.data["input"][idx_for_test:]
#             self.target = self.data["target"][idx_for_test:]        

#     def __len__(self):
#         return len(self.target)

#     def __getitem__(self, idx):
#         input = torch.from_numpy(np.array(self.input[idx]) - np.array(self.input[0])).float()
#         target = torch.from_numpy(np.array(self.target[idx])*1000 - np.array(self.target[0])*1000).unsqueeze(0).float()

#         input = torch.tensor(self.input[idx]).float()
#         target = (torch.tensor(self.target[idx])*1000).unsqueeze(0).float()
     
#         # print("shape: ", input.shape, target.shape)
#         sample = {'input': input, 'target': target}
#         return sample