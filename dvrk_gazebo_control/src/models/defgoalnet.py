import torch.nn as nn
import torch
import torch.nn.functional as F
from .pointconv_util_groupnorm import PointConvDensitySetAbstraction


class DefGoalNet(nn.Module):
    def __init__(self, num_points, embedding_size=256):
        super(DefGoalNet, self).__init__()
        self.chamfer_loss = ChamferLoss()

        point_dim = 3

        self.num_points = num_points
        self.embedding_size = embedding_size
        
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=point_dim + 3, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[embedding_size], bandwidth = 0.4, group_all=True)
        
                 
        self.sa1_kidney = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=point_dim + 3, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2_kidney = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3_kidney = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[embedding_size], bandwidth = 0.4, group_all=True)

        
        self.fc1 = nn.Linear(in_features=self.embedding_size*2, out_features=self.embedding_size)
        self.fc2 = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size)
        self.fc3 = nn.Linear(in_features=self.embedding_size, out_features=self.num_points*3)

        self.fc_bn1 = nn.GroupNorm(1, self.embedding_size)
        self.fc_bn2 = nn.GroupNorm(1, self.embedding_size)


    def forward(self, tissue_pc, kidney_pc):
        batch_size = tissue_pc.shape[0]
        point_dim = tissue_pc.shape[1]

        # Encoder for tissue point cloud
        x, l1_points = self.sa1(tissue_pc, tissue_pc)
        x, l2_points = self.sa2(x, l1_points)
        x, l3_points = self.sa3(x, l2_points)
        
        x = l3_points.view(batch_size, self.embedding_size)
        

        # Encoder for kidney point cloud
        x_kidney, l1_points_kidney = self.sa1_kidney(kidney_pc, kidney_pc)
        x_kidney, l2_points_kidney = self.sa2_kidney(x_kidney, l1_points_kidney)
        x_kidney, l3_points_kidney = self.sa3_kidney(x_kidney, l2_points_kidney)
        
        x_kidney = l3_points_kidney.view(batch_size, self.embedding_size)


        # Concatenate the tissue and kidney embeddings
        x = torch.cat((x, x_kidney), dim=-1)
        

        # Decoder
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.relu(self.fc_bn2(self.fc2(x)))
        goal_pc = self.fc3(x)

        # Reshape the output
        goal_pc = goal_pc.reshape(batch_size, point_dim, self.num_points)
        


        return goal_pc

    def get_chamfer_loss(self, input, output):
        # input shape  (batch_size, num_pts, 3)
        # output shape (batch_size, num_pts, 3)
        return self.chamfer_loss(input, output)


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2






