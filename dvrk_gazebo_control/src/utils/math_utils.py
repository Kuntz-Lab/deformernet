from scipy.spatial import cKDTree
import numpy as np
import torch
import torch.nn as nn

def chamfer_distance(pointcloud1, pointcloud2):
    """
    Compute the Chamfer distance between two point clouds (numpy).
    """    
    # Ensure pointcloud1 and pointcloud2 are both 2D arrays with shape (num_points, 3)
    if pointcloud1.ndim == 1:
        pointcloud1 = pointcloud1.reshape(-1, 1)
    if pointcloud2.ndim == 1:
        pointcloud2 = pointcloud2.reshape(-1, 1)
    
    tree1 = cKDTree(pointcloud1)
    distances_1to2, _ = tree1.query(pointcloud2)
    
    tree2 = cKDTree(pointcloud2)
    distances_2to1, _ = tree2.query(pointcloud1)
    
    chamfer_dist = np.mean(distances_1to2) + np.mean(distances_2to1)
    
    return chamfer_dist


class ChamferLoss(nn.Module):
    """
    Compute the Chamfer distance between two point clouds (pytorch).
    """
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
        loss_1 = torch.sum(mins, dim=1) / mins.size(1)  # Normalize over number of points in preds

        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins, dim=1) / mins.size(1)  # Normalize over number of points in gts

        return loss_1 + loss_2