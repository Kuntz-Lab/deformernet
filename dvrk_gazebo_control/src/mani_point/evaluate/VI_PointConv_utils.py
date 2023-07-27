import open3d as o3d
import argparse
import sys
import os
import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import datetime
import logging
from sklearn.neighbors import KDTree

sys.path.append('/home/baothach/VI_PointConv/')
from model_architecture import VI_PointConv, VI_PointConv_Double
from pathlib import Path


from easydict import EasyDict as edict
import yaml
import pickle
import open3d

def get_default_configs(cfg):
    cfg.num_level = len(cfg.grid_size)
    cfg.feat_dim  = [cfg.base_dim * (i + 1) for i in range(cfg.num_level)]
    return cfg

def compute_knn(ref_points, query_points, K, dialated_rate = 1):
    num_ref_points = ref_points.shape[0]

    if num_ref_points < K or num_ref_points < dialated_rate * K:
        num_query_points = query_points.shape[0]
        inds = np.random.choice(num_ref_points, (num_query_points, K)).astype(np.int32)

        return inds

    #start_t = time.time()
    kdt = KDTree(ref_points)
    neighbors_idx = kdt.query(query_points, k = K * dialated_rate, return_distance=False)
    #print("num_points:", num_ref_points, "time:", time.time() - start_t)
    neighbors_idx = neighbors_idx[:, ::dialated_rate]

    return neighbors_idx

def parse_args():
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--test_reps', type=int, default=3, help='number of data loading workers')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--model_name', type=str, default='pointconv_res', help='Name of model')

    return parser.parse_args()

def tensorlizeList(nplist, cfg, is_index = False):
    ret_list = []
    for i in range(len(nplist)):
        if is_index:
            if nplist[i] is None:
                ret_list.append(None)
            else:
                ret_list.append(torch.from_numpy(nplist[i]).long().to(cfg.device).unsqueeze(0))
        else:
            ret_list.append(torch.from_numpy(nplist[i]).float().to(cfg.device).unsqueeze(0))

    return ret_list

def tensorlize(pointclouds, edges_self, edges_forward, edges_propagate, feat, norms, cfg):
    pointclouds = tensorlizeList(pointclouds, cfg)
    norms = tensorlizeList(norms, cfg)
    edges_self = tensorlizeList(edges_self, cfg, True)
    edges_forward = tensorlizeList(edges_forward, cfg, True)
    edges_propagate = tensorlizeList(edges_propagate, cfg, True)

    feat = torch.from_numpy(feat).to(cfg.device).unsqueeze(0)

    return pointclouds, edges_self, edges_forward, edges_propagate, feat, norms

def processData(points, feat, norm, cfg):
    point_list, color_list, norm_List = [points.astype(np.float32)], [feat.astype(np.float32)], [norm.astype(np.float32)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_list[0])
    pcd.colors = o3d.utility.Vector3dVector(color_list[0])
    pcd.normals = o3d.utility.Vector3dVector(norm_List[0])

    nself = compute_knn(point_list[0], point_list[0], cfg.K_self)
    nei_forward_list, nei_propagate_list, nei_self_list = [], [], [nself]

    for j, grid_s in enumerate(cfg.grid_size):
        #sub_point, sub_color = grid_subsampling(point_list[-1], color_list[-1], sampleDl = grid_s)
        pcd = pcd.voxel_down_sample(voxel_size=cfg.grid_size[j])
        sub_point = np.asarray(pcd.points)
        sub_color = np.asarray(pcd.colors)
        sub_norm  = np.asarray(pcd.normals)

        nforward = compute_knn(point_list[j], sub_point, cfg.K_forward)
        npropagate = compute_knn(sub_point, point_list[j], cfg.K_propagate)

        if cfg.use_ASPP and j == len(cfg.grid_size) - 1:
            nself = compute_knn(sub_point, sub_point, 8 * cfg.K_self)
        else:
            nself = compute_knn(sub_point, sub_point, cfg.K_self)

        nei_forward_list.append(nforward)
        nei_propagate_list.append(npropagate)
        nei_self_list.append(nself)

        point_list.append(sub_point)
        color_list.append(sub_color)
        norm_List.append(sub_norm)

    return point_list, color_list, norm_List, nei_forward_list, nei_propagate_list, nei_self_list

def get_test_point_clouds(path):
    test_instances = []
    test_files = glob.glob(path)
    for x in torch.utils.data.DataLoader(
            test_files,
            collate_fn=lambda x: torch.load(x[0]), num_workers=5):
        test_instances.append(x)
    print('test instances:', len(test_instances))
    return test_instances, test_files

def get_norms(pc, color):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)
    pcd.colors = open3d.utility.Vector3dVector(color)
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30), fast_normal_computation=False)
    pcd.orient_normals_towards_camera_location(camera_location=([0., 0., 0.]))
    norms = np.asarray(pcd.normals)
    return norms

def get_mp_VI_PointConv(partial_init_pc, partial_goal_pc, gt_mp, vis=False):
    config_file_path = '/home/baothach/VI_PointConv/config.yaml'
    cfg = edict(yaml.safe_load(open(config_file_path, 'r')))
    cfg = get_default_configs(cfg)
    numOfRounds = 3
    device = cfg.device

    model = VI_PointConv_Double(cfg)
    # cfg.pretrain = './weights.pth'

    pretrain_path = '/home/baothach/VI_PointConv/scannet_pointConv_099_0.4778.pth'
    model.load_state_dict(torch.load(pretrain_path))
    print('load model %s'%cfg.pretrain)

    model = model.to(device)
    model.eval()

    coord = partial_init_pc
    coord_2 = partial_goal_pc
    color = np.zeros(partial_init_pc.shape)
    color_2 = color
    norms = get_norms(partial_init_pc, color)
    norms_2 = get_norms(partial_goal_pc, color)
    num_points = coord.shape[0]

    for _ in range(numOfRounds):

        point_idxs = []
        if num_points > cfg.MAX_POINTS_NUM:
            point_idx = np.arange(num_points)
            np.random.shuffle(point_idx)
            left_points = num_points
            while left_points > 0:
                if left_points < cfg.MAX_POINTS_NUM:
                    point_idxs.append(point_idx[-cfg.MAX_POINTS_NUM:])
                else:
                    start_idx = num_points - left_points
                    point_idxs.append(point_idx[start_idx:start_idx + cfg.MAX_POINTS_NUM])
                left_points -= cfg.MAX_POINTS_NUM
        else:
            point_idxs.append(np.arange(num_points))

        m=np.eye(3)
        m[0][0]*=np.random.randint(0,2)*2-1
        theta=np.random.rand()*2*math.pi
        m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
        coord=np.matmul(coord,m)
        norms=np.matmul(norms,m)

        #####################################
        point_idxs_2 = []
        if num_points > cfg.MAX_POINTS_NUM:
            point_idx = np.arange(num_points)
            np.random.shuffle(point_idx)
            left_points = num_points
            while left_points > 0:
                if left_points < cfg.MAX_POINTS_NUM:
                    point_idxs.append(point_idx[-cfg.MAX_POINTS_NUM:])
                else:
                    start_idx = num_points - left_points
                    point_idxs.append(point_idx[start_idx:start_idx + cfg.MAX_POINTS_NUM])
                left_points -= cfg.MAX_POINTS_NUM
        else:
            point_idxs_2.append(np.arange(num_points))

        for (point_idx, point_idx_2)  in zip(point_idxs, point_idxs_2):
            points = coord[point_idx, ...]
            feat = color[point_idx, ...]
            norm = norms[point_idx, ...]

            points_2 = coord_2[point_idx_2, ...]
            feat_2 = color_2[point_idx_2, ...]
            norm_2 = norms_2[point_idx_2, ...]


            point_list, color_list, norm_List, nei_forward_list, nei_propagate_list, nei_self_list = processData(points, feat, norm, cfg)
            pointclouds, edges_self, edges_forward, edges_propagate, feat, surnorm = tensorlize(point_list, nei_self_list, nei_forward_list, nei_propagate_list, color_list[0], norm_List, cfg)

            point_list_2, color_list_2, norm_List_2, nei_forward_list_2, nei_propagate_list_2, nei_self_list_2 = processData(points_2, feat_2, norm_2, cfg)
            pointclouds_2, edges_self_2, edges_forward_2, edges_propagate_2, feat_2, surnorm_2 = tensorlize(point_list_2, nei_self_list_2, nei_forward_list_2, nei_propagate_list_2, color_list_2[0], norm_List_2, cfg)


            with torch.no_grad():
                # pred = model(feat, pointclouds, edges_self, edges_forward, edges_propagate, surnorm)
                pred = model(feat, pointclouds, edges_self, edges_forward, edges_propagate, surnorm, 
                            feat_2, pointclouds_2, edges_self_2, edges_forward_2, edges_propagate_2, surnorm_2)
                # pred_prob = F.softmax(pred.squeeze(0), dim = -1)
                

        # print("pred_prob.shape:", pred_prob.shape)
        # success_probs = pred_prob.cpu().detach().numpy()[:,1]
        success_probs = np.exp(pred.squeeze().cpu().detach().numpy())[:,1]
        print("total candidates:", sum([1 if s>0.5 else 0 for s in success_probs]))
        # np.set_printoptions(threshold=sys.maxsize)
        # print("success_probs:", success_probs)
        print("success_probs.shape:", success_probs.shape)
        print("max(success_probs):", max(success_probs))
        best_mp = partial_init_pc[np.argmax(success_probs)]



        
        if vis:
            success_probs = (success_probs/max(success_probs))
            heats = np.array([[prob, 0, 0] for prob in success_probs])        

            pc = partial_init_pc
            pc_goal = partial_goal_pc
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            pcd_goal = o3d.geometry.PointCloud()
            pcd_goal.points = o3d.utility.Vector3dVector(pc_goal)

            
            best_mani_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            best_mani_point.paint_uniform_color([1,0,0])  
            
            colors = heats  
            pcd.colors =  o3d.utility.Vector3dVector(colors) 
            
            mani_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mani_point.paint_uniform_color([0,0,1])
            # o3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(gt_mp)), pcd_goal.translate((0.2,0,0))]) 
            # o3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(gt_mp)), \
            #                                         pcd_goal.translate((0.2,0,0))])         
            # o3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(gt_mp)), best_mani_point.translate(tuple(best_mp)), \
            #                                         pcd_goal.translate((0.2,0,0)), o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)])
        

            candidate_idxs = np.where(success_probs > 0.7)[0]
            candidates = partial_init_pc[candidate_idxs]

            vis_list = [pcd, mani_point.translate(tuple(gt_mp)), best_mani_point.translate(tuple(best_mp)), \
                                                    pcd_goal.translate((0.2,0,0)), o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]
            for can in candidates:
                candidate = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                candidate.paint_uniform_color([0,1,0])    
                vis_list.append(candidate.translate(tuple(can)))  
            o3d.visualization.draw_geometries(vis_list)          

        return best_mp