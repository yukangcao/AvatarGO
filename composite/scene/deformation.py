import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import torch.nn.init as init
from scene.hexplane import HexPlaneField

# from knn_cuda import KNN
# from nets.mlp_delta_body_pose import BodyPoseRefiner
# from nets.mlp_delta_weight_lbs import LBSOffsetDecoder

def save_obj(mesh_path, verts):
    file = open(mesh_path, 'w')    
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    file.close()
    
def SMPL_to_tensor(params, device):
    key_ = ['v_template', 'shapedirs', 'J_regressor', 'kintree_table', 'f', 'weights', "posedirs"]
    for key1 in key_:
        if key1 == 'J_regressor':
            if isinstance(params[key1], np.ndarray):
                params[key1] = torch.tensor(params[key1].astype(float), dtype=torch.float32, device=device)
            else:
                params[key1] = torch.tensor(params[key1].toarray().astype(float), dtype=torch.float32, device=device)
        elif key1 == 'kintree_table' or key1 == 'f':
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.long, device=device)
        else:
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.float32, device=device)
    return params

def get_rigid_transformation_torch(rot_mats, joints, parents):
    """
    rot_mats: bs x 24 x 3 x 3
    joints: bs x 24 x 3
    parents: 24
    """
    # obtain the relative joints
    bs, joints_num = joints.shape[0:2]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # create the transformation matrix
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)
    padding = torch.zeros([bs, joints_num, 1, 4], device=rot_mats.device)  #.to(rot_mats.device)
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=-2)

    # rotate each part
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)

    # obtain the rigid transformation
    padding = torch.zeros([bs, joints_num, 1], device=rot_mats.device)  #.to(rot_mats.device)
    joints_homogen = torch.cat([joints, padding], dim=-1)
    rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms

def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), device=poses.device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.reshape([batch_size, 3, 3])

    ident = torch.eye(3)[None].to(poses.device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat

def get_transform_params_torch(smpl, params, rot_mats=None, correct_Rs=None):
    """ obtain the transformation parameters for linear blend skinning
    """
    v_template = smpl['v_template']

    # add shape blend shapes
    shapedirs = smpl['shapedirs']
    betas = params['shapes']
    # v_shaped = v_template[None] + torch.sum(shapedirs[None] * betas[:,None], axis=-1).float()
    v_shaped = v_template[None] + torch.sum(shapedirs[None][...,:betas.shape[-1]] * betas[:,None], axis=-1).float()

    if rot_mats is None:
        # add pose blend shapes
        poses = params['poses'].reshape(-1, 3)
        # bs x 24 x 3 x 3
        rot_mats = batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)

        if correct_Rs is not None:
            rot_mats_no_root = rot_mats[:, 1:]
            rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, rot_mats.shape[1]-1, 3, 3)
            rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

    # obtain the joints
    joints = torch.matmul(smpl['J_regressor'][None], v_shaped) # [bs, 24 ,3]

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0]
    A = get_rigid_transformation_torch(rot_mats, joints, parents)

    # apply global transformation
    R = params['R'] 
    Th = params['Th'] 

    return A, R, Th, joints

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat
class Linear_Res(nn.Module):
    def __init__(self, W):
        super(Linear_Res, self).__init__()
        self.main_stream = nn.Linear(W, W)

    def forward(self, x):
        x = F.relu(x)
        return x + self.main_stream(x)


class Feat_Res_Net(nn.Module):
    def __init__(self, W, D):
        super(Feat_Res_Net, self).__init__()
        self.D = D
        self.W = W
    
        self.feature_out = [Linear_Res(self.W)]
        for i in range(self.D-2):
            self.feature_out.append(Linear_Res(self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
    
    def initialize_weights(self,):
        for m in self.feature_out.modules():
            if isinstance(m, nn.Linear):
                init.constant_(m.weight, 0)
                # init.xavier_uniform_(m.weight,gain=1)
                if m.bias is not None:
                    # init.xavier_uniform_(m.bias,gain=1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.feature_out(x)


class Head_Res_Net(nn.Module):
    def __init__(self, W, H):
        super(Head_Res_Net, self).__init__()
        self.W = W
        self.H = H

        self.feature_out = [Linear_Res(self.W)]
        self.feature_out.append(nn.Linear(W, self.H))
        self.feature_out = nn.Sequential(*self.feature_out)
    
    def initialize_weights(self,):
        for m in self.feature_out.modules():
            if isinstance(m, nn.Linear):
                init.constant_(m.weight, 0)
                # init.xavier_uniform_(m.weight,gain=1)
                if m.bias is not None:
                    # init.xavier_uniform_(m.bias,gain=1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.feature_out(x)



class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=[], args=None, use_res=False):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips

        self.no_grid = args.no_grid # False
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)

        self.use_res = use_res
        if not self.use_res:
            self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_net()
        else:
            self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_res_net()
        self.args = args
        
        
        # neutral_smplx_path = os.path.join('/mnt/petrelfs/caoyukang/ykcao/test/layout/smplx/models/smplx/SMPLX_NEUTRAL.npz')
        # params_init = dict(np.load(neutral_smplx_path, allow_pickle=True))
        # self.smplx_neutral = SMPL_to_tensor(params_init, device = "cuda")
        
        # self.knn = KNN(k=1, transpose_mode=True)
        # self.knn_near_2 = KNN(k=2, transpose_mode=True)
        
        # self.motion_offset_flag = True
        
        # if self.motion_offset_flag:
            
        #     total_bones = self.smplx_neutral['weights'].shape[-1]
        #     self.pose_decoder = BodyPoseRefiner(total_bones=total_bones, embedding_size=3*(total_bones-1), mlp_width=128, mlp_depth=2)
        #     self.pose_decoder.to("cuda")

        #     # load lbs weight module
        #     self.lweight_offset_decoder = LBSOffsetDecoder(total_bones=total_bones)
        #     self.lweight_offset_decoder.to("cuda")
    
    def create_net(self):
        
        mlp_out_dim = 0
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        output_dim = self.W
        return  \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
    
    def create_res_net(self,):
        
        mlp_out_dim = 0

        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)

        # self.feature_in = nn.Linear(mlp_out_dim + self.grid.feat_dim ,self.W)
        # self.feature_out = Feat_Res_Net(self.W, self.D)

        output_dim = self.W
        return  \
            Head_Res_Net(self.W, 3), \
            Head_Res_Net(self.W, 3), \
            Head_Res_Net(self.W, 4), \
            Head_Res_Net(self.W, 1) 

    
    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_emb):

        if not self.use_res:
            if self.no_grid:
                h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
            else:
                grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])

                h = grid_feature
            
            h = self.feature_out(h)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # h =  self.feature_out(self.feature_in(grid_feature))
            h = self.feature_out(grid_feature)
        return h

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None, time_emb=None, t_vertices=None, pose_params=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, time_emb):
        
        '''
        bs = 1
        batch_size = bs
        
        query_pts = rays_pts_emb.unsqueeze(0)
        smpl_pts = t_vertices.unsqueeze(0)
        
        # pose offset
        dst_posevec = pose_params['poses'][:, 3:]
        pose_out = self.pose_decoder(dst_posevec)
        correct_Rs = pose_out['Rs']

        # SMPL lbs weights
        lbs_weights = self.lweight_offset_decoder(query_pts)
        lbs_weights = lbs_weights.permute(0,2,1)
        
        
        joints_num = self.smplx_neutral['weights'].shape[-1]
        vertices_num = t_vertices.shape[0]
        
        
        _, vert_ids = self.knn(smpl_pts.float(), query_pts.float())
        bweights = self.smplx_neutral['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)
        
        # A, R, Th, joints = get_transform_params_torch(self.smplx_neutral, pose_params)
        
        # A = torch.matmul(bweights, A.reshape(bs, joints_num, -1))
        # A = torch.reshape(A, (bs, -1, 4, 4))
        # query_pts = query_pts - A[..., :3, 3]
        # query_pts = torch.matmul(R, query_pts[..., None]).squeeze(-1)
        
        # transforms = R
        translation = None
        
        # posedirs = self.smplx_neutral['posedirs'].cuda().float()
        # pose_ = pose_params['poses']
        # ident = torch.eye(3).cuda().float()
        # batch_size = bs
        # rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
        # pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])#.cuda()
        # pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
        # pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
        # query_pts = query_pts - pose_offsets

        # From mean shape to normal shape
        shapedirs = self.smplx_neutral['shapedirs'][..., :pose_params['shapes'].shape[-1]]#.cuda()
        shape_offset = torch.matmul(shapedirs.unsqueeze(0), torch.reshape(pose_params['shapes'].cuda(), (batch_size, 1, -1, 1))).squeeze(-1)
        shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
        query_pts = query_pts + shape_offset

        posedirs = self.smplx_neutral['posedirs']#.cuda().float()
        pose_ = pose_params['poses']
        ident = torch.eye(3).cuda().float()
        rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3]).cuda()

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])#.cuda()
        pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
        pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
        query_pts = query_pts + pose_offsets


        # get tar_pts, smpl space source pose
        A, R, Th, joints = get_transform_params_torch(self.smplx_neutral, pose_params, rot_mats=rot_mats)

        self.s_A = A
        A = torch.matmul(bweights, self.s_A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = torch.matmul(A[..., :3, :3], query_pts[..., None]).squeeze(-1)
        smpl_src_pts = can_pts + A[..., :3, 3]
        
        print(smpl_src_pts.shape)
        exit(1)
        # transforms = torch.matmul(A[..., :3, :3], transforms)

        # transform points from the smpl space to the world space
        # R_inv = torch.inverse(R)
        # world_src_pts = torch.matmul(smpl_src_pts, R_inv) + Th
        '''
        
        
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_emb).float()
        dx = self.pos_deform(hidden)
        pts = rays_pts_emb[:, :3] + dx
        if self.args.no_ds: # False
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = scales_emb[:,:3] + ds
        if self.args.no_dr: # False
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = rotations_emb[:,:4] + dr
        if self.args.no_do: # True
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
            opacity = opacity_emb[:,:1] + do
        # + do
        # print("deformation value:","pts:",torch.abs(dx).mean(),"rotation:",torch.abs(dr).mean())

        return pts, scales, rotations, opacity
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        return list(self.grid.parameters() ) 
    # + list(self.timegrid.parameters())


class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        
        self.use_res = args.use_res
        if self.use_res:
            print("Using zero-init and residual")
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(4+3)+((4+3)*scale_rotation_pe)*2, input_ch_time=timenet_output, args=args, use_res=self.use_res)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)

        if self.use_res:
            # self.deformation_net.feature_out.initialize_weights()
            self.deformation_net.pos_deform.initialize_weights()
            self.deformation_net.scales_deform.initialize_weights()
            self.deformation_net.rotations_deform.initialize_weights()
            self.deformation_net.opacity_deform.initialize_weights()

        # self.deformation_net.feature_out.apply(initialize_zeros_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None, t_vertices=None, pose_params=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, times_sel)
        else:
            return self.forward_static(point)

        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, times_sel=None, t_vertices=None, pose_params=None):
        # times_emb = poc_fre(times_sel, self.time_poc)

        means3D, scales, rotations, opacity = self.deformation_net( point,
                                                  scales,
                                                rotations,
                                                opacity,
                                                # times_feature,
                                                times_sel)
        return means3D, scales, rotations, opacity
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.xavier_uniform_(m.bias,gain=1)
            # init.constant_(m.bias, 0)

def initialize_zeros_weights(m):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 0)
        # init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            # init.xavier_uniform_(m.bias,gain=1)
            init.constant_(m.bias, 0)
