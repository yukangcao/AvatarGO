import math
import numpy as np

import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from sh_utils import eval_sh, SH2RGB, RGB2SH

from gaussian_model_rst import GaussianModel, BasicPointCloud

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, time=0, gs_convention=True):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        w2c = np.linalg.inv(c2w)

        if gs_convention:
            # rectify...
            w2c[1:3, :3] *= -1
            w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

        self.time = time


class Renderer:
    def __init__(self, sh_degree=3, white_background=True, radius=1):
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        from arguments import ModelHiddenParams
        hyper = ModelHiddenParams(None) # args
        self.gaussians = GaussianModel(sh_degree, hyper)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
    
    def initialize(self, input=None, num_pts=5000, radius=0.5):
        # load checkpoint
        if input is None:
            # init from random point cloud
            
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            # self.gaussians.create_from_pcd(pcd, 10)
            self.gaussians.create_from_pcd(pcd, 10, 1)
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.gaussians.create_from_pcd(input, 1, 1)
        else:
            # load from saved ply
            self.gaussians.load_ply(input)

    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        t_vertices=None, face_vertices=None, pose_params=None,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        time = (time.float()  - 4) / 10 # hack
        # time = None

        means2D = screenspace_points
        opacity = self.gaussians._opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians._scaling
            rotations = self.gaussians._rotation

        means3D_deform, scales_deform, rotations_deform, opacity_deform = self.gaussians._deformation(means3D.cuda(), scales.cuda(), 
                                                            rotations.cuda(), opacity.cuda(),
                                                            time.cuda(), t_vertices.cuda(), face_vertices.cuda(), pose_params) #  time is not none

        means3D_final = torch.zeros_like(means3D)
        rotations_final = torch.zeros_like(rotations)
        scales_final = torch.zeros_like(scales)
        opacity_final = torch.zeros_like(opacity)
        means3D_final =  means3D_deform
        rotations_final =  rotations_deform
        scales_final =  scales_deform
        opacity_final = opacity_deform
        

        scales_final = self.gaussians.scaling_activation(scales_final)
        rotations_final = self.gaussians.rotation_activation(rotations_final)
        opacity = self.gaussians.opacity_activation(opacity)

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color


        rendered_image, radii, rendered_depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }



    def comp_render(
        self,
        gaussian_item,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        t_vertices=None, face_vertices=None, pose_params=None,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        time = (time.float()  - 4) / 10 # hack
        # time = None

        means2D = screenspace_points
        # opacity = self.gaussians.get_opacity
        opacity = self.gaussians._opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians._scaling
            rotations = self.gaussians._rotation
        means3D_deform, scales_deform, rotations_deform, opacity_deform = self.gaussians._deformation(means3D.cuda(), scales.cuda(), 
                                                            rotations.cuda(), opacity.cuda(),
                                                            time.cuda(), t_vertices.cuda(), face_vertices.cuda(), pose_params) #  time is not none
        
        means3D_final = []
        rotations_final = []
        scales_final = []
        opacity_final = []
        
        means3D_final.append(means3D_deform)
        
        scales_final.append(self.gaussians.scaling_activation(scales_deform))
        rotations_final.append(self.gaussians.rotation_activation(rotations_deform))
        opacity_final.append(self.gaussians.opacity_activation(opacity_deform))



        _xyz, _rotation, _scaling = gaussian_item.transform_gaussian()
        means3D_item = _xyz
        screenspace_points_item = (
            torch.zeros_like(
                _xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass
    
        means2D_item = screenspace_points_item
        opacity_item = gaussian_item._opacity

        scales_item = _scaling
        rotations_item = _rotation

        time_item = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D_item.shape[0],1)
        time_item = (time_item.float()  - 4) / 10 # hack
        means3D_deform_item, scales_deform_item, rotations_deform_item, opacity_deform_item = gaussian_item._deformation(means3D_item.cuda(), scales_item.cuda(), 
                                                            rotations_item.cuda(), opacity_item.cuda(),
                                                            time_item.cuda(), t_vertices.cuda(), face_vertices.cuda(), pose_params) #  time is not none

        means3D_final.append(means3D_deform_item)
        scales_final.append(gaussian_item.scaling_activation(scales_deform_item))
        rotations_final.append(gaussian_item.rotation_activation(rotations_deform_item))
        opacity_final.append(gaussian_item.opacity_activation(opacity_deform_item))

        means3D_final = torch.cat(means3D_final, dim=0)
        scales_final = torch.cat(scales_final, dim=0)
        rotations_final = torch.cat(rotations_final, dim=0)
        opacity_final = torch.cat(opacity_final, dim=0)
        
        means2D_final = torch.cat([means2D, means2D_item], dim=0)
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color

        rendered_image, radii, rendered_depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D_final,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
        }
