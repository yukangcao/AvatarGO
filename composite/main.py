import os
import cv2
import time
import tqdm
import numpy as np

import torch
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer_rst import Renderer, MiniCam
from argparse import ArgumentParser, Namespace
from gaussiansplatting.arguments import PipelineParams

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize

import copy

from scene.skeleton import Skeleton
from utils.graphics_utils import BasicPointCloud

import threestudio
from threestudio.utils.config import ExperimentConfig, load_config

import copy

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

import sys
sys.path.append('./composite')
os.environ['HF_HOME'] = '/mnt/sfs-common/ykcao/cache'

def init_from_pointe(prompt):
    def generate(prompt):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('creating base model...')
        base_name = 'base40M-textvec'
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
        print('creating upsample model...')
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
        print('downloading base checkpoint...')
        base_model.load_state_dict(load_checkpoint(base_name, device))
        print('downloading upsampler checkpoint...')
        upsampler_model.load_state_dict(load_checkpoint('upsample', device))
        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
        )
        # Produce a sample from the model.
        from tqdm import tqdm
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
            samples = x

        pc = sampler.output_to_point_clouds(samples)[0]
        xyz = pc.coords
        return xyz
    
    num_pts = int(10000/5000)
    xyz = generate(prompt)
    xyz[:, [1, 2]] = xyz[:, [2, 1]]
    
    thetas = np.random.rand(num_pts)*np.pi
    phis = np.random.rand(num_pts)*2*np.pi        
    radius = np.random.rand(num_pts)*0.05
    # We create random points inside the bounds of sphere
    xyz_ball = np.stack([
        radius * np.sin(thetas) * np.sin(phis),
        radius * np.sin(thetas) * np.cos(phis),
        radius * np.cos(thetas),
    ], axis=-1) # [B, 3]expend_dims
    
    xyz = (np.expand_dims(xyz,axis=1)+np.expand_dims(xyz_ball,axis=0)).reshape(-1,3)
    return xyz


def get_projection_matrix(fovy, aspect_wh, near, far):
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx

def get_mvp_matrix(c2w, proj_mtx):
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx


class GUI:
    def __init__(self, opt, cfg):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.cfg = cfg
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        # self.seed = "random"
        self.seed = 888

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None
        self.guidance_svd = None


        self.enable_sd = False
        self.enable_zero123 = False
        self.enable_svd = False


        # renderer
        self.renderer_human = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1
        self.renderer_item = Renderer(sh_degree=self.opt.sh_degree)
        self.renderer_combine = Renderer(sh_degree=self.opt.sh_degree)

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        self.input_img_list = None
        self.input_mask_list = None
        self.input_img_torch_list = None
        self.input_mask_torch_list = None

        # input text
        self.prompt = ""
        self.negative_prompt = ""
        self.combine_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        self.motion_skeleton_list = []
        # skeleton
        self.motion_path = '/mnt/sfs-common/ykcao/avatargo/stage3/poses/-03_L_1_stageii.npz'
        self.apose_skel = Skeleton(self.motion_path)
        self.apose_skel.load_smplx('./data/smpl_related/models', time_step=-1)
        self.frames_all = self.apose_skel.num_frames
        self.frames = 16
        self.intervals = list(range(0, self.frames_all, self.frames))
        
        #add motion skeleton
        for time in range(self.frames):
            # time = time + 140
            idx = self.intervals[time]
            motion_skeleton = Skeleton(self.motion_path)
            motion_skeleton.load_smplx('./data/smpl_related/models', time_step=idx)
            self.motion_skeleton_list.append(motion_skeleton)
        
        # load input data from cmdline
        if self.opt.input is not None: # True
            self.load_input(self.opt.input, self.opt.input_item) # load imgs, if has bg, then rm bg; or just load imgs
        
        self.renderer_combine.initialize(num_pts=self.opt.num_pts)
        
        
        # override if provide a checkpoint
        if self.opt.load_human is not None: # not None
            self.renderer_human.initialize(self.opt.load_human)             
        else:
            # initialize gaussians to a blob
            self.renderer_human.initialize(num_pts=self.opt.num_pts)
        
        if self.opt.load_item is not None: # not None
            self.renderer_item.initialize(self.opt.load_item)             
        else:
            # initialize gaussians to a blob
            self.renderer_item.initialize(num_pts=self.opt.num_pts)

        self.seed_everything()
        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        print(f'Seed: {seed:d}')
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer_human.gaussians.training_setup(self.opt)
        self.renderer_item.gaussians.training_setup(self.opt)
        self.renderer_combine.gaussians.training_setup(self.opt)

        # do not do progressive sh-level
        self.renderer_human.gaussians.active_sh_degree = self.renderer_human.gaussians.max_sh_degree
        self.optimizer_human = self.renderer_human.gaussians.optimizer
        self.renderer_item.gaussians.active_sh_degree = self.renderer_item.gaussians.max_sh_degree
        self.optimizer_item = self.renderer_item.gaussians.optimizer
        self.renderer_combine.gaussians.active_sh_degree = self.renderer_combine.gaussians.max_sh_degree
        self.optimizer_combine = self.renderer_combine.gaussians.optimizer

        # default camera
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )
        
        
        self.prompt_processor = threestudio.find(self.cfg.system.prompt_processor_type)(self.cfg.system.prompt_processor)
        self.guidance = threestudio.find(self.cfg.system.guidance_type)(self.cfg.system.guidance)
        
        self.prompt_processor_combine = threestudio.find(self.cfg.system.prompt_processor_type_combine)(self.cfg.system.prompt_processor_combine)
        self.guidance_combine = threestudio.find(self.cfg.system.guidance_type_combine)(self.cfg.system.guidance_combine)
        
        self.getInitializationBias(opt.position)
        
        self.combine_gaussians()
        self.enable_sd = self.opt.lambda_sd > 0 and self.item_prompt != "" # False
        self.enable_zero123 = self.opt.lambda_zero123 > 0 # True
        self.enable_svd = self.opt.lambda_svd > 0 and self.input_img is not None # False

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd: # False
            if self.opt.mvdream: # False
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123: # True
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            self.guidance_zero123_human = Zero123(self.device, t_range=[0.02, self.opt.t_max])
            self.guidance_zero123_item = Zero123(self.device, t_range=[0.02, self.opt.t_max])
            print(f"[INFO] loaded zero123!")


        if self.guidance_svd is None and self.enable_svd: # False
            print(f"[INFO] loading SVD...")
            from guidance.svd_utils import StableVideoDiffusion
            self.guidance_svd = StableVideoDiffusion(self.device)
            print(f"[INFO] loaded SVD!")

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
            
            self.input_item_img_torch = torch.from_numpy(self.input_item_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_item_img_torch = F.interpolate(self.input_item_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_item_mask_torch = torch.from_numpy(self.input_item_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_item_mask_torch = F.interpolate(self.input_item_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.item_prompt], [self.negative_prompt])

            if self.enable_zero123:
                c_list, v_list = [], []
                c, v = self.guidance_zero123_human.get_img_embeds(self.input_img_torch)
                for _ in range(self.opt.batch_size_human):
                    c_list.append(c)
                    v_list.append(v)
                self.guidance_zero123_human.embeddings = [torch.cat(c_list, 0), torch.cat(v_list, 0)]
                
                
                c_item_list, v_item_list = [], []
                c_item, v_item = self.guidance_zero123_item.get_img_embeds(self.input_img_torch)
                for _ in range(self.opt.batch_size_human):
                    c_item_list.append(c)
                    v_item_list.append(v)
                self.guidance_zero123_item.embeddings = [torch.cat(c_item_list, 0), torch.cat(v_item_list, 0)]
            
            if self.enable_svd:
                self.guidance_svd.get_img_embeds(self.input_img)

    def getInitializationBias(self, position):
        
        from torchvision.ops import masks_to_boxes
        from diffusers import AutoPipelineForInpainting
        from utils.sam import LangSAMTextSegmentor
        from tqdm import tqdm        
        
        with torch.no_grad():
            self.text_segmentor = LangSAMTextSegmentor().to(torch.device('cuda'))
            self.seg_prompt=position
            render_resolution = 512
            weights = torch.zeros_like(self.renderer_human.gaussians._opacity)
            weights_cnt = torch.zeros_like(self.renderer_human.gaussians._opacity, dtype=torch.int32)
            
            for b_idx in range(self.opt.batch_size):
                time = b_idx
                ver = 0
                hor = 0
                radius = 0
                pose = orbit_camera(ver, hor, self.cam.radius)
                
                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far, time=time)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device=self.device)

                out_human = self.renderer_human.render(cur_cam, bg_color=bg_color, t_vertices=self.apose_skel.vertices.float(), face_vertices=self.apose_skel.faces.float(), pose_params=self.motion_skeleton_list[b_idx].params)

                image_in = torch.clip(out_human["image"], 0.0, 1.0).unsqueeze(0).permute(0, 2, 3, 1)                
                mask = self.text_segmentor(image_in, self.seg_prompt)[0].to(torch.device('cuda'))            
                self.renderer_human.gaussians.apply_weights(cur_cam, weights, weights_cnt, mask)
                
                break
            weights /= weights_cnt + 1e-7
            
            xyz_bias=(weights.T@self.renderer_human.gaussians._xyz)/weights.sum().to(self.device)
            self.renderer_item.gaussians.translate_param = self.renderer_item.gaussians.translate_param.to(self.device) + xyz_bias.reshape(3)
    
    def combine_gaussians(self):
        
        _xyz, _rotation, _scaling = self.renderer_item.gaussians.transform_gaussian()
        self.renderer_combine.gaussians._xyz = _xyz
        self.renderer_combine.gaussians._features_dc = self.renderer_item.gaussians._features_dc.detach().clone()
        self.renderer_combine.gaussians._features_rest = self.renderer_item.gaussians._features_rest.detach().clone()
        self.renderer_combine.gaussians._scaling = _scaling
        self.renderer_combine.gaussians._rotation = _rotation
        self.renderer_combine.gaussians._opacity = self.renderer_item.gaussians._opacity.detach().clone()
        self.renderer_combine.gaussians.max_radii2D = self.renderer_item.gaussians.max_radii2D.detach().clone()
        
        _xyz, _rotation, _scaling = self.renderer_human.gaussians._xyz, self.renderer_human.gaussians._rotation, self.renderer_human.gaussians._scaling
        self.renderer_combine.gaussians._xyz = torch.cat((self.renderer_combine.gaussians._xyz, _xyz.cuda()), dim=0)
        self.renderer_combine.gaussians._features_dc = torch.cat((self.renderer_combine.gaussians._features_dc, self.renderer_human.gaussians._features_dc.detach().clone().cuda()), dim=0)
        self.renderer_combine.gaussians._features_rest = torch.cat((self.renderer_combine.gaussians._features_rest, self.renderer_human.gaussians._features_rest.detach().clone().cuda()), dim=0)
        self.renderer_combine.gaussians._scaling = torch.cat((self.renderer_combine.gaussians._scaling, _scaling.cuda()), dim=0)
        self.renderer_combine.gaussians._rotation = torch.cat((self.renderer_combine.gaussians._rotation, _rotation.cuda()), dim=0)
        self.renderer_combine.gaussians._opacity = torch.cat((self.renderer_combine.gaussians._opacity, self.renderer_human.gaussians._opacity.detach().clone().cuda()), dim=0)
        self.renderer_combine.gaussians.max_radii2D = torch.cat((self.renderer_combine.gaussians.max_radii2D, self.renderer_human.gaussians.max_radii2D.detach().clone().cuda()), dim=0)
    
    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        # self.zero_grad()
        for _ in range(self.train_steps): # 1

            self.step += 1 # self.step starts from 0
            step_ratio = min(1, self.step / self.opt.iters) # 1, step / 500

            # update lr
            self.renderer_human.gaussians.update_learning_rate(self.step)
            self.renderer_item.gaussians.update_learning_rate(self.step)
            self.renderer_combine.gaussians.update_learning_rate(self.step)

            self.combine_gaussians()
            
            loss_human = 0
            loss_item = 0
            loss_combine = 0
            
            
            ### novel view (manual batch)
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            # render_resolution = 512
            images_item = []
            images_item_mv = []
            images_combine = []
            depths_item = []
            
            images_human = []
            depths_human = []
            
            poses = []
            poses_item = []
            vers, hors, radii = [], [], []
            vers_human, hors_human, radii_human = [], [], []
            batch = {'elevation':[], 'azimuth':[], 'camera_distances':[]}
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)
            
            for b_idx in range(self.opt.batch_size_human):
                
                time = b_idx
                
                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers_human.append(ver)
                hors_human.append(hor)
                radii_human.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far, time=time)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device=self.device)
                out_combine = self.renderer_combine.render(cur_cam, bg_color=bg_color, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[b_idx].params)
                image_combine = out_combine["image"].permute(1, 2, 0)
                images_combine.append(image_combine)
                
            images_combine = torch.stack(images_combine, 0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)
            
            
            batch['elevation'] = torch.tensor(vers_human)
            batch['azimuth'] = torch.tensor(hors_human)
            batch['camera_distances'] = torch.tensor(radii_human) + self.opt.radius
            batch['fovy'] = torch.tensor([self.cam.fovy]).repeat(self.opt.batch_size_human)
            batch['c2w'] = torch.tensor(poses)
            proj_mtx= get_projection_matrix(batch['fovy'], self.cam.H / self.cam.W, 0.1, 100.0)
            batch['mvp_mtx'] = get_mvp_matrix(batch['c2w'].to(self.device), proj_mtx.to(self.device))


            prompt_utils_combine = self.prompt_processor_combine()
            guidance_eval = False
            guidance_out = self.guidance_combine(self.opt.attention_word, self.opt.combine_prompt, images_combine, prompt_utils_combine, **batch, 
                                rgb_as_latents=False, guidance_eval=guidance_eval)   
            loss_combine = guidance_out['loss_sds']
    
            # optimize step
            loss_item = loss_item + loss_combine
            loss_human = loss_human + loss_combine
            
            loss_human.backward(retain_graph=True)
            self.optimizer_human.step()
            self.optimizer_human.zero_grad()
            
            loss_combine.backward(retain_graph=True)
            self.optimizer_item.step()
            self.optimizer_item.zero_grad()
            
            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor_human, visibility_filter_human, radii_human = out_human["viewspace_points"], out_human["visibility_filter"], out_human["radii"]
                self.renderer_human.gaussians.max_radii2D[visibility_filter_human] = torch.max(self.renderer_human.gaussians.max_radii2D[visibility_filter_human], radii_human[visibility_filter_human])
                self.renderer_human.gaussians.add_densification_stats(viewspace_point_tensor_human, visibility_filter_human)

                if self.step % self.opt.densification_interval == 0:
                    # size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
                    self.renderer_human.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=0.5, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer_human.gaussians.reset_opacity()
                    
                
                viewspace_point_tensor_item, visibility_filter_item, radii_item = out_item["viewspace_points"], out_item["visibility_filter"], out_item["radii"]
                self.renderer_item.gaussians.max_radii2D[visibility_filter_item] = torch.max(self.renderer_item.gaussians.max_radii2D[visibility_filter_item], radii_item[visibility_filter_item])
                self.renderer_item.gaussians.add_densification_stats(viewspace_point_tensor_item, visibility_filter_item)

                if self.step % self.opt.densification_interval == 0:
                    # size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
                    self.renderer_item.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=0.5, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer_item.gaussians.reset_opacity()
            

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

    
    def load_input(self, file, file_item):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if '_rgba' not in file:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)
            file = file.replace('.png', '_rgba.png')
            cv2.imwrite(file, img) 

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()


        print(f'[INFO] load image from {file_item}...')
        img_item = cv2.imread(file_item, cv2.IMREAD_UNCHANGED)
        if '_rgba' not in file_item:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img_item = rembg.remove(img_item, session=self.bg_remover)
            file_item = file_item.replace('.png', '_rgba.png')
            cv2.imwrite(file_item, img_item) 

        img_item = cv2.resize(img_item, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img_item = img_item.astype(np.float32) / 255.0
 
        self.input_item_mask = img_item[..., 3:]
        # white bg
        self.input_item_img = img_item[..., :3] * self.input_item_mask + (1 - self.input_item_mask)
        # bgr to rgb
        self.input_item_img = self.input_item_img[..., ::-1].copy()
        
    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024, t=0):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = f'logs_composite/{opt.save_path}_mesh_{t:03d}.ply'
            mesh = self.renderer.gaussians.extract_mesh_t(path, self.opt.density_thresh, t=t, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[t].params)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = f'logs_composite/{opt.save_path}_mesh_{t:03d}.obj'
            mesh = self.renderer.gaussians.extract_mesh_t(path, self.opt.density_thresh, t=t, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[t].params)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                    time=t
                )
                
                cur_out = self.renderer.render(cur_cam, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[t].params)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)
        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_human_model.ply')
            self.renderer_human.gaussians.save_ply(path)
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_item_model.ply')
            self.renderer_item.gaussians.save_ply(path, item=True)
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_combine_model.ply')
            self.renderer_combine.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    # no gui mode
    def train(self, iters=500, ui=False):
        image_human_list =[]
        image_item_list =[]
        image_combine_list =[]
        from PIL import Image
        from diffusers.utils import export_to_video, export_to_gif
        
        interval = 1
        nframes = iters // interval # 250
        hor = 180
        delta_hor = 4 * 360 / nframes
        time = 0
        delta_time = 1
        if self.gui:
            from visergui import ViserViewer
            self.viser_gui = ViserViewer(device=self.device, viewer_port=8080)
        if iters > 0:
            self.prepare_train()
            if self.gui:
                self.viser_gui.set_renderer(self.renderer, self.fixed_cam)
            
            for i in tqdm.trange(iters): # 500
                self.train_step()
                if self.gui:
                    self.viser_gui.update()

                if i % interval == 0:
                    pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
                    cur_cam = MiniCam(
                        pose,
                        1024,
                        1024,
                        self.cam.fovy,
                        self.cam.fovx,
                        self.cam.near,
                        self.cam.far,
                        time=time
                    )
                    with torch.no_grad():
                        outputs_human = self.renderer_human.render(cur_cam, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[time].params)
                        outputs_item = self.renderer_item.render(cur_cam, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[time].params)
                        outputs_combine = self.renderer_combine.render(cur_cam, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[time].params)

                    out_human = outputs_human["image"].cpu().detach().numpy().astype(np.float32)
                    out_human = np.transpose(out_human, (1, 2, 0))
                    out_human = Image.fromarray(np.uint8(out_human*255))
                    image_human_list.append(out_human)
                    
                    out_item = outputs_item["image"].cpu().detach().numpy().astype(np.float32)
                    out_item = np.transpose(out_item, (1, 2, 0))
                    out_item = Image.fromarray(np.uint8(out_item*255))
                    image_item_list.append(out_item)
                    
                    image_combine = outputs_combine["image"]
                    
                    
                    image_combine = image_combine.cpu().detach().numpy().astype(np.float32)
                    image_combine = np.transpose(image_combine, (1, 2, 0))
                    image_combine = Image.fromarray(np.uint8(image_combine*255))
                    image_combine_list.append(image_combine)

                    time = (time + delta_time) % self.opt.batch_size_human
                    hor = (hor+delta_hor) % 360
            # final eval
            for _ in range(nframes // 4):
                pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
                cur_cam = MiniCam(
                    pose,
                    1024,
                    1024,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                    time=time
                )
                with torch.no_grad():
                    outputs_human = self.renderer_human.render(cur_cam, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[time].params)
                    outputs_item = self.renderer_item.render(cur_cam, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[time].params)
                    outputs_combine = self.renderer_combine.render(cur_cam, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[time].params)

                out_human = outputs_human["image"].cpu().detach().numpy().astype(np.float32)
                out_human = np.transpose(out_human, (1, 2, 0))
                out_human = Image.fromarray(np.uint8(out_human*255))
                image_human_list.append(out_human)
                    
                out_item = outputs_item["image"].cpu().detach().numpy().astype(np.float32)
                out_item = np.transpose(out_item, (1, 2, 0))
                out_item = Image.fromarray(np.uint8(out_item*255))
                image_item_list.append(out_item)
                    
                image_combine = outputs_combine["image"]
                    
                    
                image_combine = image_combine.cpu().detach().numpy().astype(np.float32)
                image_combine = np.transpose(image_combine, (1, 2, 0))
                image_combine = Image.fromarray(np.uint8(image_combine*255))
                image_combine_list.append(image_combine)

                time = (time + delta_time) % self.opt.batch_size_human
                hor = (hor+delta_hor) % 360
                
            os.makedirs('./vis_data_composite', exist_ok=True)
            os.makedirs('./logs_composite', exist_ok=True)
            
            export_to_gif(image_human_list, f'vis_data_composite/train_{opt.save_path}_human.gif')
            export_to_gif(image_item_list, f'vis_data_composite/train_{opt.save_path}_item.gif')
            export_to_gif(image_combine_list, f'vis_data_composite/train_{opt.save_path}_combine.gif')
            
            # # do a last prune
            # self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        # render eval
        image_human_list =[]
        image_item_list =[]
        image_combine_list =[]
        nframes = 14 *5
        hor = 180
        delta_hor = 360 / nframes
        time = 0
        delta_time = 1
        for _ in range(nframes):
            pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
            cur_cam = MiniCam(
                pose,
                1024,
                1024,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                time=time
            )
            with torch.no_grad():
                outputs_human = self.renderer_human.render(cur_cam, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[time].params)
                outputs_item = self.renderer_item.render(cur_cam, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[time].params)
                outputs_combine = self.renderer_combine.render(cur_cam, t_vertices=self.apose_skel.vertices, face_vertices=self.apose_skel.faces, pose_params=self.motion_skeleton_list[time].params)

            out_human = outputs_human["image"].cpu().detach().numpy().astype(np.float32)
            out_human = np.transpose(out_human, (1, 2, 0))
            out_human = Image.fromarray(np.uint8(out_human*255))
            image_human_list.append(out_human)
                    
            out_item = outputs_item["image"].cpu().detach().numpy().astype(np.float32)
            out_item = np.transpose(out_item, (1, 2, 0))
            out_item = Image.fromarray(np.uint8(out_item*255))
            image_item_list.append(out_item)
                    
            image_combine = outputs_combine["image"]
                    
                    
            image_combine = image_combine.cpu().detach().numpy().astype(np.float32)
            image_combine = np.transpose(image_combine, (1, 2, 0))
            image_combine = Image.fromarray(np.uint8(image_combine*255))
            image_combine_list.append(image_combine)
            

            time = (time + delta_time) % self.opt.batch_size_human
            hor = (hor+delta_hor) % 360

        export_to_gif(image_human_list, f'vis_data_composite/{opt.save_path}_human.gif')
        export_to_gif(image_item_list, f'vis_data_composite/{opt.save_path}_item.gif')
        export_to_gif(image_combine_list, f'vis_data_composite/{opt.save_path}_combine.gif')


        # save
        self.save_model(mode='model')

        if self.gui:
            while True:
                self.viser_gui.update()

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    
    
    parser2 = argparse.ArgumentParser()
    parser2.add_argument("--config", required=True, help="path to config file")
    parser2.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser2.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    parser2.add_argument(
        "--gradio", action="store_true", help="if true, run in gradio mode"
    )

    parser2.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    parser2.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )
    
    args2, extras2 = parser2.parse_known_args(['--config','./configs/dual.yaml', '--train', f'system.prompt_processor.prompt={opt.prompt}', f'system.prompt_processor_combine.prompt={opt.combine_prompt}'])
    
    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args2.config, cli_args=extras2, n_gpus=1)

    gui = GUI(opt, cfg)

    gui.train(opt.iters)
