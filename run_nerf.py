import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from load_pleno import load_pleno_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
   
    if c2w is not None:
        rays_o, rays_d = get_rays_HC(H, W, K, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)

        norm_viewdirs = torch.norm(viewdirs, dim=-1, keepdim=True)
        norm_viewdirs = torch.where(norm_viewdirs > 1e-6, norm_viewdirs, torch.tensor(1e-6, device=viewdirs.device))
        viewdirs = viewdirs / norm_viewdirs
        viewdirs = torch.reshape(viewdirs, [-1,4]).float()

    sh = rays_d.shape
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,4]).float()
    rays_d = torch.reshape(rays_d, [-1,4]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])

    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict]

def render_path_plenoptic(target_images, i_test, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, precomputed_rays_o=None, precomputed_rays_d=None):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor    
    
    rgbs = []
    disps = []
    psnrs = []

    t = time.time()
    for i, img_i in enumerate(tqdm(i_test)):
        t = time.time()

        if precomputed_rays_o is not None and precomputed_rays_d is None:
            raise ValueError("Les rayons pré-calculés ne sont pas fournis !")
        if i >= len(precomputed_rays_o) or i >= len(precomputed_rays_d):
            raise ValueError(f"L'index {i} est hors des limites des precomputed_rays !")
        
        rays_o, rays_d = precomputed_rays_o[i], precomputed_rays_d[i]
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, rays=(rays_o,rays_d), **render_kwargs)

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if gt_imgs is not None:
            target = target_images[i].to(rgb.device)
            psnr = mse2psnr(img2mse(rgb[...,:3],target[...,:3])).item()
            psnrs.append(psnr)

        if i==0:
            print(f"In render_path_plenoptic rgb.shape : {rgb.shape}, and dips.shape : {disp.shape}")

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            rgb8 = rgb8[...,[2,1,0,3]]
            if rgb8.shape[0] == 1:
                rgb8 = np.squeeze(rgb8, axis=0)
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps, psnrs
  

def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = [] 
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        
        disps.append(disp.cpu().numpy())
        if i==0:
            print(f"In render_path rgb.shape : {rgb.shape}, and dips.shape : {disp.shape}")

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            rgb8 = rgb8[...,[2,1,0,3]]
            filename = os.path.join(savedir, 'rendu_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]   

    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch,
                 output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, 
                          output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################
    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs_v1(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])
    alpha_raw = raw[..., 3]

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(alpha_raw.shape) * raw_noise_std 
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(alpha_raw.shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(alpha_raw + noise, dists)

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    rgba_map = torch.sum(weights[...,None] * rgb, -2)

    alpha_map = torch.sum(weights, -1)

    rgba_map = torch.cat([rgba_map, alpha_map[..., None]], dim=-1)
    
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = alpha_map

    if white_bkgd:
        rgba_map[..., :3] = rgba_map[..., :3] + (1. - rgba_map[..., 3][..., None])

    return rgba_map, disp_map, acc_map, weights, depth_map

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])
    alpha_raw = raw[..., 3]

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(alpha_raw.shape) * raw_noise_std 

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(alpha_raw.shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(alpha_raw + noise, dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    rgba_map = torch.sum(weights[...,None] * rgb, -2)
    alpha_map = torch.sum(weights, -1)

    rgba_map = torch.cat([rgba_map, alpha_map[..., None]], dim=-1)
    
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = alpha_map

    if white_bkgd:
        rgba_map[..., :3] = rgba_map[..., :3] + (1. - rgba_map[..., 3][..., None])
    
    return rgba_map, disp_map, acc_map, weights, depth_map



def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):

    N_rays = ray_batch.shape[0]
    rays_o = ray_batch[:, 0:4]
    rays_d = ray_batch[:, 4:8]
    viewdirs = ray_batch[:, -4:] if ray_batch.shape[-1] > 10 else None    

    bounds = torch.reshape(ray_batch[..., 8:10], [-1, 1, 2])
    near, far = bounds[...,0], bounds[...,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
        

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    raw = network_query_fn(pts, viewdirs, network_fn)

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default= 64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    ## dataset options (add plenoptic)
    parser.add_argument("--dataset_type", type=str, default='plenoptic', 
                        help='options: plenoptic / llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    parser.add_argument("--camera_file", type=str, 
                        help="File containing camera parameters")
    parser.add_argument("--image_file", type=str, 
                        help="File containing image parameters")
    parser.add_argument("--images_folder", type=str, 
                        help="Path to the images folder")
    parser.add_argument("--micro_images_folder", type=str, 
                        help="Folder to store micro-images after slicing")
    parser.add_argument("--test_set", type=str, 
                        help="Folder to store test set micro-images")
    parser.add_argument("--train_set", type=str, 
                        help="Folder to store train set micro-images")
    parser.add_argument("--indices_file", type=str, 
                        help="File to store train and test indexs")

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    K = None
    if args.dataset_type == 'plenoptic':
        images, poses, intrinsic_params, i_test, i_train = load_pleno_data(
            args.camera_file,
            args.image_file, 
            args.images_folder, 
            args.micro_images_folder, 
            args.test_set, 
            args.train_set,
            args.indices_file
        )
        poses = [pose for sublist in poses for pose in sublist]
        render_poses = [poses[i] for i in i_test]
        intrinsic_params['mla_centers'] = [center for sublist in intrinsic_params['mla_centers'] for center in sublist]

        i_val = i_test

        if isinstance(render_poses, list):
            render_poses = np.array([pose.cpu().numpy() for pose in render_poses], dtype=np.float32)

        if isinstance(poses, list):
            poses = np.array([pose.cpu().numpy() for pose in poses], dtype=np.float32)
        
        focal = 0.2
        H,W = images.shape[1:3]
        hwf = np.array([H, W, focal])
        near = 400
        far  = 700
    
    elif args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [9500, 0, 0.5*3840],
            [0, 9500, 0.5*2160],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)
            precomputed_rays_o, precomputed_rays_d = get_rays_plenoptic(
                    intrinsic_params['F'],
                    intrinsic_params['M'], 
                    intrinsic_params['s'], 
                    images,   
                    intrinsic_params['pix'],
                    intrinsic_params['mla_centers'],
                    intrinsic_params['lens_diameter']
                )
            rgbs, _ = render_path_plenoptic(i_test, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor, 
                precomputed_rays_o=precomputed_rays_o, precomputed_rays_d=precomputed_rays_d)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    # use_batching = False
   
    rays_o_all, rays_d_all = [], []
    if use_batching:
        # For random ray batching
        rays_o_all, rays_d_all = get_rays_plenoptic(
                    intrinsic_params['F'],
                    intrinsic_params['M'],
                    intrinsic_params['s'],
                    images,
                    intrinsic_params['pix'],
                    intrinsic_params['mla_centers'],
                    intrinsic_params['lens_diameter'],
                    poses
        )
        device = rays_o_all.device
        images = images.to(device)

        rays_o_all = rays_o_all.view(rays_o_all.shape[0], rays_o_all.shape[1], rays_o_all.shape[2], 4)
        rays_d_all = rays_d_all.view(rays_d_all.shape[0], rays_d_all.shape[1], rays_d_all.shape[2], 4)

        rays = torch.stack([rays_o_all, rays_d_all], 1)
        images = images.view(images.shape[0], 1, images.shape[1], images.shape[2], images.shape[3])
        
        rays_rgb = torch.cat([rays, images], 1)
        rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)
        rays_rgb = torch.stack([rays_rgb[i] for i in i_train], 0)

        rays_rgb = rays_rgb.view(-1, 3, 4)
        rays_rgb = rays_rgb.float()

        rays_rgb = rays_rgb[torch.randperm(rays_rgb.size(0))]

        images = images.squeeze(1)

        print('done')
        i_batch = 0


    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 600000 + 1  # 200000
    print('Begin')
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print(" Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:

            img_i = np.random.choice(i_train.cpu().numpy())
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            rays_o_all, rays_d_all = get_rays_plenoptic(
                intrinsic_params['F'],
                intrinsic_params['M'],
                intrinsic_params['s'],
                images,
                intrinsic_params['pix'],
                intrinsic_params['mla_centers'],
                intrinsic_params['lens_diameter'],
                poses, 
                img_i
            )
            rays_o_all = rays_o_all[0]
            rays_d_all = rays_d_all[0]

            H, W, _ = rays_o_all.shape
            rays_o = rays_o_all.view(H, W, 4)
            rays_d = rays_d_all.view(H, W, 4)

            if i < args.precrop_iters:
                dH = int(H // 2 * args.precrop_frac)
                dW = int(W // 2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                        torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                    ), -1)
            else:
                coords = torch.stack(
                    torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                    -1
                )

            coords = torch.reshape(coords, [-1, 2])
            if N_rand > coords.shape[0]:
                N_rand = coords.shape[0]
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
            select_coords = coords[select_inds].long()
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]

        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()


        img_loss = img2mse(rgb[...,:3],target_s[...,:3])
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)
        msetest = img2mse(rgb[...,:3],target_s[...,:3])
        psnrtest = mse2psnr(msetest)
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if isinstance(args.i_weights, torch.Tensor):
            args.i_weights = args.i_weights.item()

        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                print("Rendering translation video...")
                video_dir = os.path.join(basedir, expname, 'video')
                video_dir_disp = os.path.join(basedir, expname, 'video_disp')
                os.makedirs(video_dir, exist_ok=True)
                os.makedirs(video_dir_disp, exist_ok=True)
                print("Video directory", video_dir)

                render_poses = generate_linear_positions(num_positions=30, step=0.5)
                rgbs, disps = render_path(render_poses, [256,256,250], K, args.chunk, render_kwargs_test, savedir=video_dir)
                rgbs = rgbs[...,[2,1,0,3]]
            moviebase = os.path.join(basedir, expname, '{}_translation_x_{:06d}_'.format(expname, i))
            print(moviebase)
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=4, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=4, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            target_images = images[i_test] 
            precomputed_rays_o, precomputed_rays_d = get_rays_plenoptic(
                    intrinsic_params['F'],
                    intrinsic_params['M'], 
                    intrinsic_params['s'], 
                    images,   
                    intrinsic_params['pix'],
                    intrinsic_params['mla_centers'],
                    intrinsic_params['lens_diameter'],
                    poses,
                    indices=i_test  
                )

            with torch.no_grad():
                 _,_, psnrs = render_path_plenoptic(target_images, i_test, hwf, K, args.chunk, render_kwargs_test, gt_imgs=[i_test], savedir=testsavedir, precomputed_rays_o=precomputed_rays_o, precomputed_rays_d=precomputed_rays_d)
            print("Testset saved for iter' :", i)

            """
            # Calcul du PSNR 
            avg_test_psnr = np.mean(psnrs)
            test_psnrs.append(avg_test_psnr)
            test_iters.append(i)

            # Sauvegarde du log
            np.savetxt(psnr_log_path, np.column_stack([test_iters, test_psnrs]), delimiter=',', header='iteration,psnr' , comments='')

            
            plt.figure()
            plt.plot(test_iters, test_psnrs, label='PSNR - test set')
            plt.xlabel('Iterations')
            plt.ylabel('PSNR (dB)')
            plt.title('Evolution of PSNR during Training')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(basedir, expname, 'psnr_curve.png'))
            plt.close()

        
        """
        
            
            

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
    
            with torch.no_grad():
                pose_1 = poses[0]
                render_path([pose_1], [2160, 3840,9500], K, args.chunk, render_kwargs_test, gt_imgs=None, savedir=testsavedir, render_factor=0)
            print('Saved test set')
    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")  # Iter: 400 Loss: nan  PSNR: nan
            #print("psnr test : ",psnrtest)
        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
