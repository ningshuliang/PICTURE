import cv2
import torch
import os
from basicsr.utils import img2tensor, tensor2img, scandir, get_time_str, get_root_logger, get_env_info
from ldm.data.dataset_coco import dataset_coco_mask_color
import argparse
from pytorch_lightning import seed_everything

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.modules.encoders.adapter import Adapter,MLP_Adapter
from PIL import Image
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import sys
from CLIP.clip import clip
import random
import os.path as osp
from basicsr.utils.options import copy_opt_file, dict2str
import logging
from dist_util import init_dist, master_only, get_bare_model, get_dist_info
from datas.stage2dataset import TrainDataset,TestDataset 

def modify_weights(w, scale = 1e-6, n=128):
    """Modify weights to accomodate concatenation to unet"""
    extra_w = scale*torch.randn_like(w)
    new_w = w.clone()

    for i in range(n):
        new_w = torch.cat((new_w, extra_w.clone()), dim=1)

    return new_w


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)


    keys_to_change = [
                "model.diffusion_model.input_blocks.0.0.weight"
            ]
    scale = 1e-8
    for k in keys_to_change:
        print("modifying input weights for compatibitlity")
        sd[k] = modify_weights(sd[k], scale=scale, n=2)


    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

@master_only
def mkdir_and_rename(path):

    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(experiments_root, 'models'))
    os.makedirs(osp.join(experiments_root, 'training_states'))
    os.makedirs(osp.join(experiments_root, 'visualization'))

def load_resume_state(opt):
    resume_state_path = None
    if opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt.resume_state_path = resume_state_path
    # else:
    #     if opt['path'].get('resume_state'):
    #         resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        # check_resume(opt, resume_state['iter'])
    return resume_state

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bsize",
    type=int,
    default=96,
    help="the prompt to render"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10000,
    help="the prompt to render"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="the prompt to render"
)
parser.add_argument(
    "--use_shuffle",
    type=bool,
    default=True,
    help="the prompt to render"
)
parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
)
parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
)
parser.add_argument(
        "--auto_resume",
        action='store_true',
        help="use plms sampling",
)
parser.add_argument(
        "--ckpt",
        type=str,
        default="models/v1-5-pruned-emaonly.ckpt",
        help="path to checkpoint of model",
)
parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/train_mask.yaml",
        help="path to config which constructs model",
)
parser.add_argument(
        "--print_fq",
        type=int,
        default=100,
        help="path to config which constructs model",
)
parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=256,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
)
parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
        "--scale",
        type=float,
        default=20.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
        "--gpus",
        default=[0,1],
        help="gpu idx",
)
parser.add_argument(
        '--local_rank', 
        default=0, 
        type=int,
        help='node rank for distributed training'
)

parser.add_argument(
        '--seed', 
        default=23, 
        type=int,
        help='radom seed '
)

parser.add_argument(
        '--launcher', 
        default='pytorch', 
        type=str,
        help='node rank for distributed training'
)
opt = parser.parse_args()

if __name__ == '__main__':
    config = OmegaConf.load(f"{opt.config}")
    opt.name = config['name']
    
    # distributed setting
    init_dist(opt.launcher)
    torch.backends.cudnn.benchmark = True
    device='cuda'
    torch.cuda.set_device(opt.local_rank)

    # dataset
    train_path1 = '/data4/Shuliang/CVPR_2024/SHHQ/Stage2/train/Img_Clear/'
    train_path2 = '/data4/Shuliang/CVPR_2024/SHHQ/Stage2/train/Img_Wo_Cloth'
    train_path3 = '/data4/Shuliang/CVPR_2024/SHHQ/Stage2/train/Parsing_Clear'
    train_path4 = '/data4/Shuliang/CVPR_2024/SHHQ/Stage2/train/Cloth_Train'

    test_path1 = '/data4/Shuliang/CVPR_2024/SHHQ/Stage2/test/Img_Clear/'
    test_path2 = '/data4/Shuliang/CVPR_2024/SHHQ/Stage2/test/Img_Wo_Cloth'
    test_path3 = '/data4/Shuliang/CVPR_2024/SHHQ/Stage2/test/Parsing_Clear'
    test_path4 = '/data4/Shuliang/CVPR_2024/SHHQ/Stage2/test/Cloth_Train'

    train_dataset = TrainDataset(train_path1,train_path2,train_path3,train_path4)
    val_dataset = TestDataset(test_path1,test_path2,test_path3,test_path4)






    


    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.bsize,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False)

   
    # stable diffusion
    model = load_model_from_config(config, f"{opt.ckpt}").to(device)
    
    # change vae checkpoint
    model.first_stage_model.init_from_ckpt('/data4/Shuliang/New_data/T2I-Adapter_CLIP_Freeze_MLP_Change_Adapter_MLFeature_Guidance_select_features_position_embedding/models/vae-ft-mse-840000-ema-pruned.ckpt')
    

    # img_wo_cloth and parsing encoder

    mlp_layer0 = MLP_Adapter(dim=1024).to(device)


    clip_model, preprocess = clip.load("ViT-L/14", device=device)


    # to gpus

    
    mlp_layer0 = torch.nn.parallel.DistributedDataParallel(
        mlp_layer0,
        device_ids=[opt.local_rank], 
        output_device=opt.local_rank)
    

    
    
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[opt.local_rank], 
        output_device=opt.local_rank)
        # device_ids=[torch.cuda.current_device()])

    
    clip_model = torch.nn.parallel.DistributedDataParallel(
        clip_model,
        device_ids=[opt.local_rank], 
        output_device=opt.local_rank)
    

    # optimizer
    params = list(mlp_layer0.parameters())+list(model.module.model.diffusion_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=config['training']['lr'])

    experiments_root = osp.join('experiments', opt.name)


    # resume state
    resume_state = load_resume_state(opt)
    if resume_state is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
    else:
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
        resume_optimizers = resume_state['optimizers']
        optimizer.load_state_dict(resume_optimizers)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)

    
    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    for epoch in range(start_epoch, opt.epochs):

        
        # val
        rank, _ = get_dist_info()
        if rank==0:
            for batch_id,batch in enumerate(val_dataloader):
    
                seed_everything(opt.seed)
                with torch.no_grad():
                    if opt.dpm_solver:
                        sampler = DPMSolverSampler(model.module)
                    elif opt.plms:
                        sampler = PLMSSampler(model.module)
                    else:
                        sampler = DDIMSampler(model.module)

                    x = batch['image']
                    
                    image_wo_cloth,parsing = batch['concat']

                    image_wo_cloth_latent = model.module.get_first_stage_encoding(model.module.encode_first_stage(image_wo_cloth.cuda(non_blocking=True)))
 

                    parsing_latent = model.module.get_first_stage_encoding(model.module.encode_first_stage(parsing.cuda(non_blocking=True)))
                    concat_feature = torch.concat([image_wo_cloth_latent,parsing_latent],1)
                    cloth = batch['txt']
                    parsing_original = batch['original_img']

                    
                    

                    feature_list,c = clip_model.module.encode_image(F.interpolate(cloth.cuda(non_blocking=True),size=224))


                    condition_list = []
                    condition_temp= []

                    features = torch.cat([feature_list[1].permute(1,0,2),
                                          feature_list[5].permute(1,0,2),
                                          feature_list[8].permute(1,0,2),
                                          feature_list[9].permute(1,0,2),
                                          feature_list[10].permute(1,0,2),
                                          feature_list[12].permute(1,0,2),
                                          feature_list[16].permute(1,0,2),
                                          feature_list[21].permute(1,0,2),

                    ],1)
                   
                    condition_list.append(mlp_layer0(features))
                    
                    zero_cloth= torch.zeros_like(cloth)
                    zero_feature_list,zero_c = clip_model.module.encode_image(F.interpolate(zero_cloth.cuda(non_blocking=True),size=224))


                    u_condition_list = []
                    u_condition_temp= []
                    zero_features = torch.cat([zero_feature_list[1].permute(1,0,2),
                                          zero_feature_list[5].permute(1,0,2),
                                          zero_feature_list[8].permute(1,0,2),
                                          zero_feature_list[9].permute(1,0,2),
                                          zero_feature_list[10].permute(1,0,2),
                                          zero_feature_list[12].permute(1,0,2),
                                          zero_feature_list[16].permute(1,0,2),
                                          zero_feature_list[21].permute(1,0,2),

                    ],1)
                    u_condition_list.append(mlp_layer0(zero_features))
              

                    uc = [concat_feature,u_condition_list]

                   



                    

                    c = [concat_feature,condition_list]
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=None,
                                                        features_adapter = None)
                    x_samples_ddim = model.module.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    
                    for id_sample, x_sample in enumerate(x_samples_ddim):
                        x_sample = 255.*x_sample
                        img = x_sample.astype(np.uint8)

                        image_wo_cloth1 = ((x+1)/2).permute(0,2,3,1)
                        image_wo_cloth1 = image_wo_cloth1[0].detach().cpu().numpy()
                        image_wo_cloth1 = (image_wo_cloth1*255).astype(np.uint8)


                        parsing1 = parsing_original.permute(1,2,0).detach().cpu().numpy()
                        mask = ((parsing1==14)+(parsing1==11)+(parsing1==0))*1
                        mask = np.concatenate([mask]*3,2)
                        img = mask*image_wo_cloth1+(1-mask)*img


                        cloth = ((cloth+1)/2).permute(0,2,3,1)
                        cloth = cloth[0].detach().cpu().numpy()
                        template = np.ones([512,256,3])*255
                        template[256-112:256+112,128-112:128+112,:]=cloth*255
                        img = np.concatenate([img,template],1)
                        
                        cv2.imwrite(os.path.join(experiments_root, 'visualization', 'sample_e%04d_s%04d.png'%(epoch, batch_id)), img[:,:,::-1])
        # train
        for _, batch in enumerate(train_dataloader):
            current_iter += 1

            x = batch['image']
            image_wo_cloth,parsing = batch['concat']
            cloth = batch['txt']
            

            with torch.no_grad():

                number = random.random()
                if number<0.2:
                    cloth= torch.zeros_like(cloth)
                feature_list,c = clip_model.module.encode_image(F.interpolate(cloth.cuda(non_blocking=True),size=224))
                


                condition_list = []
                condition_temp= []

                features = torch.cat([feature_list[1].permute(1,0,2),
                                          feature_list[5].permute(1,0,2),
                                          feature_list[8].permute(1,0,2),
                                          feature_list[9].permute(1,0,2),
                                          feature_list[10].permute(1,0,2),
                                          feature_list[12].permute(1,0,2),
                                          feature_list[16].permute(1,0,2),
                                          feature_list[21].permute(1,0,2),

                    ],1)
                   
                condition_list.append(mlp_layer0(features))
                z = model.module.get_first_stage_encoding(model.module.encode_first_stage(x.cuda(non_blocking=True)))
                image_wo_cloth_latent = model.module.get_first_stage_encoding(model.module.encode_first_stage(image_wo_cloth.cuda(non_blocking=True)))
                parsing_latent = model.module.get_first_stage_encoding(model.module.encode_first_stage(parsing.cuda(non_blocking=True)))
              

            concat_feature = torch.concat([image_wo_cloth_latent,parsing_latent],1)
            
            optimizer.zero_grad()
            model.zero_grad()
            c = [concat_feature,condition_list]
            l_pixel, loss_dict = model(z, c=c, features_adapter = None)
            
            l_pixel.backward()
            optimizer.step()

            if (current_iter+1)%opt.print_fq == 0:
                logger.info(loss_dict)
            
            # save checkpoint
            rank, _ = get_dist_info()
            if (rank==0) and ((current_iter+1)%config['training']['save_freq'] == 0):
                

                


                save_filename = f'mlp_layer0_{current_iter+1}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                save_dict = {}
                mlp_layer0_bare = get_bare_model(mlp_layer0)
                state_dict = mlp_layer0_bare.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    save_dict[key] = param.cpu()
                torch.save(save_dict, save_path)

                


               



                save_filename = f'model_{current_iter+1}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                save_dict = {}
                model_bare = get_bare_model(model)
                state_dict = model_bare.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    save_dict[key] = param.cpu()
                torch.save(save_dict, save_path)

                


            # save state
                state = {'epoch': epoch, 'iter': current_iter+1, 'optimizers': optimizer.state_dict()}
                save_filename = f'{current_iter+1}.state'
                save_path = os.path.join(experiments_root, 'training_states', save_filename)
                torch.save(state, save_path)
