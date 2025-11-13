import itertools
import logging
import pathlib
import random
import shutil
import time
import pandas as pd
import os
import numpy as np
import torch
import torchvision
import nibabel
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio
from sewar.full_ref import vifp
from image_similarity_measures.quality_metrics import fsim
import argparse
import transforms 
import h5py
import matplotlib.pyplot as plt
from models.subsampling_model import Subsampling_Model_VST as Subsampling_Model, FeatureVSTBInterleaved, CNN3D
from models.rec_models.unet_model import UnetModel
import pickle as pkl
from common.utils import get_vel_acc
#Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RES = [144,384]

import torchmetrics

def compute_ssim(video1: torch.Tensor, video2: torch.Tensor):
    """
    Compute the average SSIM between two videos using batch processing.
    
    Args:
        video1: Tensor of shape (T, W, H) or (T, 1, W, H) if already in grayscale.
        video2: Tensor of shape (T, W, H) or (T, 1, W, H).

    Returns:
        Average SSIM score across frames.
    """
    assert video1.shape == video2.shape, "Videos must have the same shape"
    
    # Add channel dimension if missing (convert (T, W, H) → (T, 1, W, H))
    if video1.ndim == 3:
        video1 = video1.unsqueeze(1)  # Shape: (T, 1, W, H)
        video2 = video2.unsqueeze(1)  # Shape: (T, 1, W, H)
    
    # Compute SSIM on all frames at once
    ssim_fn = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_scores = ssim_fn(video1, video2)  # Batched SSIM
    
    return ssim_scores.item()  # Convert tensor to float
class DecayModel(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.feature_model = CNN3D(9,3) if args.cnn3d else FeatureVSTBInterleaved(in_chans = 1, layer_num = args.num_layers,vst_depth = args.depth, feature_num = args.num_chans, out_chans = 3) #A,B,T1
        self.fc = torch.nn.Sequential(torch.nn.Linear(args.num_frames_per_example,1))
    
    def forward(self, x):
        features = self.feature_model(x)
        return  self.fc(features.permute(0,2,3,1) if len(features.shape)==4 else features.permute(0,1,3,4,2)).squeeze()
        

#This class performs data transformations on the data coming from the dataset
class DataTransform:
    def __init__(self, resolution=RES):
        self.resolution = resolution

    def __call__(self, kspace):
        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.resolution[0], self.resolution[1]))
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)

        #target = transforms.to_tensor(target)
        #target = transforms.center_crop(target.unsqueeze(0), (self.resolution[0], self.resolution[1])).squeeze()
        #target, mean, std = transforms.normalize_instance(target, eps=1e-11)
        #mean = std = 0
        return image
def get_metrics(gt, pred,compute_vif=False,compute_fsim=False):
    """ Compute Peak Signal to Noise Ratio metric (PSNR).
     By default we do not compute FSIM and VIF for every iteration because their long
     compute times. We advise training the model and then computing these values only for
     post-training evaluations.
     """
    gt = gt.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    stacked_gt, stacked_pred = gt.reshape(-1, gt.shape[2], gt.shape[3]), pred.reshape(-1, gt.shape[2], gt.shape[3])

    vif = np.mean([vifp(stacked_gt[i], stacked_pred[i]) for i in range(stacked_gt.shape[0])]) if compute_vif else None
    fsim_val = fsim(stacked_gt,stacked_pred) if compute_fsim else None

    return peak_signal_noise_ratio(gt, pred, data_range=gt.max()), vif, fsim_val








def plot_ims(predictions, labels, a,b,t1_star,out_dir=".", prefix="", new_best=False):
    
    assert predictions.shape == labels.shape, "Predictions and labels must have the same shape"
    assert predictions.shape[0] == 9, "Both tensors must have 9 images in the first dimension"
  
    eps = 1e-6
    if new_best:
        out_dir += "/best"
    os.makedirs(out_dir, exist_ok=True)
    
    for i in range(9):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot prediction
        axes[0].imshow(predictions[i].cpu().numpy(), cmap='gray')
        axes[0].set_title("Prediction")
        axes[0].axis("off")
        
        # Plot label
        axes[1].imshow(labels[i].cpu().numpy(), cmap='gray')
        axes[1].set_title("Label")
        axes[1].axis("off")
        
        # Save the plot
        output_path = os.path.join(out_dir, f"{prefix}_image_{i+1}.png")
        plt.savefig(output_path)
        plt.close(fig)
    plt.imsave(f"{out_dir}/A.png",a.cpu().detach().numpy())
    plt.imsave(f"{out_dir}/B.png",b.cpu().detach().numpy())
    plt.imsave(f"{out_dir}/T_star.png",t1_star.cpu().detach().numpy())
    t1_map = t1_star*(b/(a+eps) - 1)
    plt.imsave(f"{out_dir}/T1_map.png",t1_map.cpu().detach().numpy())
    with open(f'{out_dir}/t1_map.npy', 'wb') as f:
        np.save(f, t1_map.cpu().detach().numpy())
   


def signal_model(A, B, t1, tis, func="min"):
    #eps = 1e-6   
    #if func not in ["relu","abs","min"]:
    #    raise Exception
    #t1  = torch.nn.functional.relu(t1)+eps if func=="relu" else ((t1 - t1.amin(dim=(1, 2, 3), keepdim=True) + eps) if func=="min" else (t1.abs() + eps)) #t1 - t1.amin(dim=(1, 2, 3), keepdim=True) + eps#t1 - t1.min() + eps#t1.abs() + eps#torch.nn.functional.relu(t1)+eps
  
    return A-B* (torch.exp(-tis / t1)) 

def train(args, decay_model, rec_model, data, optimizer, roi_indices, writer, logger,iters=100, eps = 1e-6, out_dir=".", loss_window_length = 300, early_stop_tolerance = 0.0005):
    
    if args.opt_decay:
        decay_model.train()
    else: 
        decay_model.eval()
    if args.opt_rec:
        rec_model.train()
    else: 
        rec_model.eval()
    best_loss = float('inf')
    losses = []
    best_map = None
    best_vif = None
    best_fsim = None
    for iter in range(iters):
        
        new_best = False
        if optimizer is not None:
            optimizer.zero_grad()
        
        sample,tis = data
        sample = sample.to(args.device)[None]
        tis = tis.to(args.device)[None]
        
        
        target = transforms.complex_abs(sample)
        target = target[:,:,:,22:-22]
        target[:,:3] *= -1
        #target[flip_indices[None]] *= -1
        
        rec = rec_model(sample.permute(0,1,3,2,4),tis)
      
        
        output = decay_model(rec[:,:,22:-22,:].permute(0,1,3,2),tis,take_complex_abs=False).unsqueeze(2)
        A,B,t1 = output[:,0], output[:,1], output[:,2]
        t1  = t1 - t1.amin(dim=(1, 2, 3), keepdim=True) + eps
        decay = signal_model(A,B,t1,tis,func="min")
        
        if args.rec_only:
            loss = F.l1_loss(rec[:,:,22:-22,:].permute(0,1,3,2),target) if args.l1 else F.mse_loss(rec[:,:,22:-22,:].permute(0,1,3,2),target)
        else:
            loss = F.l1_loss(decay,target) if args.l1 else F.mse_loss(decay,target)
        
        decay_roi = decay[:, :,roi_indices[0]:roi_indices[1],roi_indices[2]:roi_indices[3]].cpu().detach()
        target_roi = target[:, :,roi_indices[0]:roi_indices[1],roi_indices[2]:roi_indices[3]].cpu().detach()
        roi_loss = F.l1_loss(decay_roi,target_roi) if args.l1 else F.mse_loss(decay_roi,target_roi)
     
        if loss.item()<best_loss: #we can't use roi for stop crietria because we assume we don't have it
            new_best = True
            best_loss_roi = roi_loss.item()
            best_loss = loss.item()
            best_map = (t1*(B/(A+eps) - 1)).cpu().detach()
            best_decay = decay.clone().cpu().detach()
            
            
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        if iter >= loss_window_length:
            losses = losses[-loss_window_length:]
            min_loss = min(losses)
            if ((losses[0] - min_loss)/min_loss) < early_stop_tolerance:
                
                logger.info("Early stop - no loss improvement.")
                break

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        psnr_train, vif_train, fsim_train = get_metrics(target, decay,args.vif,args.fsim)
        psnr_roi, vif_roi, fsim_roi = get_metrics(target_roi, decay_roi,args.vif,args.fsim)

        
        
        if iter % args.report_interval == 0:
            with torch.no_grad():
                plot_ims(decay[0],target[0],A.squeeze(),B.squeeze(),t1.squeeze(),out_dir = out_dir,new_best=new_best)
                plot_ims(decay_roi[0],target_roi[0],A[:, :,roi_indices[0]:roi_indices[1],roi_indices[2]:roi_indices[3]].squeeze(),\
                    B[:, :,roi_indices[0]:roi_indices[1],roi_indices[2]:roi_indices[3]].squeeze(),\
                    t1[:, :,roi_indices[0]:roi_indices[1],roi_indices[2]:roi_indices[3]].squeeze(),out_dir = f"{out_dir}_roi",new_best=new_best)
            logger.info(
                f'Epoch = [{iter:3d}/{iters:3d}] '
                f'Loss = {loss.item():.4g} ROI Loss = {roi_loss.item():.4g} '
                f' PSNR: {psnr_train}, SSIM: {compute_ssim(decay.cpu().detach(),target.cpu().detach())}'
                f' PSNR ROI: {psnr_roi}, SSIM ROI: {compute_ssim(decay_roi.cpu().detach(),target_roi.cpu().detach())}'
            )
    try:
        best_psnr, best_vif, best_fsim = get_metrics(target, best_decay,True,True)
    except:
        best_psnr, best_vif, best_fsim = get_metrics(target, best_decay,False,True)
    return best_loss, best_map.squeeze(), best_psnr, best_vif, best_fsim

def plot_scatter(x):
    if len(x.shape) == 4:
        return plot_scatters(x)
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    for i in range(x.shape[0]):
        ax.plot(x[i, :, 0], x[i, :, 1], '.')
    return fig


def plot_scatters(x):
    fig = plt.figure(figsize=[10, 10])
    for frame in range(x.shape[0]):
        ax = fig.add_subplot(2, x.shape[0]//2, frame + 1)
        for i in range(x.shape[1]):
            ax.plot(x[frame, i, :, 0], x[frame, i, :, 1], '.')
            ax.axis([-165, 165, -165, 165])
    return fig


def plot_trajectory(x):
    if len(x.shape) == 4:
        return plot_trajectories(x)
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    for i in range(x.shape[0]):
        ax.plot(x[i, :, 0], x[i, :, 1])
    return fig


def plot_trajectories(x):
    fig = plt.figure(figsize=[10, 10])
    for frame in range(x.shape[0]):
        ax = fig.add_subplot(2, x.shape[0]//2, frame + 1)
        for i in range(x.shape[1]):
            ax.plot(x[frame, i, :, 0], x[frame, i, :, 1])
            ax.axis([-165, 165, -165, 165])
    return fig


def plot_acc(a, a_max=None):
    fig, ax = plt.subplots(2, sharex=True)
    for i in range(a.shape[0]):
        ax[0].plot(a[i, :, 0])
        ax[1].plot(a[i, :, 1])
    if a_max is not None:
        limit = np.ones(a.shape[1]) * a_max
        ax[1].plot(limit, color='red')
        ax[1].plot(-limit, color='red')
        ax[0].plot(limit, color='red')
        ax[0].plot(-limit, color='red')
    return fig



def save_model(args, exp_dir, epoch, model, optimizer, is_new_best = False, best_loss = 0):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_dir': exp_dir,
            'best loss': best_loss
        },
        f=(exp_dir + '/model.pt')
    )
    if is_new_best:
        shutil.copyfile(exp_dir + '/model.pt', exp_dir + '/best_model.pt')



    
    return train_loader, dev_loader
def build_model(args,rec_model=False):
    model = Subsampling_Model(
        in_chans=args.num_frames_per_example,
        out_chans=args.num_frames_per_example if rec_model else 3,
        chans=args.num_chans,
        num_layers=args.num_layers,
        depth = args.depth,
        drop_prob=args.drop_prob,
        decimation_rate=args.decimation_rate,
        trajectory_learning=args.trajectory_learning if rec_model else False,
        initialization=args.initialization,
        SNR=args.SNR,
        projection_iters=args.proj_iters,
        project=args.project,
        n_shots=args.n_shots,
        interp_gap=args.interp_gap,
        multiple_trajectories=not args.single_traj,
        embedding_dim = 0 if rec_model else args.embedding_dim,
        no_subsample = not rec_model
    ).to(args.device)
    return model


def load_model(checkpoint_file,args,rec_model=False):
   
    checkpoint = torch.load(checkpoint_file)
    #args = checkpoint['args']
    model = build_model(args,rec_model)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    
    
    return model


def build_optim(args, model):
    if args.embedding_dim:
        optimizer = torch.optim.Adam([{'params': model.subsampling.parameters(), 'lr': args.sub_lr},{'params': model.embedding.parameters(), 'lr': args.embedding_lr},
                              {'params': model.reconstruction_model.parameters()}], args.lr)

    else:
    
        optimizer = torch.optim.Adam([{'params': model.subsampling.parameters(), 'lr': args.sub_lr},
                                      {'params': model.reconstruction_model.parameters()}], args.lr)
    return optimizer
def load_sample(path, device="cuda"):

    f = h5py.File(f"{path}/T1map.mat","r")['kspace_single_full']   
    data = f['real'] + f['imag']*1j
    data = DataTransform()(data)
    sample =  data.to(device)
   
    return sample



def save_comparison_plot(img1: np.ndarray, img2: np.ndarray, save_path: str):
    """
    Saves a single file containing three subplots:
    - First image
    - Second image
    - Absolute difference between them
    The titles include the mean and standard deviation of the L1 error.
    
    Args:
        img1 (np.ndarray): First image (144x344)
        img2 (np.ndarray): Second image (144x344)
        save_path (str): Path to save the output plot
    """
    
    
    abs_diff = np.abs(img1 - img2)
    mean_l1 = np.mean(abs_diff)
    std_l1 = np.std(abs_diff)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title("GT")
    axes[0].axis('off')
    
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title("Rev")
    axes[1].axis('off')
    
    axes[2].imshow(abs_diff, cmap='hot')
    axes[2].set_title(f"|Diff| (mean={mean_l1:.2f}, std={std_l1:.2f})")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    


 
def main():
    args = create_arg_parser().parse_args()
    args.v_max = args.gamma * args.G_max * args.FOV * args.dt
    args.a_max = args.gamma * args.S_max * args.FOV * args.dt ** 2 * 1e3
    
    if args.rec_only:
        args.opt_decay = False #decay model does not receive gradients in this case
    
    args.exp_dir = f'summary_with_rois_and_finetuned_gt_all_image_comp/{args.test_name}'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pathlib.Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    
    with open(args.exp_dir + '/args.txt', "w") as text_file:
        print(vars(args), file=text_file)
    
    #flip_idxs_file = pkl.load(open("/home/tamirshor/T1_mapping/flip_idxs.pkl",'rb'))#pkl.load(open("/home/tamirshor/T1_mapping/pixelwise_flips.pkl",'rb'))
    test_files = sorted([str(x) for x in os.listdir(args.data_path)])[args.start:args.end]
    for file in test_files:
        samples = load_sample(f"{args.data_path}/{file}", args.device)
        tis_df = pd.read_csv(f'{args.data_path}/{file}/T1map.csv')
        mask_path = f"/home/tamir.shor/T1_mapping/SegmentROI/{file}/T1map_label.nii.gz"
        mask = nibabel.load(mask_path).get_fdata()[86:-86] #we crop the mask to image dims
       
        coords = np.argwhere(mask > 0)
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0) + 1
        (min_x, min_y, min_z), (max_x, max_y, max_z) = min_coords, max_coords
        roi_indices = (min_y,max_y, min_x,max_x)
        for slice in range(samples.shape[1]):
            
            if os.path.isdir(f"{args.exp_dir}/{file}/slice {slice}"):
                continue
            sample = samples[:,slice]
            writer = SummaryWriter(log_dir=f"{args.exp_dir}/{file}")
            slice_log_dir = f"{args.exp_dir}/{file}/slice {slice}"
            slice_log_path = f"{slice_log_dir}/log.txt"
            
         
            #flip_indices = flip_idxs_file[f"{args.data_path}/{file}"][slice]
            pathlib.Path(slice_log_dir).mkdir(parents=True, exist_ok=True)

            
            logger = logging.getLogger(f"slice {slice}")
            logger.setLevel(logging.INFO)

           
            if logger.hasHandlers():
                logger.handlers.clear()

           
            handler = logging.FileHandler(slice_log_path, mode='w')
            
            logger.addHandler(handler)
           
       
            tis = torch.Tensor(tis_df.iloc[:,slice+1].values)[:,None,None]/1000 #work in secs
            
            if args.resume:
                decay_model = load_model(args.checkpoint_decay,args,False)
            else:
                decay_model = build_model(args,False)#load_model(args.checkpoint_decay,args)
            rec_model = load_model(args.checkpoint_rec,args,True)
            
            decay_model = decay_model.to(args.device)
            rec_model = rec_model.to(args.device)
            
            opt_list = []
            if args.opt_rec:
                opt_list += list(rec_model.reconstruction_model.parameters())
            if args.opt_decay:
                opt_list += list(decay_model.reconstruction_model.parameters())
                
            
            optimizer = torch.optim.Adam(opt_list,lr=args.lr) if len(opt_list) else None
            if args.opt_decay:
                optimizer.add_param_group({'params':decay_model.embedding.parameters(),'lr':args.embedding_lr})
                
      

            best_loss,t1_map, best_psnr, best_vif, best_fsim = train(
                args,
                decay_model,
                rec_model,
                (sample,tis),
                optimizer,
                roi_indices,
                writer, logger, iters = args.num_epochs,
                out_dir = f"{args.exp_dir}/{file}/slice {slice}/")
            
            try:
                gt = np.load(f"/home/tamir.shor/T1_mapping/no_subsample_finetune_3000_steps/{file}/slice {slice}/t1_map.npy")
                gt = torch.Tensor(gt)
                l1_diff = torch.nn.functional.l1_loss(gt,t1_map).item()
                l2_diff = torch.nn.functional.mse_loss(gt,t1_map).item() 
                gt_roi = gt.squeeze()[roi_indices[0]:roi_indices[1],roi_indices[2]:roi_indices[3]].cpu().detach()
                map_roi = t1_map.squeeze()[roi_indices[0]:roi_indices[1],roi_indices[2]:roi_indices[3]].cpu().detach()
                l1_diff_roi = torch.nn.functional.l1_loss(gt_roi,map_roi).item()
                l2_diff_roi = torch.nn.functional.mse_loss(gt_roi,map_roi).item() 
                
                save_comparison_plot(gt.numpy(),t1_map.numpy(),f"{args.exp_dir}/{file}/slice {slice}/gt_comp.png")
                save_comparison_plot(gt_roi.numpy(),map_roi.numpy(),f"{args.exp_dir}/{file}/slice {slice}/gt_comp_roi.png")
                
            except:
                l2_diff = l1_diff = "No GT Found"
            
            os.makedirs(f"{args.exp_dir}/{file}/slice {slice}",exist_ok=True)
            with open(f"{args.exp_dir}/{file}/slice {slice}/best_loss.txt",'w') as f:
                f.write(f"Best loss: {best_loss}")
                if best_vif is not None:
                    f.write(f"Best PSNR, VIF, FSIM: {(best_psnr, best_vif, best_fsim)} ")
                else:
                    f.write(f"Best PSNR, FSIM: {best_psnr, best_fsim} ")
                f.write(f"l1 map diff: {l1_diff}, l2 map diff: {l2_diff}")
                f.write(f"l1 map ROI diff: {l1_diff_roi}, l2 map ROI diff: {l2_diff_roi}")
                
            
            writer.close()
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)

       
         
    
    
   



    
   





def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', type=pathlib.Path,
                      default='/home/tamir.shor/T1_mapping/FullSample_test', help='Path to the dataset')
    
    parser.add_argument('--checkpoint-rec', type=str, default= "/home/tamir.shor/T1_mapping/finetune_team_pilot/saved_weights_32/9_frames_32_shots_bs_12_u118/plain_finetune/summary/test/best_model.pt",help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--checkpoint-decay', type=str, default= "/home/tamirshor/T1_mapping/finetune_team_pilot_exps/saved_weights_32/9_frames_32_shots_bs_12_u118/evaluate/opt_decay_frozen_rec/opt_decay_frozen_rec.pt",help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--model-path', type=str, default='/home/tamirshor/T1_mapping/finetune_joint_400_600/summary/test/best_u_463.pt', help= 'Path to trained reconstruction model')
    parser.add_argument('--test-name', type=str, default='test/', help='name for the output dir')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='output/',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    
    parser.add_argument('--fsim',action='store_true',help="calculate fsim values (advised to only use this over a trained model, not in training - computing fsim takes ~30 secs)")
    parser.add_argument('--vif', action='store_true',
                        help="calculate vif values (advised to only use this over a trained model, not in training - computing vif takes ~30 secs)")

    # model parameters
    parser.add_argument('--num-layers', type=int, default=1, help='Number of VST Block layers')
    parser.add_argument('--depth', type=int, default=1, help='Depth of VST Block layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=96, help='Number of channels for feature extraction')
    parser.add_argument('--data-parallel', action='store_true', default=False,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--decimation-rate', default=10, type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for each volume.')

    # optimization parameters
    parser.add_argument('--batch-size', default=1, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=10000, help='Number of training epochs per frame')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for reconstruction model')
    parser.add_argument('--lr-step-size', type=int, default=300,
                        help='Period of learning rate decay for reconstruction model')
    parser.add_argument('--lr-gamma', type=float, default=1,
                        help='Multiplicative factor of reconstruction model learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    # trajectory learning parameters
    parser.add_argument('--sub-lr', type=float, default=0.0, help='learning rate of the sub-samping layer')
    parser.add_argument('--sub-lr-time', type=float, default=300,
                        help='learning rate decay timestep of the sub-sampling layer')
    parser.add_argument('--sub-lr-stepsize', type=float, default=1,
                        help='learning rate decay step size of the sub-sampling layer')

    parser.add_argument('--trajectory-learning', default=False, action = "store_true",
                        help='trajectory_learning, if set to False, fixed trajectory, only reconstruction learning.')

    #MRI Machine Parameters
    parser.add_argument('--acc-weight', type=float, default=1e-2, help='weight of the acceleration loss')
    parser.add_argument('--vel-weight', type=float, default=1e-1, help='weight of the velocity loss')
    parser.add_argument('--rec-weight', type=float, default=1, help='weight of the reconstruction loss')
    parser.add_argument('--gamma', type=float, default=42576, help='gyro magnetic ratio - kHz/T')
    parser.add_argument('--G-max', type=float, default=40, help='maximum gradient (peak current) - mT/m')
    parser.add_argument('--S-max', type=float, default=180, help='maximum slew-rate - T/m/s')
    parser.add_argument('--FOV', type=float, default=0.2, help='Field Of View - in m')
    parser.add_argument('--dt', type=float, default=1e-5, help='sampling time - sec')
    parser.add_argument('--a-max', type=float, default=0.17, help='maximum acceleration')
    parser.add_argument('--v-max', type=float, default=3.4, help='maximum velocity')
    parser.add_argument('--initialization', type=str, default='radial',
                        help='Trajectory initialization when using PILOT (spiral, EPI, rosette, uniform, gaussian).')
    parser.add_argument('--SNR', action='store_true', default=False,
                        help='add SNR decay')
    #modelization parameters
    parser.add_argument('--n-shots', type=int, default=16,
                        help='Number of shots')
    parser.add_argument('--interp_gap', type=int, default=10,
                        help='number of interpolated points between 2 parameter points in the trajectory')
    parser.add_argument('--num_frames_per_example', type=int, default=9, help='num frames per example')
   

    parser.add_argument('--project', action='store_true', default=True, help='Use projection to impose kinematic constraints.'
                                                                             'If false, use interpolation and penalty (original PILOT paper).')
    parser.add_argument('--proj_iters', default=1, help='Number of iterations for each projection run.')
    parser.add_argument('--single_traj', action='store_true', default=False, help='allow different trajectory per frame')
    parser.add_argument('--l1', action='store_true', default=False, help='Use l1 criterion.')
    parser.add_argument('--embedding-dim', type=int, default=0, help='TI time embedding dimension.')
    parser.add_argument('--no-subsample', action='store_true', default=True, help='Use full data.')
    parser.add_argument('--start', type=int, default=0,
                        help='sample start index')
    parser.add_argument('--end', type=int, default=60,
                        help='sample end index')
    parser.add_argument('--embedding-lr', type=float, default=0.01, help='Embedding Layer lr.')
    parser.add_argument('--opt-decay',action='store_true',default=False,help="finetune decay model")
    parser.add_argument('--opt-rec',action='store_true',default=False,help="finetune reconstruction model")
    parser.add_argument('--rec-only',action='store_true',default=False,help="Finetune with a reconstruction objective (for rec baseline)")
    
    return parser




if __name__ == '__main__':
    main()
