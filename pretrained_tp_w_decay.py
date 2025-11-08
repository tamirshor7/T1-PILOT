import itertools
import logging
import pathlib
import random
import shutil
import time
import os
import numpy as np
import torch
from torch import nn
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio
from sewar.full_ref import vifp
from image_similarity_measures.quality_metrics import fsim
from torch.utils.data import DataLoader
import argparse
import transforms
from data import SliceData
import h5py
import math
import matplotlib.pyplot as plt
from models.subsampling_model import Subsampling_Model_VST as Subsampling_Model
from common.utils import get_vel_acc

#Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransform:
    def __init__(self, resolution=[144,384]):
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




def create_datasets(args):
    
    rel_files = [f"{str(args.data_path)}/{str(x)}" for x in os.listdir(args.data_path)]
    train_ratio = 0.8
    num_train = int(np.ceil(len(rel_files) * train_ratio))
    train_files = rel_files[:num_train]
    val_files = rel_files[num_train:]

    train_data = SliceData(
        files=train_files,
        transform=DataTransform(),
        flip_load_path=None,#"/home/tamirshor/T1_mapping/pixelwise_flips.pkl",
        num_frames_per_example=args.num_frames_per_example
    )
    dev_data = SliceData(
        files=val_files,
        transform=DataTransform(),
        flip_load_path=None,#"/home/tamirshor/T1_mapping/pixelwise_flips.pkl",
        num_frames_per_example=args.num_frames_per_example
    )

    return dev_data, train_data

def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
   

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, dev_loader


def signal_model(A, B, t1, tis):
    eps = 1e-6
    t1 = t1 - t1.amin(dim=(1, 2, 3), keepdim=True) + eps#t1.abs() + eps#torch.nn.functional.relu(t1)+eps

    return A-B* (torch.exp(-tis / t1)) 
import os
import torch
import matplotlib.pyplot as plt

def save_comparison_images(tensor1, tensor2, save_dir):
    """
    Saves side-by-side comparison images from two tensors.

    Args:
        tensor1 (torch.Tensor): Shape (9, H, W), first set of images.
        tensor2 (torch.Tensor): Shape (9, H, W), second set of images.
        save_dir (str): Directory where images will be saved.
    """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Iterate over the 9 images
    for i in range(9):
        img1 = tensor1[i].cpu().detach()
        img2 = tensor2[i].cpu().detach()
        abs_diff = torch.abs(img1 - img2)

        # Compute mean and std of absolute difference
        mean_diff = abs_diff.mean().item()
        std_diff = abs_diff.std().item()
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Plot first tensor image
        axes[0].imshow(img1, cmap='viridis')
        axes[0].set_title("Tensor 1")
        axes[0].axis('off')

        # Plot second tensor image
        axes[1].imshow(img2, cmap='viridis')
        axes[1].set_title("Tensor 2")
        axes[1].axis('off')

        # Plot absolute difference
        im = axes[2].imshow(abs_diff, cmap='magma')
        axes[2].set_title("Abs Diff")
        axes[2].axis('off')

        # Add a colorbar for the difference image
        fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

        # Set the figure title with statistics
        fig.suptitle(f"Triplet {i+1}/9 | Mean: {mean_diff:.4f}, Std: {std_diff:.4f}")

        # Save the figure
        save_path = os.path.join(save_dir, f"comparison_{i+1}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    print(f"Saved comparison images to {save_dir}")


def train_epoch(args, epoch, model,decay_model, data_loader, optimizer, writer, loader_len):
    model.train()
    if args.opt_decay:
        decay_model.train()
    else:
        decay_model.eval()
   
    avg_loss = 0.
    criteria = F.l1_loss if args.l1 else F.mse_loss 
    #The interpolation gap is only meaningful if we do not use projection
    #if epoch < 20:
    #    model.subsampling.interp_gap = 32
    #elif epoch == 20:
    #    model.subsampling.interp_gap = 16
    #elif epoch == 30:
    #    model.subsampling.interp_gap = 8
    #elif epoch == 40:
    #    model.subsampling.interp_gap = 4
    #elif epoch == 46:
    #    model.subsampling.interp_gap = 2
    #elif epoch == 50:
    #    model.subsampling.interp_gap = 2
    model.subsampling.interp_gap = 2
    start_epoch = time.perf_counter()
    print(f'Imposing Machine Constraints: a_max={args.a_max}, v_max={args.v_max}')
    for iter, (sample, tis, target) in enumerate(data_loader):
        
        optimizer.zero_grad()
        
        
        sample = sample.to(args.device)
        tis = tis.to(args.device)
        
        target = target.to(args.device)
        
        #target[:,:3] *= -1 #undo flip
        target = target[:,:,:,22:-22]
       
        
        rec = model(sample.permute(0,1,3,2,4),tis)#.permute(0,1,3,2)
       
        output = decay_model(rec[:,:,22:-22,:].permute(0,1,3,2),tis,take_complex_abs=False).unsqueeze(2)
        #output_r = output[:,3:].squeeze()
       
        A,B,t1 = output[:,0], output[:,1], output[:,2]
        decay = signal_model(A,B,t1,tis)
        
        rec_loss = criteria(decay,target)
    
        # Loss on trajectory vel and acc
        x = model.get_trajectory()
        v, a = get_vel_acc(x)
        acc_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max).abs() + 1e-8, 2)))
        vel_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max).abs() + 1e-8, 2)))

        # target loss

       
        #weigh kinematic los in overall loss
        loss = rec_loss + args.vel_weight * vel_loss + args.acc_weight * acc_loss
      
        loss.backward()

        
        
       

        optimizer.step()



        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        psnr_train, vif_train, fsim_train = get_metrics(target, decay,args.vif,args.fsim)


        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{loader_len:4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'rec_loss: {rec_loss:.4g}, vel_loss: {vel_loss:.4g}, acc_loss: {acc_loss:.4g}'
                f' PSNR: {psnr_train}'
            )
        
    pi=0
    if epoch and not epoch%10:
        save_comparison_images(decay[pi], target[pi], f"{args.exp_dir}/comps_train")
     
    return avg_loss, time.perf_counter() - start_epoch, rec_loss, vel_loss, acc_loss, psnr_train


def evaluate(args, epoch, model,decay_model, data_loader, writer, dl_len, train_loss=None, train_rec_loss=None, train_vel_loss=None,
             train_acc_loss=None, psnr_train=None):
    model.eval()
    decay_model.eval()
    
    losses = []
    psnrs = []
    vifs = []
    fsims = []
    psnrs_r = []
    criteria = F.l1_loss if args.l1 else F.mse_loss 
    start = time.perf_counter()
    with torch.no_grad():
        if epoch != 0:
            for iter, (sample,tis,target) in enumerate(data_loader):

                
                sample = sample.to(args.device)
                tis = tis.to(args.device)
               
                target = target.to(args.device)#transforms.complex_abs(sample)
                
                target = target[:,:,:,22:-22]
                
                
                
                rec = model(sample.permute(0,1,3,2,4),tis)#.permute(0,1,3,2)
       
                output = decay_model(rec[:,:,22:-22,:].permute(0,1,3,2),tis,take_complex_abs=False).unsqueeze(2)
                #output_r = output[:,3:].squeeze()
               
                A,B,t1 = output[:,0], output[:,1], output[:,2]
                decay = signal_model(A,B,t1,tis)
                
                rec_loss = criteria(decay,target)
                
                # Loss on trajectory vel and acc
                x = model.get_trajectory()
                v, a = get_vel_acc(x)
                acc_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max).abs() + 1e-8, 2)))
                vel_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max).abs() + 1e-8, 2)))

              
                loss = rec_loss + args.vel_weight * vel_loss + args.acc_weight * acc_loss
               
                #weigh kinematic los in overall loss
               
                psnr_dev, vif_dev, fsim_dev = get_metrics(target, decay,args.vif,args.fsim)
                

                psnrs.append(psnr_dev)
                
                if vif_dev is not None:
                    vifs.append(vif_dev)
                if fsim_dev is not None:
                    fsims.append(fsim_dev)

                losses.append(loss.item())

              

            x = model.get_trajectory()
            v, a = get_vel_acc(x)
            acc_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max), 2)))
            vel_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max), 2)))
            rec_loss = np.mean(losses)

            psnr = np.mean(psnrs)
            vif = np.mean(vifs) if len(vifs) else None
            fsim = np.mean(vifs) if len(fsims) else None
            

           

        x = model.get_trajectory()
        v, a = get_vel_acc(x)
  
        #plot_trajectories(x.cpu().detach(),args.exp_dir)
 
    pi=0
    if epoch and not epoch%10:
        save_comparison_images(decay[pi], target[pi], f"{args.exp_dir}/comps_val")
       
    return np.mean(losses), time.perf_counter() - start, psnr

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
        ax = fig.add_subplot(2, 1+x.shape[0]//2, frame + 1)
        for i in range(x.shape[1]):
            ax.plot(x[frame, i, :, 0], x[frame, i, :, 1], '.')
            ax.axis([-165, 165, -165, 165])
    return fig




def plot_trajectories(x, path):
    num_frames = x.shape[0]
    cols = 2  
    rows = math.ceil(num_frames / cols) 

    fig = plt.figure(figsize=[10, 10])
    for frame in range(num_frames):
        ax = fig.add_subplot(rows, cols, frame + 1)
        for i in range(x.shape[1]):
            ax.plot(x[frame, i, :, 0], x[frame, i, :, 1])
            ax.axis([-200, 200, -200, 200])
        ax.set_title(f"Frame {frame + 1}")
    plt.tight_layout() 
    plt.savefig(f"{str(path)}/trajs.png")
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


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, name=""):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir + f'/model_{name}.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir + f'/model_{name}.pt', exp_dir + f'/best_model_{name}.pt')


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


def load_model(checkpoint_file,args, rec=False):
    checkpoint = torch.load(checkpoint_file)
    #args = checkpoint['args']
    model = build_model(args, rec)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    
    optimizer = build_optim(args, model)
    #optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, model):
    
    
    optimizer = torch.optim.Adam([{'params': model.subsampling.parameters(), 'lr': args.sub_lr},
                                    {'params': model.reconstruction_model.parameters()}], args.lr)
    return optimizer


def train(args):
    args.v_max = args.gamma * args.G_max * args.FOV * args.dt
    args.a_max = args.gamma * args.S_max * args.FOV * args.dt ** 2 * 1e3

    args.exp_dir = f'summary/{args.test_name}'#f'/mnt/walkure_public/users/tamirs/T1_Mapping/maria_run_output/{os.getcwd().split("T1_mapping")[-1]}/summary/'#f'summary/{args.test_name}'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pathlib.Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir)
    with open(args.exp_dir + '/args.txt', "w") as text_file:
        print(vars(args), file=text_file)
    
    
    #if args.embedding_dim:
    #    emb_model = nn.Sequential(nn.Linear(1,args.embedding_dim),nn.ReLU(),nn.Linear(args.embedding_dim,args.embedding_dim*2)\
    #    ,nn.ReLU(),nn.Linear(args.embedding_dim*2,args.embedding_dim)).to(args.device)
    #else:
    #    emb_model = None
    
    if args.resume: #load trained model (for evaluation or to keep training)
      
        checkpoint, model, optimizer = load_model(args.checkpoint,args, rec=True)
        
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
        
        checkpoint, decay_model, _ = load_model(args.checkpoint_decay,args)
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
        
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model)
       
        start_epoch = 0
    logging.info(args)
    if args.opt_decay:
            optimizer.add_param_group({"params": decay_model.reconstruction_model.parameters(), "lr": args.lr})
            optimizer.add_param_group({"params": decay_model.embedding.parameters(), "lr": args.embedding_lr})
    train_loader, dev_loader = create_data_loaders(args)
   
    best_dev_loss = float('inf')
   
    for epoch in range(start_epoch, args.num_epochs):

       
       

        if epoch and not epoch % args.lr_step_size:
            optimizer.param_groups[1]['lr'] *= args.lr_gamma

        if epoch and not epoch % args.sub_lr_time:
            optimizer.param_groups[0]['lr'] = max(args.sub_lr_stepsize * optimizer.param_groups[0]['lr'], 5e-4)

        train_loss, train_time, train_rec_loss, train_vel_loss, train_acc_loss, psnr_train = train_epoch(
            args, epoch,
            model,
            decay_model,
            train_loader,
            optimizer,
            writer,
            len(train_loader))
        dev_loss, dev_time, psnr_dev = evaluate(args, epoch + 1, model,decay_model, dev_loader, writer, len(dev_loader),
                                                          train_loss,
                                                          train_rec_loss, train_vel_loss, train_acc_loss, psnr_train,
                                            )

        

        if dev_loss < best_dev_loss:
            is_new_best = True
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
        else:
            is_new_best = False
        save_model(args, args.exp_dir, epoch, decay_model, optimizer, best_dev_loss, is_new_best)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best,name="rec")
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s'
            f'DevPSNR= {psnr_dev:.4g}'
        )
        
    print(args.test_name)
    
    writer.close()

   

def run():
    args = create_arg_parser().parse_args()
    train(args)



def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')

    parser.add_argument('--data-path', type=pathlib.Path,
                      default='/home/tamir.shor/T1_mapping/FullSample', help='Path to the dataset')
    parser.add_argument('--sample-rate', type=float, default=1.,
                      help='Fraction of total volumes to include')
    parser.add_argument('--test-name', type=str, default='test/', help='name for the output dir')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='output/',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',default=True,
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, default= "/home/tamir.shor/T1_mapping/finetune_team_pilot/saved_weights_32/9_frames_32_shots_bs_12_u118/plain_finetune/summary/test/best_model.pt",help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--checkpoint-decay', type=str, default= "/home/tamir.shor/T1_mapping/flip_first_3/decay_model_no_subsample.pt",help='Path to an existing checkpoint. Used along with "--resume"')
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
    parser.add_argument('--batch-size', default=12, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=200, help='Number of training epochs per frame')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for reconstruction model')
    parser.add_argument('--lr-step-size', type=int, default=300,
                        help='Period of learning rate decay for reconstruction model')
    parser.add_argument('--lr-gamma', type=float, default=1,
                        help='Multiplicative factor of reconstruction model learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    # trajectory learning parameters
    parser.add_argument('--sub-lr', type=float, default=0.05, help='learning rate of the sub-samping layer')
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
    parser.add_argument('--boost', action='store_true', default=False, help='boost to equalize num examples per file')

    parser.add_argument('--project', action='store_true', default=True, help='Use projection to impose kinematic constraints.'
                                                                             'If false, use interpolation and penalty (original PILOT paper).')
    parser.add_argument('--proj_iters', default=10e1, help='Number of iterations for each projection run.')
    parser.add_argument('--single_traj', action='store_true', default=False, help='allow different trajectory per frame')
    parser.add_argument('--augment', action='store_true', default=True, help='Use augmented files. data-path argument from should lead to augmented '
                                                                             'files generated by the augment script. If false,'
                                                                             'path should lead to relevant ocmr dataset')
    parser.add_argument('--noise', action='store_true', default=False, help='add noise to traj.')
    parser.add_argument('--traj-dropout', action='store_true', default=False, help='randomly fix traj coordinates.')
    parser.add_argument('--recons_resets', action='store_true', default=False, help='Use Reconstruction Resets.')
    parser.add_argument('--traj_freeze', action='store_true', default=False, help='Use Trajectory Freezing.')
    parser.add_argument('--l1', action='store_true', default=False, help='Use l1 criterion.')
    parser.add_argument('--embedding-dim', type=int, default=0, help='TI time embedding dimension.')
    parser.add_argument('--embedding-lr', type=float, default=0.05, help='Embedding Layer lr.')
    parser.add_argument('--start-on-recons', type=int, default=float('inf'), help='Epoch from which decay loss is w.r.t reconstruction')
    parser.add_argument('--opt-decay', action='store_true', default=False, help='finetune frozen decay model')
    
    
    return parser


if __name__ == '__main__':
    run()


