import pathlib
import random
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
from transforms import complex_abs

def load_sample(path, slice_idx, transform, device="cuda"):

    f = h5py.File(f"{path}/T1map.mat","r")['kspace_single_full']   
    data = f['real'] + f['imag']*1j
    data = transform(data)
    sample =  data[:,slice_idx].to(device)
   
    return sample


def flip_per_pixel_for_monotonicity(img_batch):
    """
    For each pixel in the batch, find the best flip index to maximize monotonicity 
    along the channel axis and apply the optimal per-pixel flip.

    Parameters:
    img_batch (torch.Tensor): Shape (B, 9, H, W), where B is the batch size.

    Returns:
    torch.Tensor: The batch with optimal flips applied per pixel.
    """
    assert img_batch.dim() == 4 and img_batch.shape[1] == 9, "Input must have shape (B, 9, H, W)."

    B, C, H, W = img_batch.shape

    # Expand image batch to evaluate all flip possibilities (B, 9, 9, H, W)
    img_expanded = img_batch.unsqueeze(1).expand(-1, C, -1, -1, -1)

    # Generate a flip mask per pixel for each possible flip index
    flip_mask = torch.arange(C, device=img_batch.device).view(1, C, 1, 1, 1) > torch.arange(C, device=img_batch.device).view(1, 1, C, 1, 1)
    flip_mask = flip_mask.expand(B, C, C, H, W)

    # Apply flipping to create flipped versions for each possible t
    flipped_versions = img_expanded.clone()
    flipped_versions[flip_mask] *= -1

    # Compute monotonicity violations (negative differences along the channel axis)
    diffs = flipped_versions[:, :, 1:] - flipped_versions[:, :, :-1]  # Shape (B, 9, 8, H, W)
    violations = (diffs < 0).sum(dim=2)  # Sum violations across channels -> (B, 9, H, W)

    # Find the best per-pixel flip index t that minimizes violations
    best_t = violations.argmin(dim=1)  # Shape (B, H, W)

    # Construct a final per-pixel flip mask
    per_pixel_flip_mask = torch.arange(C, device=img_batch.device).view(1, C, 1, 1) < best_t.unsqueeze(1)  # (B, 9, H, W)
    return per_pixel_flip_mask

def flip_batch_for_monotonicity(img_batch):
    """
    For each tensor in a batch, find the best flip index to maximize monotonicity 
    along the channel axis and apply the optimal flip.

    Parameters:
    img_batch (torch.Tensor): Shape (B, 9, H, W), where B is the batch size.

    Returns:
    torch.Tensor: The batch with optimal flips applied.
    """
    assert img_batch.dim() == 4 and img_batch.shape[1] == 9, "Input must have shape (B, 9, H, W)."

    B, C, H, W = img_batch.shape

    # Generate all possible flipped versions for every index 0 to 8
    img_expanded = img_batch.unsqueeze(1).expand(-1, C, -1, -1, -1)  # (B, 9, 9, H, W)
    
    # Create a flip mask with the correct shape
    flip_mask = torch.arange(C, device=img_batch.device).view(1, C, 1, 1, 1) > torch.arange(C, device=img_batch.device).view(1, 1, C, 1, 1)
    
    # Broadcast flip_mask properly to match img_expanded
    flip_mask = flip_mask.expand(B, C, C, H, W)  # Now shape (B, 9, 9, H, W)

    # Apply flipping
    flipped_versions = img_expanded.clone()
    flipped_versions[flip_mask] *= -1  # Flip elements up to index i

    # Compute monotonicity violations (negative differences along channel axis)
    diffs = flipped_versions[:, :, 1:] - flipped_versions[:, :, :-1]  # (B, 9, 8, H, W)
    violations = (diffs < 0).sum(dim=(2, 3, 4))  # Sum violations -> Shape (B, 9)

    # Find the best flip index per batch element
    best_i = violations.argmin(dim=1)  # Shape (B,)

    # Create final flip mask
    flip_mask= torch.arange(C, device=img_batch.device).view(1, C, 1, 1) < best_i.view(-1, 1, 1, 1)  # Shape (B, 9, H, W)


    
    

    return flip_mask
    
class SliceData(Dataset):
    def __init__(self, files, transform, flip_load_path, num_frames_per_example=9):
        
        
        self.examples = []
        self.transform = transform
        self.flip_idxs_file = pkl.load(open(flip_load_path,'rb')) if flip_load_path is not None else None
        #for fname in sorted(files):
        #    samples = None
        #    for slice in range(5):
        #        self.examples += [(fname,slice)]
                
               
            
            #flip = flip_per_pixel_for_monotonicity(samples)
            #flip[:,6:] = False #can't flip after 6, according to challenge github
            #flips[fname] = flip.cpu().detach().numpy()
            #print(len(flips))
            
           
        examples = pkl.load(open("/home/tamir.shor/T1_mapping/example_list.pkl",'rb'))#np.random.shuffle(self.examples)#random.shuffle(self.examples)
        self.examples = [(example[0].replace("/mnt/walkure_public/users/tamirs/T1_Mapping/ChallengeData/SingleCoil/Mapping/TrainingSet/FullSample","/home/tamir.shor/T1_mapping/FullSample"),example[1]) for example in examples]
     
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice = self.examples[i]
        
        sample = load_sample(fname, slice, self.transform, 'cpu')
        target = complex_abs(sample)
        if self.flip_idxs_file is None:
            target[:3] *= -1
            #sample[:3] *= -1
        else:
            target[self.flip_idxs_file[fname][slice]] *= -1
        #target[:flip_idxs] *= -1
   
        tis_df = pd.read_csv(f'{fname}/T1map.csv')
        tis = torch.Tensor(tis_df.iloc[:,slice+1].values)[:,None,None]/1000 #work in secs
        return sample,tis, target

