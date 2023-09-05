import os
import torch
import matplotlib.pyplot as plt

from typing import Any, Callable, Optional, Tuple, Type, cast
from torch import Tensor, optim
from tqdm.auto import tqdm
from pprint import pprint

from deepali.core.environ import cuda_visible_devices
from deepali.core.image import center_crop
from deepali.data import Image
from deepali.losses import functional as L

from losses import DMMRLoss
from utils import show_image
from metric import measure_seg_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Use first device specified in CUDA_VISIBLE_DEVICES if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() and cuda_visible_devices() else "cpu")

target_path = 'imgs/sub-CC110101/T1_brain.nii.gz'
source_path = 'imgs/sub-CC110062/T2_brain.nii.gz'
target_seg_path = 'imgs/sub-CC110101/T1_brain_MALPEM_tissues.nii.gz'
source_seg_path = 'imgs/sub-CC110062/T1_brain_MALPEM_tissues.nii.gz'

target = Image.from_uri(target_path, device=device)
source = Image.from_uri(source_path, device=device)
target_seg = Image.from_uri(target_seg_path, device=device)
source_seg = Image.from_uri(source_seg_path, device=device)

target_batch = target.unsqueeze(0)  # (N, C, Y, X)
source_batch = source.unsqueeze(0)  # (N, C, Y, X)
target_seg_batch = target_seg.unsqueeze(0)  # (N, C, Y, X)
source_seg_batch = source_seg.unsqueeze(0)  # (N, C, Y, X)

print(f"target_batch.shape: {target_batch.shape}")
print(f"source_batch.shape: {source_batch.shape}")
print(f"target_seg_batch.shape: {target_seg_batch.shape}")
print(f"source_seg_batch.shape: {source_seg_batch.shape}")

nmi = L.nmi_loss

# model_path = "/home/joao/repos/dmmr/outputs/dmmr_models/camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs59_online_aug_tuned_tfms_single_axis_small_rot_bound.pt"
model_path = "/home/joao/repos/dmmr/outputs/dmmr_models/camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs54_online_aug_tuned_tfms_single_axis_small_rot.pt"
dmmr = DMMRLoss(model_path=model_path)

nmi_loss_before = nmi(source_batch, target_batch)
dmmr_loss_before = dmmr(source_batch, target_batch)

print(f"nmi_loss_before: {nmi_loss_before}")
print(f"dmmr_loss_before: {dmmr_loss_before}")

out_size = 189

source_batch = center_crop(source_batch, out_size)
target_batch = center_crop(target_batch, out_size)
source_seg_batch = center_crop(source_seg_batch, out_size)
target_seg_batch = center_crop(target_seg_batch, out_size)

seg_dict = {
    'target_seg': target_seg_batch.squeeze(0).cpu().numpy(),
    'warped_source_seg': source_seg_batch.squeeze(0).cpu().numpy(),
}

metrics_before = measure_seg_metrics(seg_dict)
# metrics_before = {k: round(v, 5) for k, v in metrics_before.items()}
pprint(metrics_before)

out_size = 50

source_batch_cropped = center_crop(source_batch, out_size)
target_batch_cropped = center_crop(target_batch, out_size)
source_seg_batch_cropped = center_crop(source_seg_batch, out_size)
target_seg_batch_cropped = center_crop(target_seg_batch, out_size)

print(f"source_batch_cropped.shape: {source_batch_cropped.shape}")
print(f"target_batch_cropped.shape: {target_batch_cropped.shape}")
print(f"source_seg_batch_cropped.shape: {source_seg_batch_cropped.shape}")
print(f"target_seg_batch_cropped.shape: {target_seg_batch_cropped.shape}")

seg_dict = {
    'target_seg': target_seg_batch_cropped.squeeze(0).cpu().numpy(),
    'warped_source_seg': source_seg_batch_cropped.squeeze(0).cpu().numpy(),
}

metrics_after = measure_seg_metrics(seg_dict)
# metrics_after = {k: round(v, 5) for k, v in metrics_after.items()}
pprint(metrics_after)

nmi_loss_after = nmi(source_batch_cropped, target_batch_cropped)
dmmr_loss_after = dmmr(source_batch_cropped, target_batch_cropped)

print(f"nmi_loss_after crop ({out_size}^3): {nmi_loss_after}")
print(f"dmmr_loss_after crop ({out_size}^3): {dmmr_loss_after}")

fig, axes = plt.subplots(1, 4, figsize=(12, 8), tight_layout=True)
original_center_index = target_batch.shape[2] // 2
cropped_center_index = target_batch_cropped.shape[2] // 2
original_center_index_source = source_batch.shape[2] // 2
cropped_center_index_source = source_batch_cropped.shape[2] // 2

original_middle_slice = target_batch[0, 0, original_center_index]
cropped_middle_slice = target_batch_cropped[0, 0, cropped_center_index]
original_middle_slice_src = source_batch[0, 0, original_center_index_source]
cropped_middle_slice_src = source_batch_cropped[0, 0, cropped_center_index_source]

show_image(original_middle_slice, f"target\nNMI: {nmi_loss_before:.2f}\nDMMR: {dmmr_loss_before:.5f}", ax=axes[0])
show_image(original_middle_slice_src, f"source\n"
                                      f"Mean Dice: {metrics_before['mean_dice']:.5f}\n"
                                      f"Dice-1: {metrics_before['dice_class_1.0']:.5f}\n"
                                      f"Dice-2: {metrics_before['dice_class_2.0']:.5f}\n"
                                      f"Dice-3: {metrics_before['dice_class_3.0']:.5f}\n"
                                      f"Dice-4: {metrics_before['dice_class_4.0']:.5f}\n"
                                      f"Dice-5: {metrics_before['dice_class_5.0']:.5f}\n"
           , ax=axes[1])
show_image(cropped_middle_slice, f"target cropped\nNMI: {nmi_loss_after:.5f}\nDMMR: {dmmr_loss_after:.5f}", ax=axes[2])
show_image(cropped_middle_slice_src, f"source cropped\n"
                                     f"Mean Dice: {metrics_after['mean_dice']:.5f}\n"
                                     f"Dice-1: {metrics_after['dice_class_1.0']:.5f}\n"
                                     f"Dice-2: {metrics_after['dice_class_2.0']:.5f}\n"
                                     f"Dice-3: {metrics_after['dice_class_3.0']:.5f}\n"
                                     f"Dice-4: {metrics_after['dice_class_4.0']:.5f}\n"
                                     f"Dice-5: {metrics_after['dice_class_5.0']:.5f}\n"
           , ax=axes[3])

plt.show()
