#!/bin/bash

# Define an array of model paths
base_path = '/home/joao/repos/ffd/'
dmmr_models_dir='/home/joao/repos/dmmr/outputs/dmmr_models'
model_paths=(
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs69_online_aug_extra_tfms.pt'
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs100_nonorm.pt'
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs100_norm.pt'
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs79_online_aug.pt'
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs64_online_aug.pt'
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs50_intersubject.pt'
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs50_intersubject.pt'
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs69_online_aug_extra_tfms.pt'  # -> same as single axis ro
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs69_online_aug_extra_tfms.pt'
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs100_online_aug_tuned_tfms.pt'  # -> no single axis rot
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs59_online_aug_tuned_tfms.pt'  # just 3 axis rot at same tim
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs100_online_aug_tuned_tfms_single_axis.pt'
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs34_online_aug_tuned_tfms_single_axis.pt'
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs54_online_aug_tuned_tfms_single_axis_small_rot.pt'
        ${dmmr_models_dir}'/camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs54_online_aug_tuned_tfms_single_axis_small_rot.pt'
#        ${dmmr_models_dir}'/camcan_t1t1_dmmr_net_sigmoid_bce_lr0.0001_epochs89_online_aug_tuned_tfms_single_axis_small_rot.pt'
#        ${dmmr_models_dir}'/camcan_t1t1_dmmr_net_tanh_hinge_lr0.0001_epochs89_online_aug_tuned_tfms_single_axis_small_rot.pt'
#        ${dmmr_models_dir}'/camcan_t1t1_dmmr_net_sigmoid_bce_lr0.0001_epochs19_online_aug_tuned_tfms_single_axis.pt'
#        ${dmmr_models_dir}'/camcan_t1t1_dmmr_net_tanh_hinge_lr0.0001_epochs59_online_aug_tuned_tfms_single_axis.pt'
)

for model_path in "${model_paths[@]}"; do
    if [ -f "$model_path" ]; then
        echo "Model path exists: $model_path"
        echo "Running register.py with model path: $model_path"
        python $base_path/register.py --dmmr_model_path "$model_path"
    else
        echo "Model path does not exist: $model_path... Skipping"
    fi
done