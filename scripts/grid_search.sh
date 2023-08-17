#!/bin/bash

base_path = '/home/joao/repos/ffd/'
yaml_file="/home/joao/repos/ffd/params.yaml"
dmmr_models_dir='/home/joao/repos/dmmr/outputs/dmmr_models'
tanh_model_path="${dmmr_models_dir}/camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs54_online_aug_tuned_tfms_single_axis_small_rot.pt"
sigmoid_model_path="${dmmr_models_dir}/camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs59_online_aug_tuned_tfms_single_axis_small_rot_bound.pt"
output_file="/home/joao/repos/ffd/imgs/grid_search_results.csv"

# List of values to iterate over for the "reg" parameter
loss_values=("MSE" "NMI" "LNCC" "DMMR_TANH" "DMMR_SIGMOID")
reg_values=("0" "0.1" "0.01" "0.001" "0.0001" "0.00001" "0.000001" "0.0000001" "0.00000001" "0.000000001" "0.0000000001")
lr_values=("0.1" "0.01" "0.001" "0.0001")
be_values=("0" "0.1" "0.5" "0.01" "0.05" "0.001" "0.005" "0.0001" "0.0005" "0.00001" "0.00005")

echo "loss,model_path,lr,be,reg,before_reg_dice_class_1,before_reg_dice_class_2,before_reg_dice_class_3,before_reg_dice_class_4,before_reg_dice_class_5,before_reg_mean_dice,before_reg_rmse,after_reg_dice_class_1,after_reg_dice_class_2,after_reg_dice_class_3,after_reg_dice_class_4,after_reg_dice_class_5,after_reg_folding_ratio,after_reg_mag_det_jac_det,after_reg_mean_dice,after_reg_rmse" > "$output_file"

for loss_value in "${loss_values[@]}"; do
    if [[ $loss_value == "DMMR_TANH" ]]; then
        model_path="$tanh_model_path"
    elif [[ $loss_value == "DMMR_SIGMOID" ]]; then
        model_path="$sigmoid_model_path"
    else
        model_path=""
    fi
    yaml_loss_value=$(echo "$loss_value" | sed 's/_TANH\|_SIGMOID//g')

    for lr_value in "${lr_values[@]}"; do
        for be_value in "${be_values[@]}"; do
            for reg_value in "${reg_values[@]}"; do
                echo "------------------------------------------------------------------------------------"
                echo "Running with loss=${loss_value}, lr=${lr_value}, be=${be_value}, reg=${reg_value}..."

                # Replace the "seg" key w/ the loss value in the YAML file
                sed -i "s/\(seg:\s*\[.*\]\)/seg: [1, ${yaml_loss_value}]/" "$yaml_file"
                # Replace the "reg" value in the YAML file
                sed -i "s/\(reg:\s*\[.*L2Norm\]\)/reg: [${reg_value}, L2Norm]/" "$yaml_file"
                # Replace the "step_size" value in the YAML file
                sed -i "s/\(step_size:\s*\).*$/step_size: ${lr_value}/" "$yaml_file"
                # Replace the "be" value in the YAML file
                sed -i "s/\(be:\s*\[.*BSplineBending, stride: \[.*\]\]\)/be: [${be_value}, BSplineBending, stride: [*stride]]/" "$yaml_file"

                # Run the Python script with the updated YAML file and capture output
                script_output=$(python $base_path/register.py --dmmr_model_path "$model_path")

                # Capture the "Before registration metrics:" block
                before_metrics=$(echo "$script_output" | sed -n '/Before registration metrics:/,/After registration metrics:/ p' | sed -e '1d;$d')
                # Capture the "After registration metrics:" block
                after_metrics=$(echo "$script_output" | awk '/After registration metrics:/,/^[ \t]*$/ {if (!/After registration metrics:/)print}')

                # Extract values from before_metrics and after_metrics
                before_reg_dice_class_1=$(echo "$before_metrics" | grep 'dice_class_1' | awk '{print $2}' | sed 's/,//')
                before_reg_dice_class_2=$(echo "$before_metrics" | grep 'dice_class_2' | awk '{print $2}' | sed 's/,//')
                before_reg_dice_class_3=$(echo "$before_metrics" | grep 'dice_class_3' | awk '{print $2}' | sed 's/,//')
                before_reg_dice_class_4=$(echo "$before_metrics" | grep 'dice_class_4' | awk '{print $2}' | sed 's/,//')
                before_reg_dice_class_5=$(echo "$before_metrics" | grep 'dice_class_5' | awk '{print $2}' | sed 's/,//')
                before_reg_mean_dice=$(echo "$before_metrics" | grep 'mean_dice' | awk '{print $2}' | sed 's/,//')
                before_reg_rmse=$(echo "$before_metrics" | grep 'rmse' | awk '{print $2}' | sed 's/[},]//g')

                after_reg_dice_class_1=$(echo "$after_metrics" | grep 'dice_class_1' | awk '{print $2}' | sed 's/,//')
                after_reg_dice_class_2=$(echo "$after_metrics" | grep 'dice_class_2' | awk '{print $2}' | sed 's/,//')
                after_reg_dice_class_3=$(echo "$after_metrics" | grep 'dice_class_3' | awk '{print $2}' | sed 's/,//')
                after_reg_dice_class_4=$(echo "$after_metrics" | grep 'dice_class_4' | awk '{print $2}' | sed 's/,//')
                after_reg_dice_class_5=$(echo "$after_metrics" | grep 'dice_class_5' | awk '{print $2}' | sed 's/,//')
                after_reg_folding_ratio=$(echo "$after_metrics" | grep 'folding_ratio' | awk '{print $2}' | sed 's/,//')
                after_reg_mag_det_jac_det=$(echo "$after_metrics" | grep 'mag_det_jac_det' | awk '{print $2}' | sed 's/,//')
                after_reg_mean_dice=$(echo "$after_metrics" | grep 'mean_dice' | awk '{print $2}' | sed 's/,//')
                after_reg_rmse=$(echo "$after_metrics" | grep 'rmse' | awk '{print $2}' | sed 's/[},]//g')

                # Append the values to the CSV file
                echo "${loss_value},${model_path},${lr_value},${be_value},${reg_value},${before_reg_dice_class_1},${before_reg_dice_class_2},${before_reg_dice_class_3},${before_reg_dice_class_4},${before_reg_dice_class_5},${before_reg_mean_dice},${before_reg_rmse},${after_reg_dice_class_1},${after_reg_dice_class_2},${after_reg_dice_class_3},${after_reg_dice_class_4},${after_reg_dice_class_5},${after_reg_folding_ratio},${after_reg_mag_det_jac_det},${after_reg_mean_dice},${after_reg_rmse}" >> "$output_file"
            done
        done
    done
done
