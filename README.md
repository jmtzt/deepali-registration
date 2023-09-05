# Free-form deformation algorithm

Traditional pairwise registration using free-form deformation model using the [Deepali](https://github.com/BioMedIA/deepali/) framework.

## Installation
1. Clone this repository
2. In a fresh Python 3.7+ virtual environment, install dependencies by running:
    ```
    pip install -r <path_to_cloned_repository>/requirements.txt
    ```

## Usage

The following command computes a free-form deformation which aligns a given moving image with a fixed image, and saves the result as dense displacement vector field.

```
CUDA_VISIBLE_DEVICES=0 python -m register.py \
    --target fixed_image.mha \
    --source moving_image.mha \
    --warped deformed_image.mha \
    --output displacement_field.mha \
    --device cuda \
    --verbose 1
```

See help output for a full list of command arguments, i.e.,

```
python -m register.py --help
```

## Output Directory Structure

Upon running the register.py script, the outputs will be saved in the outdir directory, which is located under the 
outputs/registration/%d%m%Y path. The directory structure is organized as follows:

```
outputs/
└── registration/
    └── %d%m%Y/          # Current date in the format DayMonthYear (e.g., 050923)
        ├── {hash}_level_0_steps_250_loss_values.pkl
        ├── {hash}_level_1_steps_250_loss_values.pkl
        ├── {hash}_level_2_steps_250_loss_values.pkl
        ├── ...             # Additional loss value files for each pyramid level
        ├── warped_{hash}.yaml # Configuration file for the registration
        ├── warped_vis_{hash}.png # Visualization of the before/after registration process
        ├── warped_seg_diff_{hash}.png # Segmentation difference image
        ├── warped_t1t1_diff_{hash}.png # T1-T1 difference image
        ├── warped_vis_deformation_{hash}.png # Deformation field image
        └── out_hashes.txt  # Hashes of all registration runs 
```

## Utility Scripts

- `scripts/dice_volume_crop.py`: Visualizes the dice scores for different center crops of the same volume. 
- `scripts/grid_search.sh`: Runs a grid search over the hyperparameters/loss functions with the registration script. 
- `scripts/parse_loss_curves.py`: Parses the .pkl loss files and generates visualizations of the loss curves. 
- `scripts/refined_grid_search.sh`: Runs a grid search over the hyperparameters/loss functions with the registration script, but uses the best hyperparameters from the previous grid search as the starting point for the new grid search. 
- `scripts/register_best_hparams.py`: Runs registration with the best set of hyperparameters for each of the loss functions. 
- `scripts/visualization.py`: Generates visualizations using the outputs of the grid search process.