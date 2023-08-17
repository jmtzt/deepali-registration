# Free-form deformation algorithm

Traditional pairwise registration using free-form deformation model.

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
