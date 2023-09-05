import numpy as np
import torch
import copy
import SimpleITK as sitk
import matplotlib.pyplot as plt

from typing import Optional
from torch import Tensor
from matplotlib.figure import Figure

from deepali.data import Image
from deepali.core import functional as U
from deepali import spatial as spatial


def show_image(
        image: Tensor,
        label: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
) -> None:
    r"""Render image data in last two tensor dimensions using matplotlib.pyplot.imshow().

    Args:
        image: Image tensor of shape ``(..., H, W)``.
        ax: Figure axes to render the image in. If ``None``, a new figure is created.
        label: Image label to display in the axes title.
        kwargs: Keyword arguments to pass on to ``matplotlib.pyplot.imshow()``.
            When ``ax`` is ``None``, can contain ``figsize`` to specify the size of
            the figure created for displaying the image.

    """
    if ax is None:
        figsize = kwargs.pop("figsize", (4, 4))
        _, ax = plt.subplots(figsize=figsize)
    kwargs["cmap"] = kwargs.get("cmap", "gray")
    im = ax.imshow(image.reshape((-1,) + image.shape[-2:])[0].cpu().numpy(), **kwargs)
    if label:
        ax.set_title(label, fontsize=12, y=1.04)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    return im


def difference_image(image1, image2):
    image1 = image1.sitk()
    image2 = image2.sitk()

    # resample image2 if it does not occupy the same space as image1
    if not np.array_equal(image1.GetOrigin(), image2.GetOrigin()) or not np.array_equal(image1.GetSize(),
                                                                                        image2.GetSize()) or not np.array_equal(
        image1.GetSpacing(), image2.GetSpacing()):
        image2 = resample(image2, downsample=0, reference=image1, interp='linear')
    image1 = Image.from_sitk(image1)
    image2 = Image.from_sitk(image2)
    #
    # diff_image = sitk.Subtract(image1, image2)
    #
    # diff_image = Image.from_sitk(diff_image)

    diff_image = torch.abs(image1.data - image2.data)

    return diff_image


def resample(img, reference=None, new_size=None, new_spacing=None, transform=None, downsample=0, interp='linear'):
    """
    Resample an ITK image to either:
    a reference image given by an ITK image or,
    a desired voxel spacing in mm given by [spXmm, spYmm, spZmm] or,
    a desired size [x, y, z], or
    a downsampling factor (applied to the voxel spacing)

    Arguments
    ---------
    `reference` : ITK image
    `downsample` : scalar
    `out_spacing` : tuple or list (e.g [1.,1.,1.])
        New spacing in mm.
    `out_size` : tuple or list of ints (e.g. [100, 100, 100])
    `interp` : string or list/tuple of string
        possible values from this set: {'linear', 'nearest', 'bspline'}
        Different types of interpolation can be provided for each input,
        e.g. for two inputs, `interp=['linear','nearest']
    `transform`: spatial tranform
    """

    in_size = img.GetSize()
    in_spacing = img.GetSpacing()
    new_origin = img.GetOrigin()
    dim = img.GetDimension()

    if not downsample:
        if reference is not None:
            new_spacing = reference.GetSpacing()
            new_size = reference.GetSize()
            new_origin = reference.GetOrigin()

        else:
            # if new_size is None and new_spacing is None:
            # return img

            if new_size is None and new_spacing is not None:
                # Compute new image dimensions based on the desired rescaling of the voxel spacing
                new_size = [int(np.ceil(in_size[d] * in_spacing[d] / new_spacing[d])) for d in range(dim)]

            if new_spacing is None and new_size is not None:
                # Compute new voxel spacing based on the desired rescaling of the image dimensions
                new_spacing = [in_spacing[d] * in_size[d] / new_size[d] for d in range(dim)]
    else:
        new_spacing = [in_spacing[d] * downsample for d in range(dim)]
        new_size = [int(np.ceil(in_size[d] * in_spacing[d] / new_spacing[d])) for d in range(dim)]

    if new_spacing is None:
        new_spacing = in_spacing
    if new_size is None:
        new_size = in_size

    if transform is None:
        transform = sitk.Transform()

    if interp == 'linear':
        interp_func = sitk.sitkLinear
    elif interp == 'nearest':
        interp_func = sitk.sitkNearestNeighbor
    elif interp == 'bspline':
        interp_func = sitk.sitkBSpline

    # Smooth the input image with anisotropic Gaussian filter
    img_smoothed = img
    for d in range(dim):
        # Note how the blurring strength can be different in each direction,
        # if the scaling factors are different.
        factor = new_spacing[d] / in_spacing[d]
        sigma = 0.2 * factor
        img_smoothed = sitk.RecursiveGaussian(img_smoothed, sigma=sigma, direction=d)

    # Finally, apply the resampling operation
    # img_resampled = sitk.ResampleImageFilter().Execute(
    img_resampled = sitk.Resample(
        img_smoothed,  # Input image
        new_size,  # Output image dimensions
        transform,  # Coordinate transformation. sitk.Transform() is a dummy identity transform,
        # as we want the brain to be in exactly the same place. When we do image registration,
        # for example, this can be a linear or nonlinear transformation.
        interp_func,  # Interpolation method (cf. also sitk.sitkNearestNeighbor and many others)
        new_origin,  # Output image origin (same)
        new_spacing,  # Output voxel spacing
        img.GetDirection(),  # Output image orientation (same)
        0,  # Fill value for points outside the input domain
        img.GetPixelID())  # Voxel data type (same)

    return img_resampled


# noinspection PyNoneFunctionAssignment
def plot_images_and_diffs(target_img, source_img, warped_img, diff_start,
                          diff_end, metrics_before, metrics_after, dice_dict, suptitle, outfile, args):
    # Plot the images and difference maps
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    channel = target_img.shape[1] // 2

    show_image(target_img[0, channel],
               f"fixed \n{str(args.target_img.parent).split('/')[-1]}",
               ax=axes[0, 0])
    show_image(source_img[0, channel],
               f"moving \n{str(args.source_img.parent).split('/')[-1]}",
               ax=axes[0, 1])
    show_image(warped_img.detach()[0, channel], "warped", ax=axes[0, 2])

    err_before = show_image(diff_start[0, channel],
                            f"diff start \n"
                            f"mean dice {metrics_before['mean_dice']:.3f}\nrmse {metrics_before['rmse']:.3f}",
                            ax=axes[1, 0])
    cbar_err_before = plt.colorbar(err_before, ax=axes[1, 0])

    err_after = show_image(diff_end[0, channel],
                           f"diff end \n"
                           f"mean dice {metrics_after['mean_dice']:.3f}\nrmse {metrics_after['rmse']:.3f}",
                           ax=axes[1, 1])
    cbar_err_after = plt.colorbar(err_after, ax=axes[1, 1])

    x = np.arange(len(dice_dict['labels']))
    axes[1, 2].bar(x - 0.2, dice_dict['before_values'], width=0.4, label='Dice before registration', alpha=0.7)
    axes[1, 2].bar(x + 0.2, dice_dict['after_values'], width=0.4, label='Dice after registration', alpha=0.7)
    axes[1, 2].set_xticks(x + 0.1)
    axes[1, 2].set_xticklabels(dice_dict['labels'], rotation=45, ha='right')
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].legend()
    axes[1, 2].set_title('Dice scores')
    axes[1, 2].set_ylabel('Dice score')
    axes[1, 2].set_xlabel('Subclasses (classes 1-5)')

    plt.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()


def invertible_registration_figure(
    target: Tensor,
    source: Tensor,
    transform: spatial.SpatialTransform,
    title: str,
) -> Figure:
    r"""Create figure visualizing result of diffeomorphic registration.

    Args:
        target: Fixed target image.
        source: Moving source image.
        transform: Invertible spatial transform, i.e., must implement ``SpatialTransform.inverse()``.

    Returns:
        Instance of ``matplotlib.pyplot.Figure``.

    """
    device = transform.device

    highres_grid = transform.grid()#.resize(512) # TODO: resize this to grid.x*2, grid.y*2, grid.z*2
    grid_image = U.grid_image(highres_grid, num=1, stride=4, inverted=True, device=device)

    with torch.inference_mode():
        try:
            txt_inverse = ' (inverse)'
            inverse = transform.inverse()
        except NotImplementedError:
            txt_inverse = ' (no inverse)'
            inverse = transform

        source_transformer = spatial.ImageTransformer(transform)
        target_transformer = spatial.ImageTransformer(inverse)

        source_grid_transformer = spatial.ImageTransformer(transform, highres_grid, padding="zeros")
        target_grid_transformer = spatial.ImageTransformer(inverse, highres_grid, padding="zeros")

        warped_source: Tensor = source_transformer(source.to(device))
        warped_target: Tensor = target_transformer(target.to(device))

        warped_source_grid: Tensor = source_grid_transformer(grid_image)
        warped_target_grid: Tensor = target_grid_transformer(grid_image)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), tight_layout=True)

    show_image(target[0, target.shape[1] // 2], "target", ax=axes[0, 0])
    show_image(warped_source[0, warped_source.shape[1] // 2], "warped source", ax=axes[0, 1])
    show_image(warped_source_grid[0, 0, warped_source_grid.shape[2] // 2], "forward deformation", ax=axes[0, 2])

    show_image(source[0, source.shape[1] // 2], "source", ax=axes[1, 0])
    show_image(warped_target[0, warped_target.shape[1] // 2], f"warped target {txt_inverse}", ax=axes[1, 1])
    show_image(warped_target_grid[0, 0, warped_target_grid.shape[2] // 2], "inverse deformation", ax=axes[1, 2])
    # plt.suptitle(title)
    plt.savefig(title)
    plt.show()

    return fig


def plot_t1t1_diff(target, source, transform, outfile):
    device = transform.device

    highres_grid = transform.grid()
    grid_image = U.grid_image(highres_grid, num=1, stride=4, inverted=True, device=device)

    with torch.inference_mode():

        source_transformer = spatial.ImageTransformer(transform)

        source_grid_transformer = spatial.ImageTransformer(transform, highres_grid, padding="zeros")

        warped_source: Tensor = source_transformer(source.to(device))

        warped_source_grid: Tensor = source_grid_transformer(grid_image)

        # target_norm = U.normalize_image(data=target.data, min=0, max=1)
        # warped_source_norm = U.normalize_image(data=warped_source, min=0, max=1)
        # warped_source_norm_img = copy.deepcopy(target_norm)
        # warped_source_norm_img.data = warped_source_norm
        # diff = difference_image(target_norm, warped_source_norm_img)
        warped_source_img = copy.deepcopy(target)
        warped_source_img.data = warped_source
        diff = difference_image(target, warped_source_img)

    fig, axes = plt.subplots(1, 5, figsize=(12, 8), tight_layout=True)

    ch = target.shape[1] // 2
    show_image(target[0, ch], "target", ax=axes[0])
    show_image(source[0, ch], "t1-source", ax=axes[1])
    show_image(warped_source[0, ch], "warped t1-source", ax=axes[2])
    show_image(warped_source_grid[0, 0, ch], "deformation", ax=axes[3])
    dff_img = show_image(diff[0, ch], label="difference", ax=axes[4])
    plt.colorbar(dff_img, ax=axes[4], shrink=0.3)

    # plt.suptitle(title)
    plt.savefig(outfile)
    plt.show()


def plot_seg_diff(target_seg, source_seg, warped_source_seg, outfile):
    diff_og = torch.abs(target_seg - source_seg)
    diff_after = torch.abs(target_seg - warped_source_seg)
    diff_tt = torch.abs(target_seg - target_seg)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), tight_layout=True)

    show_image(target_seg[0, target_seg.shape[1] // 2], label='target_seg', ax=axes[0,0])
    show_image(source_seg[0, source_seg.shape[1] // 2], label='source_seg', ax=axes[0,1])
    show_image(warped_source_seg[0, warped_source_seg.shape[1] // 2], label='warped_source_seg', ax=axes[0,2])
    diff_targ_targ = show_image(diff_tt[0, diff_tt.shape[1] // 2], label='diff target-target', ax=axes[1,0])
    plt.colorbar(diff_targ_targ, ax=axes[1,0], shrink=0.5)
    diff_og_img = show_image(diff_og[0, diff_og.shape[1] // 2], label='diff target-source', ax=axes[1,1])
    plt.colorbar(diff_og_img, ax=axes[1,1], shrink=0.5)
    diff_after_img = show_image(diff_after[0, diff_after.shape[1] // 2], label='diff target-warped-source', ax=axes[1,2])
    plt.colorbar(diff_after_img, ax=axes[1,2], shrink=0.5)

    plt.savefig(outfile)

    plt.show()
