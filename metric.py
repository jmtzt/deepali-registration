import numpy as np
import pandas as pd
import torch
import cv2
import omegaconf
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk


def measure_metrics(metric_data, metric_groups, return_tensor=False):
    """
    Wrapper function for calculating all metrics
    Args:
        metric_data: (dict) data used for calculation of metrics, could be Tensor or Numpy Array
        metric_groups: (list of strings) name of metric groups
        return_tensor: (bool) return Torch Tensor if True

    Returns:
        metrics_results: (dict) {metric_name: metric_value}
    """

    # cast Tensor to Numpy Array if needed
    for k, x in metric_data.items():
        if isinstance(x, torch.Tensor):
            metric_data[k] = x.cpu().numpy()

    # keys must match metric_groups and params.metric_groups
    # (using groups to share pre-scripts)
    metric_group_fns = {'disp_metrics': measure_disp_metrics,
                        'image_metrics': measure_image_metrics,
                        'seg_metrics': measure_seg_metrics,}

    metric_results = dict()
    for group in metric_groups:
        metric_results.update(metric_group_fns[group](metric_data))

    # cast Numpy arrary to Tensor if needed
    if return_tensor:
        for k, x in metric_results.items():
            metric_results[k] = torch.tensor(x)

    return metric_results


"""
Functions calculating groups of metrics
"""


def measure_disp_metrics(metric_data):
    """
    Calculate DVF-related metrics.
    If roi_mask is given, the disp is masked and only evaluate in the bounding box of the mask.

    Args:
        metric_data: (dict)

    Returns:
        metric_results: (dict)
    """
    # new object to avoid changing data in metric_data
    disp_pred = metric_data['disp_pred']
    if 'disp_gt' in metric_data.keys():
        disp_gt = metric_data['disp_gt']

    # mask the disp with roi mask if given
    if 'roi_mask' in metric_data.keys():
        roi_mask = metric_data['roi_mask']  # (N, 1, *(sizes))

        # find roi mask bbox mask
        mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])

        # mask and bbox crop dvf gt and pred by roi_mask
        disp_pred = disp_pred * roi_mask
        disp_pred = bbox_crop(disp_pred, mask_bbox)

        if 'disp_gt' in metric_data.keys():
            disp_gt = disp_gt * roi_mask
            disp_gt = bbox_crop(disp_gt, mask_bbox)

    # Regularity (Jacobian) metrics
    folding_ratio, mag_det_jac_det = calculate_jacobian_metrics(disp_pred)

    disp_metric_results = dict()
    disp_metric_results.update({'folding_ratio': folding_ratio,
                               'mag_det_jac_det': mag_det_jac_det})

    # DVF accuracy metrics if ground truth is available
    if 'disp_gt' in metric_data.keys():
        disp_metric_results.update({'aee': calculate_aee(disp_pred, disp_gt),
                                   'rmse_disp': calculate_rmse_disp(disp_pred, disp_gt)})
    return disp_metric_results


def measure_image_metrics(metric_data):
    # unpack metric data, keys must match metric_data input
    img = metric_data['target']
    img_pred = metric_data['target_pred']  # (N, 1, *sizes)

    # crop out image by the roi mask bounding box if given
    if 'roi_mask' in metric_data.keys():
        roi_mask = metric_data['roi_mask']
        mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])
        img = bbox_crop(img, mask_bbox)
        img_pred = bbox_crop(img_pred, mask_bbox)

    return {'rmse': calculate_rmse(img, img_pred)}


def measure_seg_metrics(metric_data):
    """ Calculate segmentation """
    seg_gt = metric_data['target_seg']
    seg_pred = metric_data['warped_source_seg']
    seg_gt = seg_gt[np.newaxis, ...]
    seg_pred = seg_pred[np.newaxis, ...]
    assert seg_gt.ndim == seg_pred.ndim
    assert seg_gt.ndim in (4, 5)  # (N, 1, *2D sizes) or (N, 1, *3D sizes)

    results = dict()
    for label_cls in np.unique(seg_gt):
        # calculate DICE score for each class
        if label_cls == 0:
            # skip background
            continue
        results[f'dice_class_{label_cls}'] = calculate_dice(seg_gt, seg_pred, label_class=label_cls)

    # calculate mean dice
    results['mean_dice'] = np.mean([dice for k, dice in results.items()])
    return results


"""
Functions calculating individual metrics
"""


def calculate_aee(x, y):
    """
    Average End point Error (AEE, mean over point-wise L2 norm)
    Input DVF shape: (N, dim, *(sizes))
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1)).mean()


def calculate_rmse_disp(x, y):
    """
    RMSE of DVF (square root over mean of sum squared)
    Input DVF shape: (N, dim, *(sizes))
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1).mean())


def calculate_rmse(x, y):
    """Standard RMSE formula, square root over mean
    (https://wikimedia.org/api/rest_v1/media/math/render/svg/6d689379d70cd119e3a9ed3c8ae306cafa5d516d)
    """
    return np.sqrt(((x - y) ** 2).mean())


def calculate_jacobian_metrics(disp):
    """
    Calculate Jacobian related regularity metrics.

    Args:
        disp: (numpy.ndarray, shape (N, ndim, *sizes) Displacement field

    Returns:
        folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
        mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
    """
    folding_ratio = []
    mag_grad_jac_det = []
    for n in range(disp.shape[0]):
        disp_n = np.moveaxis(disp[n, ...], 0, -1)  # (*sizes, ndim)
        jac_det_n = calculate_jacobian_det(disp_n)
        folding_ratio += [(jac_det_n < 0).sum() / np.prod(jac_det_n.shape)]
        mag_grad_jac_det += [np.abs(np.gradient(jac_det_n)).mean()]
    return np.mean(folding_ratio), np.mean(mag_grad_jac_det)


def calculate_jacobian_det(disp):
    """
    Calculate Jacobian determinant of displacement field of one image/volume (2D/3D)

    Args:
        disp: (numpy.ndarray, shape (*sizes, ndim)) Displacement field

    Returns:
        jac_det: (numpy.adarray, shape (*sizes) Point-wise Jacobian determinant
    """
    disp_img = sitk.GetImageFromArray(disp, isVector=True)
    jac_det_img = sitk.DisplacementFieldJacobianDeterminant(disp_img)
    jac_det = sitk.GetArrayFromImage(jac_det_img)
    return jac_det


def calculate_dice(mask1, mask2, label_class=0):
    """
    Dice score of a specified class between two label masks.
    (classes are encoded but by label class number not one-hot )

    Args:
        mask1: (numpy.array, shape (N, 1, *sizes)) segmentation mask 1
        mask2: (numpy.array, shape (N, 1, *sizes)) segmentation mask 2
        label_class: (int or float)

    Returns:
        volume_dice
    """
    mask1_pos = (mask1 == label_class).astype(np.float32)
    mask2_pos = (mask2 == label_class).astype(np.float32)

    assert mask1.ndim == mask2.ndim
    axes = tuple(range(2, mask1.ndim))
    pos1and2 = np.sum(mask1_pos * mask2_pos, axis=axes)
    pos1 = np.sum(mask1_pos, axis=axes)
    pos2 = np.sum(mask2_pos, axis=axes)
    return np.mean(2 * pos1and2 / (pos1 + pos2 + 1e-7))


def contour_distances_2d(image1, image2, dx=1):
    """
    Calculate contour distances between binary masks.
    The region of interest must be encoded by 1

    Args:
        image1: 2D binary mask 1
        image2: 2D binary mask 2
        dx: physical size of a pixel (e.g. 1.8 (mm) for UKBB)

    Returns:
        mean_hausdorff_dist: Hausdorff distance (mean if input are 2D stacks) in pixels
    """

    # Retrieve contours as list of the coordinates of the points for each contour
    # convert to contiguous array and data type uint8 as required by the cv2 function
    image1 = np.ascontiguousarray(image1, dtype=np.uint8)
    image2 = np.ascontiguousarray(image2, dtype=np.uint8)

    # extract contour points and stack the contour points into (N, 2)
    contours1, _ = cv2.findContours(image1.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour1_pts = np.array(contours1[0])[:, 0, :]
    for i in range(1, len(contours1)):
        cont1_arr = np.array(contours1[i])[:, 0, :]
        contour1_pts = np.vstack([contour1_pts, cont1_arr])

    contours2, _ = cv2.findContours(image2.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour2_pts = np.array(contours2[0])[:, 0, :]
    for i in range(1, len(contours2)):
        cont2_arr = np.array(contours2[i])[:, 0, :]
        contour2_pts = np.vstack([contour2_pts, cont2_arr])

    # distance matrix between two point sets
    dist_matrix = np.zeros((contour1_pts.shape[0], contour2_pts.shape[0]))
    for i in range(contour1_pts.shape[0]):
        for j in range(contour2_pts.shape[0]):
            dist_matrix[i, j] = np.linalg.norm(contour1_pts[i, :] - contour2_pts[j, :])

    # symmetrical mean contour distance
    mean_contour_dist = 0.5 * (np.mean(np.min(dist_matrix, axis=0)) + np.mean(np.min(dist_matrix, axis=1)))

    # calculate Hausdorff distance using the accelerated method
    # (doesn't really save computation since pair-wise distance matrix has to be computed for MCD anyways)
    hausdorff_dist = directed_hausdorff(contour1_pts, contour2_pts)[0]

    return mean_contour_dist * dx, hausdorff_dist * dx


def contour_distances_stack(stack1, stack2, label_class, dx=1):
    """
    Measure mean contour distance metrics between two 2D stacks

    Args:
        stack1: stack of binary 2D images, shape format (W, H, N)
        stack2: stack of binary 2D images, shape format (W, H, N)
        label_class: class of which to calculate distance
        dx: physical size of a pixel (e.g. 1.8 (mm) for UKBB)

    Return:
        mean_mcd: mean contour distance averaged over non-empty slices
        mean_hd: Hausdorff distance averaged over non-empty slices
    """

    # assert the two stacks has the same number of slices
    assert stack1.shape[-1] == stack2.shape[-1], 'Contour dist error: two stacks has different number of slices'

    # mask by class
    stack1 = (stack1 == label_class).astype('uint8')
    stack2 = (stack2 == label_class).astype('uint8')

    mcd_buffer = []
    hd_buffer = []
    for slice_idx in range(stack1.shape[-1]):
        # ignore empty masks
        if np.sum(stack1[:, :, slice_idx]) > 0 and np.sum(stack2[:, :, slice_idx]) > 0:
            slice1 = stack1[:, :, slice_idx]
            slice2 = stack2[:, :, slice_idx]
            mcd, hd = contour_distances_2d(slice1, slice2, dx=dx)

            mcd_buffer += [mcd]
            hd_buffer += [hd]

    return np.mean(mcd_buffer), np.mean(hd_buffer)


class MetricReporter(object):
    """
    Collect and report values
        self.collect_value() collects value in `report_data_dict`, which is structured as:
            self.report_data_dict = {'value_name_A': [A1, A2, ...], ... }

        self.summarise() construct the report dictionary if called, which is structured as:
            self.report = {'value_name_A': {'mean': A_mean,
                                            'std': A_std,
                                            'list': [A1, A2, ...]}
                            }
    """
    def __init__(self, id_list, save_dir, save_name='analysis_results'):
        self.id_list = id_list
        self.save_dir = save_dir
        self.save_name = save_name

        self.report_data_dict = {}
        self.report = {}

    def reset(self):
        self.report_data_dict = {}
        self.report = {}

    def collect(self, x):
        for name, value in x.items():
            if name not in self.report_data_dict.keys():
                self.report_data_dict[name] = []
            self.report_data_dict[name].append(value)

    def summarise(self):
        # summarise aggregated results to form the report dict
        for name in self.report_data_dict:
            self.report[name] = {
                'mean': np.mean(self.report_data_dict[name]),
                'std': np.std(self.report_data_dict[name]),
                'list': self.report_data_dict[name]
            }

    def save_mean_std(self):
        report_mean_std = {}
        for metric_name in self.report:
            report_mean_std[metric_name + '_mean'] = self.report[metric_name]['mean']
            report_mean_std[metric_name + '_std'] = self.report[metric_name]['std']
        # save to CSV
        csv_path = self.save_dir + f'/{self.save_name}.csv'
        save_dict_to_csv(report_mean_std, csv_path)

    def save_df(self):
        # method_column = [str(model_name)] * len(self.id_list)
        # df_dict = {'Method': method_column, 'ID': self.id_list}
        df_dict = {'ID': self.id_list}
        for metric_name in self.report:
            df_dict[metric_name] = self.report[metric_name]['list']

        df = pd.DataFrame(data=df_dict)
        df.to_pickle(self.save_dir + f'/{self.save_name}_df.pkl')


def bbox_crop(x, bbox):
    """
    Crop image by slicing using bounding box indices (2D/3D)

    Args:
        x: (numpy.ndarray, shape (N, ch, *dims))
        bbox: (list of tuples) [*(bbox_min_index, bbox_max_index)]

    Returns:
        x cropped using bounding box
    """
    # slice all of batch and channel
    slicer = [slice(0, x.shape[0]), slice(0, x.shape[1])]

    # slice image dimensions
    for bb in bbox:
        slicer.append(slice(*bb))
    return x[tuple(slicer)]


def bbox_from_mask(mask, pad_ratio=0.2):
    """
    Find a bounding box indices of a mask (with positive > 0)
    The output indices can be directly used for slicing
    - for 2D, find the largest bounding box out of the N masks
    - for 3D, find the bounding box of the volume mask

    Args:
        mask: (numpy.ndarray, shape (N, H, W) or (N, H, W, D)
        pad_ratio: (int or tuple) the ratio of between the mask bounding box to image boundary to pad

    Return:
        bbox: (list of tuples) [*(bbox_min_index, bbox_max_index)]
        bbox_mask: (numpy.ndarray shape (N, mH, mW) or (N, mH, mW, mD)) binary mask of the bounding box
    """
    dim = mask.ndim - 1
    mask_shape = mask.shape[1:]
    pad_ratio = param_ndim_setup(pad_ratio, dim)

    # find non-zero locations in the mask
    nonzero_indices = np.nonzero(mask > 0)
    bbox = [(nonzero_indices[i + 1].min(), nonzero_indices[i + 1].max())
            for i in range(dim)]

    # pad pad_ratio of the minimum distance
    #  from mask bounding box to the image boundaries (half each side)
    for i in range(dim):
        if pad_ratio[i] > 1:
            print(f"Invalid padding value (>1) on dimension {dim}, set to 1")
            pad_ratio[i] = 1
    bbox_padding = [pad_ratio[i] * min(bbox[i][0], mask_shape[i] - bbox[i][1])
                    for i in range(dim)]
    # "padding" by modifying the bounding box indices
    bbox = [(bbox[i][0] - int(bbox_padding[i]/2), bbox[i][1] + int(bbox_padding[i]/2))
            for i in range(dim)]

    # bbox mask
    bbox_mask = np.zeros(mask.shape, dtype=np.float32)
    slicer = [slice(0, mask.shape[0])]  # all slices/batch
    for i in range(dim):
        slicer.append(slice(*bbox[i]))
    bbox_mask[tuple(slicer)] = 1.0
    return bbox, bbox_mask


def param_ndim_setup(param, ndim):
    """
    Check dimensions of paramters and extend dimension if needed.

    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        ndim: (int) data/model dimension

    Returns:
        param: (tuple)
    """
    if isinstance(param, (int, float)):
        param = (param,) * ndim
    elif isinstance(param, (tuple, list, omegaconf.listconfig.ListConfig)):
        assert len(param) == ndim, \
            f"Dimension ({ndim}) mismatch with data"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param


def save_dict_to_csv(d, csv_path, model_name='modelX'):
    for k, x in d.items():
        if not isinstance(x, list):
            d[k] = [x]
    pd.DataFrame(d, index=[model_name]).to_csv(csv_path)


def prepare_and_measure_metrics(metric_data_raw, metric_groups, transform=None, warped_image=None,
                                warped_image_seg=None):
    # Update metric data if needed (after registration case)
    if transform is not None:
        metric_data_raw.update({
            'disp_pred': transform.tensor(),
            'target_pred': warped_image,
            'warped_source_seg': warped_image_seg.int(),
        })

    # Convert tensor data to numpy and detach from computation graph
    metric_data = {key: value.detach().cpu().numpy() for key, value in metric_data_raw.items()}

    # Measure metrics
    metrics_results = measure_metrics(metric_data, metric_groups, return_tensor=False)

    return metrics_results