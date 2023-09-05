r"""Example implementation of free-form deformation (FFD) algorithm."""
import logging
import os
from pathlib import Path
from pprint import pprint
import sys

import numpy as np
import yaml
import matplotlib.pyplot as plt
import hashlib
from timeit import default_timer as timer
from typing import Any, Dict
from datetime import datetime

import json
import yaml

import torch
import torch.cuda
from torch import Tensor

from deepali.core.argparse import ArgumentParser, Args, main_func
from deepali.core.environ import cuda_visible_devices
from deepali.core.grid import Grid
from deepali.core.logging import configure_logging
from deepali.core.pathlib import PathStr, unlink_or_mkdir
from deepali.data import Image
from deepali.modules import TransformImage

from pairwise import register_pairwise
from metric import prepare_and_measure_metrics
from utils import difference_image, show_image, plot_images_and_diffs, invertible_registration_figure, plot_t1t1_diff, \
    plot_seg_diff

log = logging.getLogger()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parser(**kwargs) -> ArgumentParser:
    r"""Construct argument parser."""
    if "description" not in kwargs:
        kwargs["description"] = globals()["__doc__"]

    current_date = datetime.now().strftime("%d%m%y")
    default_output_dir = Path(__file__).parent / f'outputs/registration/{current_date}'
    default_output_dir.mkdir(parents=True, exist_ok=True)
    # Create an out_hashes.txt file to store the hashes of the output files
    if not os.path.exists(default_output_dir / 'out_hashes.txt'):
        with open(default_output_dir / 'out_hashes.txt', 'w') as f:
            f.write('')

    parser = ArgumentParser(**kwargs)
    parser.add_argument(
        "-c", "--config", help="Configuration file", default=Path(__file__).parent / "params.yaml"
    )
    parser.add_argument(
        "-t", "--target", "--target-img", dest="target_img", help="Fixed target image",
        default=Path(__file__).parent / 'imgs/sub-CC110101/T1_brain.nii.gz'
    )
    parser.add_argument(
        "-s", "--source", "--source-img", dest="source_img", help="Moving source image",
        default=Path(__file__).parent / 'imgs/sub-CC110062/T2_brain.nii.gz'
    )
    parser.add_argument("--target-seg", help="Fixed target segmentation label image",
                        default=Path(__file__).parent / 'imgs/sub-CC110101/T1_brain_MALPEM_tissues.nii.gz', )
    parser.add_argument("--source-seg", help="Moving source segmentation label image",
                        default=Path(__file__).parent / 'imgs/sub-CC110062/T1_brain_MALPEM_tissues.nii.gz')
    parser.add_argument(
        "-o",
        "--output",
        "--output-transform",
        dest="output_transform",
        help="Output transformation parameters",
        default=default_output_dir / 'transform.mha',
    )
    parser.add_argument(
        "-w",
        "--warped",
        "--warped-img",
        "--output-img",
        dest="warped_img",
        help="Deformed source image",
        default=default_output_dir / 'warped.mha',
    )
    parser.add_argument(
        "--warped-seg",
        "--output-seg",
        dest="warped_seg",
        help="Deformed source segmentation label image",
        default=default_output_dir / 'warped_seg.mha',
    )
    parser.add_argument(
        "--device",
        help="Device on which to execute registration",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument("--debug-dir", help="Output directory for intermediate files",
                        default=Path(__file__).parent / 'outputs/debug')
    parser.add_argument(
        "--debug",
        "--debug-level",
        help="Debug level",
        type=int,
        default=0,
    )
    parser.add_argument("-v", "--verbose", help="Verbosity of output messages", type=int, default=5)
    parser.add_argument(
        "--log-level",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument("--show", help="Show images", action="store_true", default=True)
    parser.add_argument(
        "--dmmr_model_path",
        help="Path of the .pt file containing the DMMR model",
        # default="/home/joao/repos/dmmr/outputs/dmmr_models/camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs59_online_aug_tuned_tfms_single_axis_small_rot_bound.pt"
        default="/home/joao/repos/dmmr/outputs/dmmr_models/camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs54_online_aug_tuned_tfms_single_axis_small_rot.pt"
    )
    parser.add_argument("--outdir", help="Output directory for files",
                        default=default_output_dir)

    return parser


def init(args: Args) -> int:
    r"""Initialize registration."""
    configure_logging(log, args)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            log.error("Cannot use --device 'cuda' when torch.cuda.is_available() is False")
            return 1
        gpu_ids = cuda_visible_devices()
        if len(gpu_ids) != 1:
            log.error("CUDA_VISIBLE_DEVICES must be set to one GPU")
            return 1
    return 0


# noinspection PyNoneFunctionAssignment
def func(args: Args) -> int:
    r"""Execute registration given parsed arguments."""
    config = load_config(args.config)
    process_config_dmmr(config, args)
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    outdir = Path(args.outdir)
    with open(outdir / 'out_hashes.txt', 'a') as f:
        f.write(f'{config_hash}\n')
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    start = timer()
    transform = register_pairwise(
        target={"img": args.target_img,},  #  "seg": args.target_seg},
        source={"img": args.source_img,},  # "seg": args.source_seg},
        config=config,
        outdir=args.debug_dir,
        device=args.device,
        verbose=args.verbose,
        debug=args.debug,
        image_outdir=args.outdir,
        hash=config_hash,
    )
    log.info(f"Elapsed time: {timer() - start:.3f}s")
    if args.warped_img:
        target_grid = Grid.from_file(args.target_img)
        target_image = Image.read(args.target_img, device=device)
        source_image = Image.read(args.source_img, device=device)
        warp_image = TransformImage(
            target=target_grid,
            source=source_image.grid(),
            sampling="linear",
            padding=source_image.min(),
        ).to(device)
        data: Tensor = warp_image(transform.tensor(), source_image)
        warped_image = Image(data, target_grid)
        # Save file w/ unique name
        filename, file_extension = str(args.warped_img).split('.')

        # Create a unique output file name with the hash value
        output_filename = f"{filename}_{config_hash}.{file_extension}"
        output_yaml_filename = f"{filename}_{config_hash}.yaml"
        # warped_image.write(unlink_or_mkdir(output_filename))

        # Save the configuration dictionary to a yaml file
        with open(output_yaml_filename, 'w') as yaml_file:
            yaml.dump(config, yaml_file)

    if args.warped_seg:
        target_grid_seg = Grid.from_file(args.target_seg)
        source_image_seg = Image.read(args.source_seg, device=device)
        warp_labels = TransformImage(
            target=target_grid_seg,
            source=source_image_seg.grid(),
            sampling="nearest",
            padding=0,
        ).to(device)
        data: Tensor = warp_labels(transform.tensor(), source_image_seg)
        warped_image_seg = Image(data, target_grid_seg)
        filename, file_extension = str(args.warped_seg).split('.')
        output_filename = f"{filename}_{config_hash}.{file_extension}"
        # warped_image_seg.write(unlink_or_mkdir(output_filename))

    if args.output_transform:
        filename, file_extension = str(args.output_transform).split('.')
        output_filename = f"{filename}_{config_hash}.{file_extension}"
        path = unlink_or_mkdir(output_filename)
        if path.suffix == ".pt":
            transform.clear_buffers()
            # torch.save(transform, path)
        else:
            # transform.flow()[0].write(path)
            pass

    # measure metrics
    if args.warped_img and args.warped_seg:
        target_image_seg = Image.read(args.target_seg, device=device).int()
        # Initial Metric Data
        metric_data_raw = {
            'target': target_image,
            'source': source_image,
            'target_seg': target_image_seg.int(),
            'source_seg': source_image_seg.int(),
            'target_pred': source_image,
            'warped_source_seg': source_image_seg.int(),
        }

        # Define the metric groups to be calculated
        metric_groups = ['image_metrics', 'seg_metrics']

        # Measure before registration metrics
        before_metrics_results = prepare_and_measure_metrics(metric_data_raw, metric_groups)
        before_metrics_results = {key: float(value) for key, value in before_metrics_results.items()}
        print(60 * '-')
        print('Before registration metrics:')
        pprint(before_metrics_results)

        # Update Metric groups
        metric_groups.append('disp_metrics')

        # Measure after registration metrics
        metrics_results = prepare_and_measure_metrics(metric_data_raw, metric_groups,
                                                      transform, warped_image, warped_image_seg)
        metrics_results = {key: float(value) for key, value in metrics_results.items()}
        print(60 * '-')
        print('After registration metrics:')
        pprint(metrics_results)

        # save before metrics results and after registration metrics to a json file
        filename, file_extension = str(args.warped_img).split('.')
        output_filename = f"{filename}_{config_hash}_metrics.json"
        with open(output_filename, 'w') as f:
            json.dump({'before_registration': before_metrics_results,
                       'after_registration': metrics_results}, f)

        if args.show:
            diff_image_start = difference_image(target_image, source_image)
            diff_image_end = difference_image(target_image, warped_image)

            # Getting data to plot dice scores
            subclasses = [key for key in before_metrics_results.keys() if key.startswith('dice_class_')
                          or key.startswith('mean_dice')]

            # Values for before and after registration
            before_dice_values = [before_metrics_results[sub] for sub in subclasses]
            after_dice_values = [metrics_results[sub] for sub in subclasses]

            model_params = config['model']
            energy_params = config['energy']

            model_str = f"Model: {model_params.get('name', 'N/A')}, Stride: {model_params.get('stride', 'N/A')}\n"
            energy_str = f"Energy Seg: {energy_params.get('seg', 'N/A')}\n" \
                         f"BE: {energy_params.get('be', 'N/A')} " \
                         f"Reg: {energy_params.get('reg', 'N/A')} " \
                         f"msk: {energy_params.get('msk', 'N/A')}\n" \
                         f"curv: {energy_params.get('curv', 'N/A')} " \
                         f"diffusion: {energy_params.get('diffusion', 'N/A')} " \
                         f"tv: {energy_params.get('tv', 'N/A')}\n"
            optim_str = f"Optimizer: {config['optim']['name']}, lr: {config['optim']['step_size']}"

            filename, file_extension = str(args.warped_img).split('.')
            outfile = f"{filename}_vis_{config_hash}.png"
            plot_images_and_diffs(target_image, source_image, warped_image, diff_image_start, diff_image_end,
                                  before_metrics_results, metrics_results,
                                  dice_dict={'labels': subclasses,
                                             'before_values': before_dice_values,
                                             'after_values': after_dice_values},
                                  suptitle=f'{model_str} {energy_str}\n{optim_str}',
                                  outfile=outfile,
                                  args=args)
            outfile_deformation = f"{filename}_vis_deformation_{config_hash}.png"
            _ = invertible_registration_figure(target_image, source_image, transform, outfile_deformation)
            t1_source_image = Image.read(str(args.source_img).replace('T2', 'T1'), device=device)
            outfile_t1t1_diff = f"{filename}_t1t1_diff_{config_hash}.png"
            plot_t1t1_diff(target_image, t1_source_image, transform, outfile=outfile_t1t1_diff)
            outfile_seg_diff = f"{filename}_seg_diff_{config_hash}.png"
            plot_seg_diff(target_seg=target_image_seg.int(), source_seg=source_image_seg.int(),
                          warped_source_seg=warped_image_seg.int(), outfile=outfile_seg_diff)
            # plot_t2t2_diff()

    return 0


main = main_func(parser, func, init=init)


def load_config(path: PathStr) -> Dict[str, Any]:
    r"""Load registration parameters from configuration file."""
    config_path = Path(path).absolute()
    log.info(f"Load configuration from {config_path}")
    config_text = config_path.read_text()
    if config_path.suffix == ".json":
        return json.loads(config_text)
    return yaml.safe_load(config_text)


def process_config_dmmr(config, args):
    if 'energy' in config and 'seg' in config['energy']:
        seg_energy = config['energy']['seg']
        if isinstance(seg_energy, list) and seg_energy[1].upper() == 'DMMR':
            model_path_dict = {'model_path': args.dmmr_model_path}
            seg_energy.append(model_path_dict)
        else:
            pass
    else:
        print("Key 'energy' or 'msk' not found in the config dictionary.")


if __name__ == "__main__":
    sys.exit(main())
