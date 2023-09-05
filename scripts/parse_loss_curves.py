import glob
import yaml
import pandas as pd
import pprint
import json
import pickle as pkl
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def load_loss_values_and_metadata(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'rb') as file:
        loss_values = pkl.load(file)

    base_path = Path(file_path).parent
    hash = file_path.split('/')[-1].split('_')[0]
    yaml_path = base_path / f'warped_{hash}.yaml'
    metrics_path = base_path / f'warped_{hash}_metrics.json'

    if yaml_path.exists() and metrics_path.exists():
        with open(yaml_path, 'r') as file:
            yaml_dict = yaml.safe_load(file)
            metadata = {'loss_values': loss_values, 'steps': len(loss_values)}
            with open(metrics_path, 'r') as json_file:
                metrics_dict = json.load(json_file)
                metadata['before_reg_metrics'] = metrics_dict['before_registration']
                metadata['after_reg_metrics'] = metrics_dict['after_registration']

            if 'optim' in yaml_dict:
                optim_dict = yaml_dict['optim']
                metadata['lr'] = optim_dict.get('step_size', None)

            if 'energy' in yaml_dict:
                energy_dict = yaml_dict['energy']
                for term in ['be', 'seg', 'reg']:
                    if term in energy_dict:
                        term_list = energy_dict[term]
                        if isinstance(term_list, list) and len(term_list) >= 2:
                            metadata[term] = term_list[0].get('weight', None)
                            metadata[f'{term}_name'] = term_list[1].get('name', None)
                            if len(term_list) >= 3:
                                metadata[f'{term}_model_path'] = term_list[2].get('model_path', None)
                        else:
                            metadata[term] = 0
                            metadata[f'{term}_name'] = None
                            metadata[f'{term}_model_path'] = None

            metadata['hash'] = hash
            return metadata


def plot_losses(losses: List[Dict]):
    plt.figure(figsize=(12, 8))
    for entry in losses:
        plt.plot(entry['loss_values'], label=f'lr:{entry["lr"]}, be: {entry["be"]}, seg: {entry["seg"]},'
                                             f'reg: {entry["reg"]}'
                                             f' mean_dice: {entry["after_reg_metrics"]["mean_dice"]:.3f}')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend(loc='upper right', fontsize=8)

    if entry['seg_model_path'] is not None:
        model_path = entry['seg_model_path'].split("/")[-1].split(".")[0].split("_")[:10]
    else:
        model_path = ['']

    title = (f'Loss curve for {entry["seg"]} {" ".join(model_path)}'
             f' - Dice before {entry["before_reg_metrics"]["mean_dice"]: .3f}')

    plt.title(title)
    plt.show()


def main(pkl_files: List[str], groupby_second_level: str = None):
    losses = {}

    for file_path in pkl_files:
        metadata = load_loss_values_and_metadata(file_path)
        seg_name = metadata.get('seg_name', None)
        seg_model_path = metadata.get('seg_model_path', None)
        lr = metadata.get('lr', 0)
        be = metadata.get('be', 0)
        reg = metadata.get('reg', 0)

        if groupby_second_level is None:
            group_key = (seg_name)
        elif groupby_second_level == 'lr':
            group_key = (seg_name, lr)
        elif groupby_second_level == 'reg':
            group_key = (seg_name, reg)
        elif groupby_second_level == 'be':
            group_key = (seg_name, be)
        else:
            raise ValueError("Invalid groupby_second_level value")

        if seg_model_path is not None:
            group_key = (group_key, seg_model_path)

        if group_key not in losses:
            losses[group_key] = []

        if losses[group_key] == [] or metadata['hash'] not in [item['hash'] for item in losses[group_key]]:
            losses[group_key].append(metadata)

    for k, l in losses.items():
        plot_losses(l)


lr_be_base_folder = Path('/home/joao/repos/ffd/outputs/registration/050923/tmp_results')
lr_be_pkl_files = glob.glob(str(lr_be_base_folder / '*.pkl'))

# main(lr_be_pkl_files, groupby_second_level='lr')
# main(lr_be_pkl_files, groupby_second_level='be')
# main(lr_be_pkl_files, groupby_second_level='reg')

# TODO: add option to group by diff losses but same lr
