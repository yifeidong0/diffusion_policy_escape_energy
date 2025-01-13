import joblib
import torch
from typing import Dict
import numpy as np


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def preprocess_dataset(dataset):
    dataset["obs_encode"] = []
    dataset["episode_ends"] = list(range(1, dataset["paths"].shape[0]+1))

    for ct, rad in zip(dataset["ellipse_centers"], dataset["ellipse_radii"]):
        dataset["obs_encode"].append(np.hstack([ct, rad]).flatten()) # (12,), 4(ct,rad)+4+4
    return dataset


# dataset
class EscapeEnergy2DDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        config: Dict,
    ):
        self.set_config(config)
        # read from zarr dataset
        dataset_root = joblib.load(dataset_path)
        print("!!!!!!!!!!!", dataset_root.keys())
        print("!!!!!!!!!!!paths", dataset_root["paths"].shape)
        print("!!!!!!!!!!!object_starts", dataset_root["object_starts"].shape)

        # proprocessing
        dataset_root = preprocess_dataset(dataset_root)
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            "action": dataset_root["paths"],
            # (N, obs_dim)
            "obs": dataset_root["object_starts"],
            # (N, 4x3)
            "obstacle": np.array(dataset_root["obs_encode"]),
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root["episode_ends"]
        print("!!!!!!s!!!!!action", train_data["action"].shape)
        print("!!!!!!s!!!!!obs", train_data["obs"].shape)
        print("!!!!!!s!!!!!obstacle", train_data["obstacle"].shape)

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=self.pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=self.obs_horizon - 1,
            pad_after=self.action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        # normalized_train_data = dict()
        for key, data in train_data.items():
            if key == "obstacle":
                continue
            stats[key] = get_data_stats(data)
            # normalized_train_data[key] = normalize_data(data, stats[key])
        # normalized_train_data["obstacle"] = train_data["obstacle"]

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = train_data

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        # buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        # nsample = sample_sequence(
        #     train_data=self.normalized_train_data,
        #     sequence_length=self.pred_horizon,
        #     buffer_start_idx=buffer_start_idx,
        #     buffer_end_idx=buffer_end_idx,
        #     sample_start_idx=sample_start_idx,
        #     sample_end_idx=sample_end_idx,
        # )
        # nsample = self.normalized_train_data

        nsample = dict()
        nsample["action"] = self.normalized_train_data["action"][idx,:,:].reshape(1,-1)
        nsample["obs"] = np.concatenate((self.normalized_train_data["obs"][idx,:], self.normalized_train_data["obstacle"][idx,:]))
        nsample["obstacle"] = self.normalized_train_data["obstacle"][idx,:].reshape(1,-1)
        print("__getitem__()")
        print("idx",idx)
        print('action',nsample["action"].shape)
        print('obs',nsample["obs"].shape)
        print('obstacle',nsample["obstacle"].shape)

        return nsample

    def set_config(self, config: Dict):
        self.config = config
        self.obs_horizon = config["obs_horizon"]
        self.action_horizon = config["action_horizon"]
        self.pred_horizon = config["pred_horizon"]
