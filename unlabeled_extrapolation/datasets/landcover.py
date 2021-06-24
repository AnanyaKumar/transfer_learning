from pathlib import Path
import pickle
import numpy as np
import torch
from scipy.stats import rankdata
from scipy import signal
from torch.utils.data import Dataset
import re
from .range_dataset import RangeDataset
from unlabeled_extrapolation.utils.data_utils import get_split_idxs

DATA_ROOT = 'timeseries_by_box_v2'
DATA_CACHE = 'landcover_data.pkl'

CLASSNAMES = ['savannas', 'permanent_wetlands', 'woody_savannas',
              'deciduous_needleleaf_forests', 'permanent_snow_and_ice',
              'croplands', 'water_bodies', 'urban_and_built_up_lands',
              'open_shrublands','evergreen_broadleaf_forests',
              'closed_shrublands', 'barren', 'grasslands',
              'deciduous_broadleaf_forests', 'mixed_forests',
              'cropland_natural_vegetation_mosaics',
              'evergreen_needleleaf_forests']

TOP_LABELS = [0, 2, 5, 8, 9, 12]  # most prevelant classes in South Hemisphere
top_class_colors = ['r', 'g', 'b', 'm', 'c', 'k']

africa_top_left = [38.277609, -18.170184]
africa_bottom_right = [-36.337048, 55.037262]


def filter_nans(domain_to_data):
    '''
    Finds and removes any data points whose MODIS measurement has at least one
    NaN.
    Parameters
    ----------
    domain_to_data : dict from domain => (MODIS, lat/lon, label, domain) tuple
    '''
    num_domains = len(domain_to_data)
    for i in range(num_domains):
        data = domain_to_data[i][0]
        nan_mask = ~np.any(np.isnan(data), axis=(1, 2))  # Rows w/o NaNs.
        num_keys = len(domain_to_data[i])
        domain_data = [None] * num_keys  # Use list since tuple immutable.
        for j in range(num_keys):
            domain_data[j] = domain_to_data[i][j]

            # Only apply mask to NumPy arrays.
            if isinstance(domain_data[j], np.ndarray):
                domain_data[j] = domain_data[j][nan_mask]

        domain_to_data[i] = tuple(domain_data)


def split_map(domain_to_data, shuffle_seed=None):
    '''
    Quick helper method to split the domain_to_data map into its constituents.
    Parameters
    ----------
    domain_to_data : map from domain => (MODIS, lat/lons, labels, domains)
    shuffle_seed : int, default None
        If not None, the domains are shuffled before stacking together using
        this random seed.
    Returns
    -------
    map from str => NumPy array
        Each array is all of the data stacked together for all domains.
    '''
    indices = range(len(domain_to_data))
    if isinstance(shuffle_seed, int):
        rng = np.random.default_rng(shuffle_seed)
        indices = rng.permutation(len(domain_to_data))

    data = np.vstack([domain_to_data[i][0] for i in indices])
    lat_lon = np.vstack([domain_to_data[i][1] for i in indices])
    targets = np.hstack([domain_to_data[i][2] for i in indices])
    domains = np.hstack([[domain_to_data[i][3]
                         for _ in range(len(domain_to_data[i][2]))]
                         for i in indices])
    era5 = np.vstack([domain_to_data[i][4] for i in indices])
    return {'data': data, 'lat_lon': lat_lon, 'targets': targets,
            'domains': domains, 'era5': era5}


def load_from_file(root_path):
    '''
    Loads Landcover dataset starting from the folder given by root_path.
    Parameters
    ----------
    root_path : str
        Path to parent folder containing Landcover dataset.
    Returns
    -------
    domain_to_data : map from domain => (MODIS, lat/lon, labels, domains)
    '''
    domain_to_data = {}
    float_regex = re.compile(r'-?\d+\.-?\d+')  # Extracts lat/lon from filename
    classname_to_idx = {name: i for i, name in enumerate(CLASSNAMES)}
    root = Path(root_path)
    for domain_idx, box in enumerate(sorted(root.iterdir())):
        domain_data = []
        domain_lat_lon = []
        domain_targets = []
        for cls in box.iterdir():
            classname = cls.stem
            cls_idx = classname_to_idx[classname]
            for x in cls.iterdir():
                if x.suffix != '.npy':
                    continue
                lat, lon = float_regex.findall(x.name)
                domain_lat_lon.append([float(lat), float(lon)])
                domain_data.append(np.load(str(x)))
                domain_targets.append(cls_idx)

        domain_data = np.asarray(domain_data)
        domain_lat_lon = np.asarray(domain_lat_lon)
        domain_targets = np.asarray(domain_targets)
        domain_to_data[domain_idx] = (
            domain_data, domain_lat_lon, domain_targets, domain_idx)

    return domain_to_data


def load_data(root=DATA_ROOT, use_cache=True, save_cache=False,
              cache_path=DATA_CACHE, should_filter_nans=True, template_dataset=None):
    '''
    Reads the Landcover dataset from the filesystem and returns a map of the
    data. Can use a cached .pkl file for efficiency if desired.
    Parameters
    ----------
    root : str, default DATA_ROOT
        Path to the folder containing Landcover dataset.
    use_cache : bool, default True
        Whether to use a cached form of the dataset. The cache_path
        parameter must also be present.
    save_cache : bool, default False
        Whether to save the loaded data as a .pkl file for future use. The
        cache_path parameter must also be present.
    cache_path : str, default DATA_CACHE
        Path to .pkl file for loading/saving the Landcover dataset.
    Returns
    -------
    dict
        Map from domain => (MODIS, lat/lon, label, domain).
    '''
    domain_to_data = {}
    if use_cache:  # Default use cache.
        print(cache_path)
        with open(cache_path, 'rb') as pkl_file:
            domain_to_data = pickle.load(pkl_file)
    else:
        domain_to_data = load_from_file(root)

    if save_cache:  # Default don't save.
        with open(cache_path, 'wb') as pkl_file:
            pickle.dump(domain_to_data, pkl_file)

    return domain_to_data


def add_NDVI(domain_to_data):
    '''
    Computes the NDVI feature and appends it as a new axis to the existing
    MODIS readings. May give a RuntimeWarning for division by zero, but any NaN
    entries are replaced by zero before returning.
    Parameters
    ----------
    domain_to_data : map from domain => (MODIS, lat/lons, labels, domains)
    '''
    for i in domain_to_data:
        modis_data = domain_to_data[i][0]
        red, nir = modis_data[:, 0, :], modis_data[:, 1, :]
        ndvi = (nir - red) / np.maximum(nir + red, 1e-8)
        ndvi = ndvi[:, np.newaxis, :]
        # ndvi = np.expand_dims(np.nan_to_num(ndvi), axis=1)
        domain_data = list(domain_to_data[i])  # Since tuples are immutable.
        domain_data[0] = np.concatenate([modis_data, ndvi], axis=1)
        domain_to_data[i] = tuple(domain_data)


def resample_ERA5(domain_to_data):
    '''
    Upsamples the monthly data in ERA5 to 8-day frequency (12 -> 46) by Fourier method
    Assumes ERA5 features are at index 4
    Parameters
    ----------
    domain_to_data : map from domain => (MODIS, lat/lons, labels, domains)
    '''
    for i in domain_to_data:
        era5 = domain_to_data[i][4]
        era5_resampled = signal.resample(era5, 46, axis=2)

        domain_data = list(domain_to_data[i])  # Since tuples are immutable.
        domain_data[4] = era5_resampled
        domain_to_data[i] = tuple(domain_data)


def add_ERA5(domain_to_data):
    '''
    Adds features from ERA5 features, assumed to be at index 5 of the tuples
    in domain_to_data with shape (num_examples, 12, num_features).
    Parameters
    ----------
    domain_to_data : map from domain => (MODIS, lat/lons, labels, domains)
    '''
    for i in domain_to_data:
        modis_data = domain_to_data[i][0]
        era5 = domain_to_data[i][4]

        domain_data = list(domain_to_data[i])  # Since tuples are immutable.
        domain_data[0] = np.concatenate([modis_data, era5], axis=1)
        domain_to_data[i] = tuple(domain_data)


def filter_top_targets(data_map):
    '''
    Filters out data points whose label does not belong to TOP_LABELS.
    Parameters
    ----------
    data_map : dict
        Stores MODIS measurements, lat/lons, labels, and domain indices.
    '''
    target_mask = np.in1d(data_map['targets'], TOP_LABELS)
    for key in data_map.keys():
        data_map[key] = data_map[key][target_mask]

    # Normalize targets into range [0, num_classes - 1].
    data_map['targets'] = rankdata(data_map['targets'], method='dense') - 1


def normalize_lat_lon(lat_lon):
    '''
    Normalizes lat/lon coordinates to a point on the unit sphere. Can be moved
    to a more general file if needed.
    Parameters
    ----------
    lat_lon : numpy.ndarray
        NumPy array of size 2 storing the latitude and longitude.
    Returns
    -------
    numpy.ndarray
        NumPy array of size 3 storing the (x, y, z) coordinates.
    '''
    lat, lon = lat_lon
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z])


# Note: we modified some of the default arguments from the in-n-out code base.
# And e.g. added root and use_cache.
class Landcover(Dataset):
    def __init__(self, root, cache_path=None, use_cache=True, eval_mode=False, num_train_domains=86,
                 split='train', should_filter_nans=True, transform=None,
                 target_transform=None, shuffle=False, shuffle_domains=True,
                 seed=None, include_NDVI=True, include_lat_lon=False,
                 include_ERA5=False, standardize=True, multitask=False,
                 target_lat_lon=False, unlabeled_prop=0,
                 pretrain=False, masked_pretrain=False,
                 use_unlabeled_id=False, use_unlabeled_ood=False,
                 unlabeled_targets_path=None,
                 standardize_unlabeled_sample_size=False,
                 **kwargs):
        '''
        Constructor for a PyTorch Dataset class around the Landcover dataset.
        Parameters
        ----------
        eval_mode : bool, default False
            Whether this a dataset for evaluation.
        num_train_domains : int, default 86
            Number of train domains up to 86.
        split : str, default 'train'
            Describes how to split dataset (e.g., 'train', 'val', 'test'). If
            this string starts with 'north' or 'south', then splitting by
            hemisphere occurs.
        should_filter_nans : bool, default True
            Whether to filter NaN entries from the data map before returning
            (see filter_nans() method).
        transform:
            input transformation
        target_transform:
            target transformation
        shuffle : bool, default False
            Whether to shuffle the entire dataset. A valid seed must be passed.
        shuffle_domains : bool, default False
            Whether to shuffle the order of domains before splitting. A valid
            seed must be passed.
        seed : int, default None
            Random seed to use for consistent shuffling.
        include_NDVI : bool, default False
            Whether to calculate NDVI as an additional feature.
        include_lat_lon : bool, default False
            Whether to include the lat/lon for each measurement as a feature.
        standardize : bool, default False
            Whether to subtract mean/divide by STD when returning samples.
        multitask : bool, default False
            Whether to include the lat/lon for each measurement as a target.
        target_lat_lon : bool, default False
            Whether to use lat/lon as the target rather than the class label.
        unlabeled_prop : float, default 0
            How much data from the entire dataset to keep as unlabeled data.
        use_unlabeled : bool, default False
            Whether to use the unlabeled data in training.
        pretrain : bool, default False
            whether to pretrain
        masked_pretrain : bool, default False
            whether to do masked pretraining
        **kwargs:
            Passed through to load_data() function.
        '''
        self.split = split
        self.eval_mode = eval_mode
        self.transform = transform
        self.target_transform = target_transform
        self.include_lat_lon = include_lat_lon
        self.standardize = standardize
        self.multitask = multitask
        self.pretrain = pretrain
        self.masked_pretrain = masked_pretrain
        self.target_lat_lon = target_lat_lon
        self.use_unlabeled_all = (use_unlabeled_id and use_unlabeled_ood)
        self.use_unlabeled = (use_unlabeled_id or use_unlabeled_ood)
        self.standardize_unlabeled_sample_size = standardize_unlabeled_sample_size

        if multitask and target_lat_lon:
            msg = 'Only one of "multitask" and "target_lat_lon" can be True!'
            raise ValueError(msg)

        data_map = load_data(root=root, use_cache=use_cache, cache_path=cache_path, **kwargs)
        self.num_domains = len(data_map)
        if should_filter_nans:  # Default filter.
            filter_nans(data_map)
        if include_NDVI:
            add_NDVI(data_map)

        prepare_era5 = (include_ERA5 or multitask or pretrain)
        if prepare_era5:
            resample_ERA5(data_map)

        if include_ERA5:
            add_ERA5(data_map)

        # Try splitting by hemispheres first.
        split_by_hemi = False
        domain_seed = seed if shuffle_domains else None
        data_map = split_map(data_map, domain_seed)
        filter_top_targets(data_map)

        # get ood_idxs
        if split.startswith('north') or split.startswith('south'):
            lat_mask = (data_map['lat_lon'][:, 0] < 0)
            ood_idxs = np.where(lat_mask)[0]
            split_by_hemi = True
        elif split.startswith('nonafrica') or split.startswith('africa'):
            inside_mask = np.asarray([(africa_top_left[0] >= lat) and (africa_bottom_right[0] <= lat)
                           and (africa_top_left[1] <= lon) and (africa_bottom_right[1] >= lon)
                           for lat,lon in data_map['lat_lon']])
            ood_idxs = np.where(inside_mask)[0]
        else:
            raise ValueError(f"Split {split} not supported")

        # split procedure:
        # split 10% of data for evaluation
        all_split_idxs = get_split_idxs(
                unlabeled_proportion=unlabeled_prop, seed=seed,
                ood_idxs=ood_idxs, total_len=len(data_map['data']))

        data = data_map['data']
        lat_lon = data_map['lat_lon']
        targets = data_map['targets']
        domains = data_map['domains']
        era5 = data_map['era5']

        ood_splits = {'africa', 'south', 'africa-train', 'africa-val'}
        if split in ood_splits:
            num_total = len(all_split_idxs['test2'])
            mid = num_total // 2
            if split == 'africa-train':
                idxs = all_split_idxs['test2'][:mid]
            elif split == 'africa-val':
                idxs = all_split_idxs['test2'][mid:]
            else:
                idxs = all_split_idxs['test2']
        elif split.endswith('train'):  # 80/20 train/val split.
            idxs = all_split_idxs['train']
        elif split.endswith('val'):
            idxs = all_split_idxs['val']
        elif split.endswith('test'):
            idxs = all_split_idxs['test']
        
        # used when standardize_unlabeled_sample_size = True
        min_sample_size = min(len(all_split_idxs['unlabeled_id']), len(all_split_idxs['unlabeled_ood']))

        if self.use_unlabeled_all:
            unlabeled_idxs = all_split_idxs['unlabeled']
            if self.standardize_unlabeled_sample_size:
                # change to half - half id and ood
                unl_id = all_split_idxs['unlabeled_id'][:min_sample_size // 2]
                unl_ood = all_split_idxs['unlabeled_ood'][:min_sample_size // 2]
                unlabeled_idxs = np.concatenate([unl_id, unl_ood])
        elif use_unlabeled_id:
            unlabeled_idxs = all_split_idxs['unlabeled_id']
            if self.standardize_unlabeled_sample_size:
                # change to min_sample_size
                unlabeled_idxs = unlabeled_idxs[:min_sample_size]
        elif use_unlabeled_ood:
            unlabeled_idxs = all_split_idxs['unlabeled_ood']
            if self.standardize_unlabeled_sample_size:
                # change to min_sample_size
                unlabeled_idxs = unlabeled_idxs[:min_sample_size]
        else:
            unlabeled_idxs = []

        self.data = data[idxs]
        self.targets = targets[idxs]
        self.lat_lon = lat_lon[idxs]
        self.domain_labels = domains[idxs]
        self.unlabeled_data = data[unlabeled_idxs]
        self._unseen_unlabeled_targets = targets[unlabeled_idxs]
        self.unlabeled_lat_lon = lat_lon[unlabeled_idxs]
        self.unlabeled_domain_labels = domains[unlabeled_idxs]

        # assumes that unlabeled_targets_path uses the same split
        self.unlabeled_targets_path = unlabeled_targets_path
        if unlabeled_targets_path is not None:
            self.unlabeled_targets = np.load(unlabeled_targets_path)

            if len(self.unlabeled_targets) != len(self.unlabeled_data):
                raise ValueError("Length of pseudolabels does not match the unlabeled data length")

        if prepare_era5:
            self.era5 = era5[idxs]
            self.unlabeled_era5 = era5[unlabeled_idxs]

        if self.standardize:
            # take mean over time and examples
            # self.mean = self.data.mean(axis=(0, 2))[:, np.newaxis]
            # self.std = self.data.std(axis=(0, 2))[:, np.newaxis]
            # if prepare_era5:
            #     self.era5_mean = self.era5.mean(axis=(0, 2))[:, np.newaxis]
            #     self.era5_std = self.era5.std(axis=(0, 2))[:, np.newaxis]

            # all models use unlabeled data to normalize

            all_unlabeled_idxs = all_split_idxs['unlabeled']
            all_unlabeled_data = data[all_unlabeled_idxs]
            concatted = np.concatenate([self.data, all_unlabeled_data], axis=0)
            self.mean = concatted.mean(axis=(0, 2))[:, np.newaxis]
            self.std = concatted.std(axis=(0, 2))[:, np.newaxis]
            if prepare_era5:
                concatted_era5 = np.concatenate([self.era5, self.unlabeled_era5], axis=0)
                self.era5_mean = concatted_era5.mean(axis=(0, 2))[:, np.newaxis]
                self.era5_std = concatted_era5.std(axis=(0, 2))[:, np.newaxis]

            # if self.use_unlabeled:
            #     # if there is unlabeled data, estimate mean and std using more data
            #     concatted = np.concatenate([self.data, self.unlabeled_data], axis=0)
            #     self.mean = concatted.mean(axis=(0, 2))[:, np.newaxis]
            #     self.std = concatted.std(axis=(0, 2))[:, np.newaxis]
            #     if prepare_era5:
            #         concatted_era5 = np.concatenate([self.era5, self.unlabeled_era5], axis=0)
            #         self.era5_mean = concatted_era5.mean(axis=(0, 2))[:, np.newaxis]
            #         self.era5_std = concatted_era5.std(axis=(0, 2))[:, np.newaxis]

    def get_unlabeled_dataset(self):
        unlabeled_start_idx = len(self.data)
        unlabeled_end_idx = len(self)
        return RangeDataset(self, unlabeled_start_idx, unlabeled_end_idx)
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        labeled = True
        metadata = {}
        if index < len(self.data):
            x, target = self.data[index], self.targets[index]
            lat_lon = self.lat_lon[index]
            if self.standardize:
                x = (x - self.mean) / self.std
            metadata['domain_label'] = self.domain_labels[index]
        elif self.use_unlabeled:
            labeled = False
            # index -= len(self.data)
            if self.multitask:
                # pick a random unlabeled point
                index = np.random.choice(len(self.unlabeled_data))
            else:
                index = index - len(self.data)

            if self.unlabeled_targets_path is not None and not self.eval_mode:
                x, target = self.unlabeled_data[index], self.unlabeled_targets[index]
            else:
                x, target = self.unlabeled_data[index], -100
            if self.standardize:
                x = (x - self.mean) / self.std
            lat_lon = self.unlabeled_lat_lon[index]
            metadata['domain_label'] = self.unlabeled_domain_labels[index]
        else:
            raise IndexError('Dataset index out of range.')

        metadata['lat_lon'] = lat_lon
        metadata['labeled'] = 1 if labeled else 0

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.include_lat_lon:
            # TODO is this correct below?
            # lat_lon = self.lat_lon[index]
            if self.standardize:
                lat_lon = normalize_lat_lon(lat_lon)
            # TODO does this stuff assume that x has been flattened by self.transform?
            if isinstance(x, np.ndarray):
                x = np.hstack([x, lat_lon])
            elif isinstance(x, torch.Tensor):
                lat_lon = torch.from_numpy(lat_lon).to(x.dtype)
                x = torch.cat([x, lat_lon])

        if self.multitask or self.pretrain or self.masked_pretrain:
            if labeled:
                era5 = self.era5[index]
            else:
                era5 = self.unlabeled_era5[index]
            if self.standardize:
                era5 = (era5 - self.era5_mean) / self.era5_std
            # we just use the non-time series part as output
            era5_mean = np.mean(era5, axis=1)
            era5_std = np.std(era5, axis=1)
            era5_as_target = np.concatenate([era5_mean, era5_std], axis=0)
            era5_as_target = torch.from_numpy(era5_as_target).float()
            if self.multitask:
                target = [target, era5_as_target]
            elif self.masked_pretrain:
                x_as_target = torch.cat([x.mean(1), x.std(1)])
                target = [-100, era5_as_target, x_as_target]
            else:
                target = era5_as_target

        if self.masked_pretrain:
            use_idx = np.random.choice([1,2])
            metadata['use_idx'] = use_idx
            # mask the input accordingly
            if use_idx == 1:
                x[8:, :] = 0
            elif use_idx == 2:
                # mask the MODIS measurements
                x[:8, :] = 0

        # return {'data': x, 'target': target, 'domain_label': metadata}
        x = torch.from_numpy(x).float()
        return x, target

    def __len__(self):
        length = len(self.data)
        if self.use_unlabeled and not self.eval_mode:
            if self.multitask:
                # length += len(self.unlabeled_data)
                # try to balance labeled and unlabeled
                length *= 2
            length = len(self.data) + len(self.unlabeled_data)
        return length

    def get_mean(self):
        '''
        Returns the mean for this dataset. Useful for getting the mean of a
        training set in order to standardize val and test sets.
        Returns
        -------
        self.mean, which is a float or numpy.ndarray.
        '''
        return self.mean

    def set_mean(self, mean):
        '''
        Sets the mean to use for standardization. Useful for setting the mean
        of a val or test set from the mean of a training set.
        Parameters
        ----------
        mean : Union[float, numpy.ndarray]
            Mean to subtract from data.
        '''
        self.mean = mean

    def get_std(self):
        '''
        Returns the std for this dataset. Useful for getting the std of a
        training set in order to standardize val and test sets.
        Returns
        -------
        self.std, which is a float or numpy.ndarray.
        '''
        return self.std

    def set_std(self, std):
        '''
        Sets the std to use for standardization. Useful for setting the std of
        a val or test set from the std of a training set.
        Parameters
        ----------
        std : Union[float, numpy.ndarray]
            Std to divide by for standardization.
        '''
        self.std = std

