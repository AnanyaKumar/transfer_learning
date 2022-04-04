import numpy as np


def create_data_splits(num_examples, val_proportion, test_proportion):
    '''
    num_examples: int
    '''
    if val_proportion + test_proportion >= 1.0:
        raise ValueError("val_proportion and test_proportion add to more than 1")

    num_val = int(num_examples * val_proportion)
    num_test = int(num_examples * test_proportion)
    idxs = np.arange(num_examples)
    np.random.shuffle(idxs)
    val_idxs = idxs[:num_val]
    test_idxs = idxs[num_val:num_val+num_test]
    train_idxs = idxs[num_val+num_test:]
    return train_idxs, val_idxs, test_idxs


def split_ood_idxs(idxs, ood_idxs):
    '''
    Split idxs into two index arrays, in-domain and ood according to the indices in ood_idxs.
    Args:
        idxs: List[int] or ndarray[int]
            list of indices that we want to split
        ood_idxs: List[int] or ndarray[int]
            list of indices of ood data, where indices are wrt to the full dataset
    '''
    is_ood = (ood_idxs[np.searchsorted(ood_idxs, idxs)] == idxs)
    ood_split_idxs = idxs[is_ood]
    id_split_idxs = idxs[~is_ood]
    return id_split_idxs, ood_split_idxs


def get_split_idxs(unlabeled_proportion, ood_idxs, total_len, seed=None):
    '''
    Args:
        unlabeled_proportion: float
            float between [0, 1]. Proportion of training data to use as unlabeled
        ood_idxs: List[int] or ndarray[int]
            list of indices of ood data, where indices are wrt to the full dataset
            (range(len(data)))
        total_len: int
            the total length of dataset (len(data))
        seed: Optional[int]
            seed for splitting of the training data (not the eval data)
    '''
    ood_idxs = np.sort(ood_idxs)
    # augment with one more value larger or equal to total_len for
    # searchsorted
    ood_idxs = np.concatenate([ood_idxs, [total_len]])

    # hardcoded seed for evaluation split
    eval_split_seed = 15485863
    rng_state = np.random.get_state()
    np.random.seed(eval_split_seed)
    eval_len = int(total_len * 0.1)
    all_idxs = np.arange(total_len)
    np.random.shuffle(all_idxs)
    eval_idxs = all_idxs[:eval_len]
    noneval_idxs = all_idxs[eval_len:]

    # split eval_idxs into val/test/OOD
    id_split_idxs, ood_split_idxs = split_ood_idxs(eval_idxs, ood_idxs)
    non_ood_len = len(id_split_idxs)
    val_split_idxs = id_split_idxs[:non_ood_len//2]
    test_split_idxs = id_split_idxs[non_ood_len//2:]
    np.random.set_state(rng_state)

    rng_state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    unlabeled_len = int(len(noneval_idxs) * unlabeled_proportion)
    unlabeled_split_idxs = noneval_idxs[:unlabeled_len]
    train_split_idxs = noneval_idxs[unlabeled_len:]
    # throw away the OOD from the train split
    train_split_idxs, _ = split_ood_idxs(train_split_idxs, ood_idxs)
    np.random.set_state(rng_state)

    # split the unlabeled into ID and OOD for convenience
    unlabeled_idxs_id, unlabeled_idxs_ood = split_ood_idxs(unlabeled_split_idxs, ood_idxs)

    return {'train': train_split_idxs,
            'unlabeled': unlabeled_split_idxs,
            'unlabeled_id': unlabeled_idxs_id,
            'unlabeled_ood': unlabeled_idxs_ood,
            'val': val_split_idxs,
            'test': test_split_idxs,
            'test2': ood_split_idxs}

