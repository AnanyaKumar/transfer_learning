import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal

from collections import Counter

class MixtureOfManifolds:
    '''
    Currently only does mixture of Gaussians
    r: number of manifolds in the distribution (r <= d)
    '''
    def __init__(self, means, sigma, d, size):
        super().__init__()
        self._r = len(means)
        self._means = means
        self._d = d
        self._sigma = sigma
        self._size = size
        self._use_population = (size == float('inf'))
        if not all([len(m) == d for m in means]):
            raise ValueError(f'All means must be of dimensionality d, which was {d}.')
        if self._r > self._d:
            raise ValueError(f'r must be <= d.')
        self.samplers = []
        for mean in means:
            sampler = MultivariateNormal(
                loc=mean, covariance_matrix=(1 / self._d) * torch.eye(self._d))
            self.samplers.append(sampler)
        self.samples = []
        if self._use_population:
            sampler_idxs = list(torch.randint(self._r, size=self._size))
            counts = Counter(sampler_idxs)
            for idx, sampler in enumerate(self.samplers):
                data = sampler.sample_n(counts[idx])
                for d in data:
                    self.samples.append((data, idx))

    def add_gaussian(self, x):
        '''x is shape (batch size, d)'''
        noise = torch.randn_like(x)
        return x + (self._sigma / torch.sqrt(self._d)) * noise

    def __len__(self):
        return min(self._size, 1e9)
    
    def __getitem__(self, idx):
        if self._use_population:
            sampler_idx = torch.randint(self._r)
            data = self.samplers[sampler_idx].sample_n(1)
            return data, sampler_idx
        else:
            return self.samples[idx]

class ContrastiveDs(Dataset):
    def __init__(self, ds, transform):
        super().__init__()
        if not isinstance(transform, TwoCropsTransform):
            raise ValueError('Must provide a TwoCropsTransform instance for transform.')
        self._ds = ds
        self._transform = transform

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        x, y = self._ds[i]
        x = self._transform(x)
        return x, y

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def get_transform():
    trans = transforms.Compose([
        
    ])
    return TwoCropsTransform(trans)
