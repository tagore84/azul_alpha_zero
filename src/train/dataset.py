import torch
from torch.utils.data import Dataset
import numpy as np

def augment_factory_permutation(obs_flat: np.ndarray, pi: np.ndarray, num_factories: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies a random permutation to the factories in obs and updates the policy accordingly.
    """
    FACTORY_TILE_COUNT = 5
    COLOR_COUNT = 5
    DEST_COUNT = 6

    obs = obs_flat.copy()
    # Extraemos la parte de las fábricas
    factory_start = 10  # Suponiendo: 5 (bag) + 5 (discard)
    factory_len = num_factories * FACTORY_TILE_COUNT
    factory_end = factory_start + factory_len

    factories = obs[factory_start:factory_end].reshape(num_factories, FACTORY_TILE_COUNT)
    perm = np.random.permutation(num_factories)
    permuted_factories = factories[perm]
    obs[factory_start:factory_end] = permuted_factories.flatten()

    # Reindexamos la política
    pi_new = pi.copy()
    for old_f in range(num_factories):
        for color in range(COLOR_COUNT):
            for dest in range(DEST_COUNT):
                old_idx = old_f * (COLOR_COUNT * DEST_COUNT) + color * DEST_COUNT + dest
                new_f = perm[old_f]
                new_idx = new_f * (COLOR_COUNT * DEST_COUNT) + color * DEST_COUNT + dest
                pi_new[new_idx] = pi[old_idx]

    return obs, pi_new

class AzulDataset(Dataset):
    """
    PyTorch Dataset for Azul Zero self-play examples.
    Each example is a dict with:
      - 'obs': flat observation vector (numpy array)
      - 'pi':  policy target distribution or index
      - 'v':   value target (float)
    """
    def __init__(self, examples, augment_factories: bool = True):
        self.examples = examples
        self.augment_factories = augment_factories

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        obs = ex['obs']
        pi = ex['pi']
        v = ex['v']

        if self.augment_factories:
            obs, pi = augment_factory_permutation(obs, pi)

        obs = torch.tensor(obs, dtype=torch.float32)
        pi = torch.tensor(pi, dtype=torch.float32)
        v = torch.tensor(v, dtype=torch.float32)

        spatial_size = 6 * 5 * 5
        spatial = obs[:spatial_size].view(6, 5, 5)
        global_ = obs[spatial_size:]

        return {'spatial': spatial, 'global': global_, 'pi': pi, 'v': v}
    def __getitem__(self, idx):
        ex = self.examples[idx]
        obs = torch.tensor(ex['obs'], dtype=torch.float32)
        pi = torch.tensor(ex['pi'], dtype=torch.float32)
        v = torch.tensor(ex['v'], dtype=torch.float32)

        spatial_size = 4 * 5 * 5  # 4 channels (2 players * (pattern + wall)), 5x5 dimensions
        spatial = obs[:spatial_size].view(4, 5, 5)
        global_ = obs[spatial_size:]

        return {'spatial': spatial, 'global': global_, 'pi': pi, 'v': v}
