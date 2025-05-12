import torch
from torch.utils.data import Dataset

class AzulDataset(Dataset):
    """
    PyTorch Dataset for Azul Zero self-play examples.
    Each example is a dict with:
      - 'obs': flat observation vector (numpy array)
      - 'pi':  policy target distribution or index
      - 'v':   value target (float)
    """
    def __init__(self, examples):
        """
        Args:
            examples: list of dicts, each with keys 'obs', 'pi', 'v'
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        obs = torch.tensor(ex['obs'], dtype=torch.float32)
        pi = torch.tensor(ex['pi'], dtype=torch.float32)
        v = torch.tensor(ex['v'], dtype=torch.float32)

        spatial_size = 6 * 5 * 5  # 6 canales espaciales, 5x5 dimensiones
        spatial = obs[:spatial_size].view(6, 5, 5)
        global_ = obs[spatial_size:]

        return {'spatial': spatial, 'global': global_, 'pi': pi, 'v': v}
