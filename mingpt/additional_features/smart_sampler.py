import torch
from torch.utils.data.sampler import Sampler


def get_length_grouped_indices(dataset, dataset_length, batch_size, mega_batch_mult=None):
    """
    Adapted from huggingface transformers group_by_length
    - calculate length of dataset once
    - avoid explicit loading of lengths to memory

    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(dataset_length // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(dataset_length)
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, dataset_length, megabatch_size)]
    megabatches = [list(sorted(megabatch, key=lambda i: dataset[i]['len'], reverse=True)) for megabatch in megabatches]

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length, the longest element is the first
    megabatch_maximums = [dataset[megabatch[0]]['len'] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
            self,
            batch_size: int,
            dataset
    ):
        self.batch_size = batch_size

        self.dataset = dataset
        self.length = len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        indices = get_length_grouped_indices(self.dataset, self.length, self.batch_size)
        return iter(indices)
