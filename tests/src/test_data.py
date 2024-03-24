from src import data
from torch.utils.data import (
    SequentialSampler,
    BatchSampler
)

def test_random_sampler():
    train_data = [i for i in range(1024)]

    batch_sampler = BatchSampler(SequentialSampler(train_data), batch_size = 32, drop_last = False)
    random_batch_sampler = data.RandomBatchSampler(SequentialSampler(train_data), batch_size = 32, drop_last = False)

    batches = list(batch_sampler)
    random_batches = list(random_batch_sampler)

    assert random_batches != batches
    assert random_batches[0] == sorted(random_batches[0])