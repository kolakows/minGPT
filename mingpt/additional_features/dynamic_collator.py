from torch.nn.utils.rnn import pad_sequence


class DynamicCollator:

    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, examples):
        # truncate to block_size
        inputs = [item[0][:self.block_size] for item in examples]
        targets = [item[1][:self.block_size] for item in examples]

        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = pad_sequence(targets, batch_first=True, padding_value=-100)

        return inputs, targets
