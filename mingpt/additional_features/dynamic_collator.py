from torch.nn.utils.rnn import pad_sequence


class DynamicCollator:

    def __call__(self, examples):
        inputs = [item[0] for item in examples]
        targets = [item[1] for item in examples]
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = pad_sequence(targets, batch_first=True, padding_value=-100)
        return inputs, targets
