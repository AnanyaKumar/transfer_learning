
import logging

class LayerWiseTuner:
    # Only works for CLIP ViT at the moment.

    def __init__(self, model, num_steps, num_warmup_steps, num_cooldown_steps, freeze_embed):
        self._model = model
        self._num_steps = num_steps
        self._num_warmup_steps = num_warmup_steps
        self._num_cooldown_steps = num_cooldown_steps
        self._freeze_embed = freeze_embed
        self._step = 0
        self._num_trans_freeze = -1

    def step(self):
        num_trans_layers = self._model.get_num_trans_layers()
        # Get number of transformer layers to freeze.
        effective_steps = max(0, self._step - self._num_warmup_steps)
        effective_num_steps = max(1, self._num_steps - self._num_warmup_steps - self._num_cooldown_steps)
        if self._step >= self._num_steps - self._num_cooldown_steps:
            num_trans_tune = num_trans_layers
        elif effective_num_steps == 1:
            num_trans_tune = num_trans_layers
        else:
            num_trans_tune = int(float(effective_steps) / (effective_num_steps-1) * num_trans_layers)
        assert num_trans_tune <= num_trans_layers
        num_trans_freeze = num_trans_layers - num_trans_tune
        if num_trans_freeze != self._num_trans_freeze:
            self._model.freeze_bottom_trans(num_trans_freeze, self._freeze_embed)
            # Print statistics about number of parameters tuning, layers frozen, etc.
            logging.info(f'Freezing {num_trans_freeze} transformer blocks out of {num_trans_layers}')
        self._num_trans_freeze = num_trans_freeze
        self._step += 1
        if self._step > self._num_steps:
            logging.warning(f'self._step ({self._step}) > num_steps ({num_steps})')
