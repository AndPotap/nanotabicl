# NanoTabICL: a minimal TabICLv2 implementation

| [Full TabICLv2 code](https://github.com/soda-inria/tabicl) | [TabICLv2 Paper](https://arxiv.org/abs/2602.11139) |
|----------------------------------------------------|------------------------|

This repository provides a short (~170 LOC) implementation of the [TabICLv2](https://arxiv.org/abs/2602.11139) architecture.
For using our pre-trained TabICLv2 model, 
please visit the [main repository](https://github.com/soda-inria/tabicl).

Compared to [nanoTabPFN](https://github.com/automl/nanoTabPFN),

- the model is TabICLv2, not TabPFN,
- we implement the full model (including RoPE, QASSMax, etc.) with speed optimizations
  (but without the inference wrappers + memory optimizations from the main repository),
- we implement regression as well,
- we currently do not provide a sklearn interface,
- we currently do not provide pre-training code.
  For now, we refer to [nanoTabPFN](https://github.com/automl/nanoTabPFN) and
  the [nanoTabPFN speedrun](https://github.com/borawhocodess/modded-nanotabpfn) for pre-training code.

## Usage

```python
from model import NanoTabICLv2
import torch

model = NanoTabICLv2(max_classes=10, out_dim=10)  # original model size
X_train_and_test = torch.randn(batch_size, n_train+n_test, n_cols)
y_train = torch.randint(10, size=(batch_size, n_train)).float()
y_test_pred_logits = model(X_train_and_test, y_train)

# if you want a smaller model + regression with 999 quantiles instead
# warning: for regression, you need to standardize y yourself (and backtransform the output)
model = NanoTabICLv2(max_classes=0, out_dim=999, embed_dim=96,
                 col_num_blocks=2, row_num_blocks=2, icl_num_blocks=4,
                 col_nhead=4, row_nhead=4, icl_nhead=4)
y_train = torch.randn(batch_size, n_train)
y_test_pred_quantiles = model(X_train_and_test, y_train)
```

Note that `X_train_and_test` is standardized inside the model (based on train only). 
We do not include other preprocessing options from TabICLv2 
since they are normally not part of the architecture.
