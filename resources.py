import os
import platform

import numpy as np
import sklearn
import torch

print(f"{platform.processor()=}")
print(f"{np.__version__=}")
print(f"{sklearn.__version__=}")
print(f"{torch.__version__=}")
print(f"{torch.get_num_threads()=}")
print(f"{os.environ.get("OMP_NUM_THREADS")=}")
print(f"{os.environ.get("MKL_NUM_THREADS")=}")

print(np.show_config())
print(torch.__config__.show())
