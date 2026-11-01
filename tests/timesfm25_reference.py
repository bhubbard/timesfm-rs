"""Oracle for the ignored Rust checkpoint test.

Install Google's timesfm[torch] at e51928e27119cb17bebc005be2696b75e0a9e688,
set TIMESFM_CHECKPOINT to google/timesfm-2.5-200m-pytorch's local directory,
and optionally set TIMESFM_PYTHON to that environment's Python executable.
Run: cargo test test_timesfm25_checkpoint_parity -- --ignored --nocapture
The test compares every point and quantile; no weights or golden arrays are committed.
"""

import json
import math
import sys

import numpy as np
import torch
from timesfm.configs import ForecastConfig
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

torch.set_num_threads(4)
model = TimesFM_2p5_200M_torch(torch_compile=False)
model.model.device = torch.device("cpu")
model.model.device_count = 1
model.load_checkpoint(sys.argv[1])
x = [0.03 * i + math.sin(i * 0.37) for i in range(65)]
positive = [0.01 * (i % 5) for i in range(65)]
cases = [
    ("short", [x[:19]], 17, False, [False] * 4),
    ("autoregressive", [x], 257, False, [False] * 4),
    ("continuous", [x], 257, True, [False] * 4),
    ("constant", [[7.25] * 33], 145, True, [False] * 4),
    ("single", [[0.0]], 8, False, [False] * 4),
    ("missing", [[None] * 40 + [*x[2:17], None, x[18], None]], 145, True, [False] * 4),
    ("long_context", [[math.sin(i * 0.1) * (1 + i / 50) for i in range(513)]], 145, True, [False] * 4),
    ("price_scale", [[65000.0 + 10 * v for v in x]], 145, True, [False] * 4),
    ("beyond_quantile_head", [x[:19]], 1025, False, [False] * 4),
    ("positive_batch", [positive, [None] * 40 + [-v for v in positive[40:]]],
     145, True, [False, True, False, False]),
]
cases += [(name, [positive], 145, True, flags) for name, flags in [
    ("flip", [True, False, False, False]),
    ("crossing", [False, False, True, False]),
    ("normalize", [False, False, False, True]),
    ("all_options", [True] * 4),
]]
results = []
for name, contexts, horizon, continuous, flags in cases:
    flip, make_positive, crossing, normalize = flags
    model.compile(ForecastConfig(
        max_context=math.ceil(len(contexts[0]) / 32) * 32,
        max_horizon=math.ceil(horizon / 128) * 128,
        per_core_batch_size=len(contexts),
        use_continuous_quantile_head=continuous,
        force_flip_invariance=flip,
        infer_is_positive=make_positive,
        fix_quantile_crossing=crossing,
        normalize_inputs=normalize,
    ))
    point, quantiles = model.forecast(horizon, [np.array(c, dtype=np.float32) for c in contexts])
    results.append(dict(name=name, contexts=contexts, horizon=horizon, continuous=continuous,
                        flags=flags, expected=np.concatenate((point.ravel(), quantiles[..., 1:].ravel())).tolist()))
json.dump(results, sys.stdout, allow_nan=False)
