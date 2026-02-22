import numpy as np

# window: (T, C) or (C, T) etc.
if not np.isfinite(window).all():
    print("BAD INPUT window:", key,
          "nan%", np.isnan(window).mean(),
          "inf%", np.isinf(window).mean(),
          "min/max finite:", np.nanmin(window), np.nanmax(window))
    # optionally skip
    continue