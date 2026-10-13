"""Time-ordered splits with a label delay gap.

Fraud labels arrive late (chargebacks, scam claims, investigations), so the week before
any decision point has no reliable labels. Every split here leaves a 7-day gap between
the last labelled row used for fitting and the first row scored; rows inside the gap are
dropped, never used. No split is random.
"""

import numpy as np

DAY = 86400.0
GAP_DAYS = 7


def holdout(t, train_days=52, calib_days=14):
    """Train on days [0, 52), calibrate on [52, 66), skip the 7-day gap, test on the rest."""
    day = np.floor(t / DAY)
    tr = day < train_days
    ca = (day >= train_days) & (day < train_days + calib_days)
    te = day >= train_days + calib_days + GAP_DAYS
    return tr, ca, te


def prequential(t, folds=4, week=7):
    """Weekly folds, newest first: test week k, the gap week before it, and everything
    earlier for training. Yields (fold, train_mask, test_mask)."""
    day = np.floor(t / DAY)
    end = day.max() + 1
    for k in range(folds):
        test_end = end - k * week
        test_start = test_end - week
        train_end = test_start - GAP_DAYS
        yield k, day < train_end, (day >= test_start) & (day < test_end)
