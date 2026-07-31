"""
Stage 2: resample each user's raw (t,x,y,mode) points onto ONE global
absolute 5-second grid, per contiguous same-mode run, and cut out 20-step
(8 obs + 12 pred) windows aligned to that global grid. Two different users'
windows sharing the same global window index are, by construction, the same
100-second real-world period -- that's what makes a graph/scene possible.
"""
import os
import glob
import numpy as np
import json
from collections import defaultdict

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'prepared', 'raw_points')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'prepared')

GRID_SEC = 5
N_OBS = 8
N_PRED = 12
WIN_LEN = N_OBS + N_PRED
MAX_GAP_SEC = 30


def resample_run(t, x, y, grid_sec=GRID_SEC, max_gap=MAX_GAP_SEC):
    """t sorted ascending, single mode, single contiguous run (small internal
    gaps allowed, checked here). Returns dict: tick(int) -> (x,y)."""
    if len(t) < 2:
        return {}
    k_lo = int(np.ceil(t[0] / grid_sec))
    k_hi = int(np.floor(t[-1] / grid_sec))
    if k_hi < k_lo:
        return {}
    ticks = np.arange(k_lo, k_hi + 1)
    tt = ticks * grid_sec
    idx_right = np.searchsorted(t, tt, side='left')
    idx_right = np.clip(idx_right, 1, len(t) - 1)
    idx_left = idx_right - 1
    t_l, t_r = t[idx_left], t[idx_right]
    gap = t_r - t_l
    ok = gap <= max_gap
    # avoid div by zero when tt exactly equals a sample
    denom = np.where(t_r > t_l, t_r - t_l, 1.0)
    frac = np.where(t_r > t_l, (tt - t_l) / denom, 0.0)
    xi = x[idx_left] + frac * (x[idx_right] - x[idx_left])
    yi = y[idx_left] + frac * (y[idx_right] - y[idx_left])
    out = {}
    for i, k in enumerate(ticks):
        if ok[i]:
            out[int(k)] = (float(xi[i]), float(yi[i]))
    return out


def split_runs(t, x, y, mode):
    """Split into contiguous same-mode runs. A run also breaks on any real
    time gap > 3600s (an hour) purely to keep run arrays small/sane; the
    resample step already enforces the tighter MAX_GAP_SEC per grid tick."""
    runs = []
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or mode[i] != mode[start] or (t[i] - t[i - 1]) > 3600:
            runs.append((mode[start], t[start:i], x[start:i], y[start:i]))
            start = i
    return runs


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, '*.npz')))
    print(f'{len(files)} user files')

    # windows[mode] = list of dict(user=, win=, ticks=[...], xy=(20,2) array, valid=bool)
    windows_by_mode = defaultdict(list)

    for fp in files:
        uid = os.path.splitext(os.path.basename(fp))[0]
        d = np.load(fp, allow_pickle=True)
        t, x, y, mode = d['t'], d['x'], d['y'], d['mode']
        order = np.argsort(t)
        t, x, y, mode = t[order], x[order], y[order], mode[order]
        runs = split_runs(t, x, y, mode)
        for m, rt, rx, ry in runs:
            if len(rt) < 2:
                continue
            grid = resample_run(rt, rx, ry)
            if not grid:
                continue
            ticks_sorted = sorted(grid.keys())
            tick_set = set(ticks_sorted)
            # candidate window starts: any tick that is a multiple of WIN_LEN
            # AND all WIN_LEN ticks from there are present
            win_starts = set(k - (k % WIN_LEN) for k in ticks_sorted)
            for w0 in win_starts:
                needed = list(range(w0, w0 + WIN_LEN))
                if all(k in tick_set for k in needed):
                    xy = np.array([grid[k] for k in needed])  # (20,2)
                    windows_by_mode[str(m)].append({
                        'user': uid,
                        'win_start_tick': int(w0),
                        'xy': xy,
                    })

    summary = {m: len(v) for m, v in windows_by_mode.items()}
    print('windows per mode:', summary)

    os.makedirs(OUT_DIR, exist_ok=True)
    for m, wins in windows_by_mode.items():
        users = sorted(set(w['user'] for w in wins))
        xy_arr = np.stack([w['xy'] for w in wins])  # (N,20,2)
        win_start = np.array([w['win_start_tick'] for w in wins])
        user_arr = np.array([w['user'] for w in wins])
        np.savez(os.path.join(OUT_DIR, f'windows_{m}.npz'),
                  xy=xy_arr, win_start_tick=win_start, user=user_arr)
        print(m, 'saved', xy_arr.shape, 'unique users', len(users))

    with open(os.path.join(OUT_DIR, 'stage2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
