"""
Stage 3: turn each mode's window pool into a TrajImpute-shaped dataset:
  - seq_start_end scenes = windows sharing the same absolute win_start_tick
    (verified empirically bimodal: co-present pairs are either <50m apart,
    i.e. a real encounter, or many km apart, i.e. same clock time only by
    coincidence -- R_RADIUS=50m cleanly separates the two)
  - obs_traj/pred_traj = the clean 8+12 step window, always complete by
    construction (only fully-interpolatable windows were kept in stage 2)
  - Easy/Hard missingness simulated on the observed 8 steps, mirroring
    TrajImpute's own m~Uniform{0..4} / m~Uniform{4..7} protocol -- but here
    the clean ground truth is saved directly (obs_traj_clean) instead of
    TrajImpute's 5x-replicated-test-block trick, since we generate this
    dataset ourselves and don't need to recover a hidden original.
  - user-disjoint 70/15/15 train/val/test split, per mode.
"""
import os
import json
import numpy as np

IN_DIR = os.path.join(os.path.dirname(__file__), '..', 'prepared')
OUT_ROOT = os.path.join(os.path.dirname(__file__), '..', 'GeoLife-M')

MODES = ['walk', 'bus', 'car', 'bike', 'subway', 'taxi']
R_RADIUS = 50.0  # metres; empirically bimodal separation, see stage3 notes
SEED = 0


def build_scenes(win_start_tick):
    order = np.argsort(win_start_tick, kind='stable')
    sorted_ticks = win_start_tick[order]
    scenes = []
    i = 0
    n = len(sorted_ticks)
    while i < n:
        j = i + 1
        while j < n and sorted_ticks[j] == sorted_ticks[i]:
            j += 1
        scenes.append((i, j))
        i = j
    return order, scenes


def split_users(users, seed=SEED, fracs=(0.7, 0.15, 0.15)):
    """70/15/15 by user, with at least 1 user in val/test whenever n allows it."""
    rng = np.random.default_rng(seed)
    users = list(users)
    rng.shuffle(users)
    n = len(users)
    if n >= 3:
        n_train = max(1, min(n - 2, int(round(n * fracs[0]))))
        remaining = n - n_train
        n_val = max(1, remaining // 2)
        n_val = min(n_val, remaining - 1)
    elif n == 2:
        n_train, n_val = 1, 1
    else:
        n_train, n_val = n, 0
    train_u = set(users[:n_train])
    val_u = set(users[n_train:n_train + n_val])
    test_u = set(users[n_train + n_val:])
    return train_u, val_u, test_u


def simulate_missing(obs, rng, m_choices):
    """obs: (N,8,2). Returns obs_with_nan, mask (both (N,8,2))."""
    N, T, _ = obs.shape
    out = obs.copy()
    mask = np.zeros((N, T), dtype=bool)
    for i in range(N):
        m = rng.choice(m_choices)
        if m > 0:
            idx = rng.choice(T, size=m, replace=False)
            mask[i, idx] = True
    out[mask] = np.nan
    full_mask = np.repeat(mask[:, :, None], 2, axis=2)
    return out, full_mask


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    all_summary = {}
    for mode in MODES:
        fp = os.path.join(IN_DIR, f'windows_{mode}.npz')
        if not os.path.exists(fp):
            continue
        d = np.load(fp, allow_pickle=True)
        xy = d['xy']  # (N,20,2)
        win_start_tick = d['win_start_tick']
        users = d['user']

        uniq_users = sorted(set(users.tolist()))
        train_u, val_u, test_u = split_users(uniq_users)
        print(mode, 'users', len(uniq_users), 'train/val/test users', len(train_u), len(val_u), len(test_u))

        mode_summary = {}
        for split_name, split_users_set in [('train', train_u), ('val', val_u), ('test', test_u)]:
            sel = np.array([u in split_users_set for u in users])
            if sel.sum() == 0:
                mode_summary[split_name] = 0
                continue
            xy_s = xy[sel]
            wst_s = win_start_tick[sel]
            users_s = users[sel]

            order, scenes = build_scenes(wst_s)
            xy_o = xy_s[order]
            users_o = users_s[order]

            obs_clean = xy_o[:, :8, :]
            pred_clean = xy_o[:, 8:, :]

            for protocol, m_choices in [('Easy', [0, 1, 2, 3, 4]), ('Hard', [4, 5, 6, 7])]:
                rng = np.random.default_rng(SEED + hash((mode, split_name, protocol)) % 10000)
                obs_nan, mask = simulate_missing(obs_clean, rng, m_choices)
                out_dir = os.path.join(OUT_ROOT, mode, protocol)
                os.makedirs(out_dir, exist_ok=True)
                np.savez(os.path.join(out_dir, f'data_{split_name}.npz'),
                          obs_traj=obs_nan.astype(np.float32),
                          obs_traj_clean=obs_clean.astype(np.float32),
                          pred_traj=pred_clean.astype(np.float32),
                          missing_mask=mask,
                          seq_start_end=np.array(scenes, dtype=np.int64),
                          users=users_o)
            mode_summary[split_name] = int(sel.sum())
        all_summary[mode] = mode_summary
        print(' ', mode_summary)

    with open(os.path.join(OUT_ROOT, 'split_summary.json'), 'w') as f:
        json.dump(all_summary, f, indent=2)
    print('Saved to', OUT_ROOT)


if __name__ == '__main__':
    main()
