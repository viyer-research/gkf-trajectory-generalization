"""
Same graph-Kalman-filter generalisation check as gkf_trajimpute.py
(ETH-M/HOTEL-M), applied to the GeoLife-derived multi-mode dataset built by
geolife_prepare.py -> geolife_windows.py -> geolife_split.py.

Differences from the TrajImpute version:
  - loads .npz (our own format) instead of torch-pickled .pkl
  - ground truth comes directly from `obs_traj_clean` (we generated the
    missingness ourselves, so no need for TrajImpute's 5x-replicated-block
    reconstruction trick)
  - one fit + evaluation per (mode, protocol) since dynamics differ by
    transportation mode (walk/bike/bus/car/subway)
"""
import os
import json
import numpy as np

GEOLIFE_ROOT = os.path.join(os.path.dirname(__file__), '..', 'GeoLife-M')
MODES = ['walk', 'bus', 'car', 'bike', 'subway', 'taxi']
R_RADIUS = 50.0
R_MEAS = 1e-4


def load(mode, protocol, split):
    fp = os.path.join(GEOLIFE_ROOT, mode, protocol, f'data_{split}.npz')
    d = np.load(fp, allow_pickle=True)
    return {k: d[k] for k in d.files}


def normalized_adjacency(rep_pos, radius):
    n = rep_pos.shape[0]
    if n == 1:
        return np.zeros((1, 1))
    D = np.sqrt(((rep_pos[:, None, :] - rep_pos[None, :, :]) ** 2).sum(-1))
    A = (D < radius).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    Ahat = A + np.eye(n)
    deg = Ahat.sum(1)
    Dinv = np.diag(1.0 / np.sqrt(deg))
    return Dinv @ Ahat @ Dinv


def scene_positions(obs_traj, s, e):
    seg = obs_traj[s:e]
    return np.nanmean(seg, axis=1)


def fit_theta(train_pred_traj, train_sse):
    """NOTE: GeoLife coordinates are projected relative to a fixed Beijing
    reference point, so a single trajectory's absolute (x,y) can be tens of
    thousands of metres. theta_tm is never exactly 1.0, so applying it
    multiplicatively to such large numbers injects an error purely from
    scale (e.g. theta_tm=1.00026 on x=36000 m is a ~9 m artifact per step)
    that has nothing to do with the model being wrong. Every scene is
    therefore centred on its own per-node mean position before fitting, and
    the same centring is undone after filtering at evaluation time. ETH-M/
    HOTEL-M didn't need this because their coordinates were already
    camera-local (a few metres from an arbitrary local origin)."""
    X1_diag, X1_graph, Y = [], [], []
    for (s, e) in train_sse:
        n = e - s
        if n < 1:
            continue
        rep = np.nanmean(train_pred_traj[s:e], axis=1)
        Abar = normalized_adjacency(rep, R_RADIUS)
        for ch in range(2):
            seq = train_pred_traj[s:e, :, ch] - rep[:, ch:ch + 1]
            for t in range(1, seq.shape[1]):
                x1 = seq[:, t - 1]
                y = seq[:, t]
                X1_diag.append(x1)
                X1_graph.append(Abar @ x1)
                Y.append(y)
    if not X1_diag:
        return None
    X1_diag = np.concatenate(X1_diag)
    X1_graph = np.concatenate(X1_graph)
    Y = np.concatenate(Y)

    A_diag = X1_diag[:, None]
    w_diag, *_ = np.linalg.lstsq(A_diag, Y, rcond=None)
    q_diag = float((Y - A_diag @ w_diag).var())

    A_graph = np.stack([X1_diag, X1_graph], axis=1)
    w_graph, *_ = np.linalg.lstsq(A_graph, Y, rcond=None)
    q_graph = float((Y - A_graph @ w_graph).var())

    return {
        'no_graph': {'theta_tm': float(w_diag[0]), 'theta_sp': 0.0, 'Q': q_diag},
        'graph': {'theta_tm': float(w_graph[0]), 'theta_sp': float(w_graph[1]), 'Q': q_graph},
    }


def run_kf_scene(seq_obs, Abar, theta_tm, theta_sp, Q, R=R_MEAS):
    n, T = seq_obs.shape
    F = theta_tm * np.eye(n) + theta_sp * Abar
    obs_mask = ~np.isnan(seq_obs)
    s0 = np.full(n, np.nan)
    for i in range(n):
        obs_idx = np.where(obs_mask[i])[0]
        if len(obs_idx):
            s0[i] = seq_obs[i, obs_idx[0]]
    if np.isnan(s0).any():
        fallback = np.nanmean(seq_obs) if not np.all(np.isnan(seq_obs)) else 0.0
        s0[np.isnan(s0)] = fallback
    s = s0.copy()
    P = np.eye(n) * 1.0
    Qm = np.eye(n) * Q
    filled = np.zeros((n, T))
    for t in range(T):
        if t > 0:
            s = F @ s
            P = F @ P @ F.T + Qm
        o = obs_mask[:, t]
        if o.any():
            H = np.eye(n)[o]
            S_ = H @ P @ H.T + np.eye(o.sum()) * R
            K = P @ H.T @ np.linalg.inv(S_)
            s = s + K @ (seq_obs[:, t][o] - s[o])
            I = np.eye(n) - K @ H
            P = I @ P @ I.T + K @ (np.eye(o.sum()) * R) @ K.T
        filled[:, t] = s
    return filled


def metrics(pred, target, mask):
    pred = pred[mask]
    target = target[mask]
    if len(pred) == 0:
        return {'MAE': None, 'MSE': None, 'RMSE': None, 'MRE': None, 'n': 0}
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    denom = np.sum(np.abs(target))
    mre = float(np.sum(np.abs(err)) / denom) if denom > 0 else None
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MRE': mre, 'n': int(mask.sum())}


def persistence_baseline(test_obs):
    persist = test_obs.copy()
    N = persist.shape[0]
    for i in range(N):
        for ch in range(2):
            seq = persist[i, :, ch]
            last = None
            for t in range(len(seq)):
                if np.isnan(seq[t]):
                    if last is not None:
                        seq[t] = last
                else:
                    last = seq[t]
            if np.isnan(seq).any():
                obs_idx = np.where(~np.isnan(seq))[0]
                if len(obs_idx):
                    seq[np.isnan(seq)] = seq[obs_idx[0]]
    return persist


def run_mode_protocol(mode, protocol):
    train = load(mode, protocol, 'train')
    test = load(mode, protocol, 'test')

    fit = fit_theta(train['pred_traj'], train['seq_start_end'])
    if fit is None:
        return None

    test_obs = test['obs_traj']
    test_mask = test['missing_mask']
    ground_truth = test['obs_traj_clean']
    sse = test['seq_start_end']

    filled_by_model = {}
    per_model = {}
    for model_name in ['no_graph', 'graph']:
        theta_tm = fit[model_name]['theta_tm']
        theta_sp = fit[model_name]['theta_sp']
        Q = fit[model_name]['Q']
        filled = np.full_like(test_obs, np.nan)
        for (s, e) in sse:
            n = e - s
            rep = scene_positions(test_obs, s, e)  # per-node centre, see fit_theta note
            Abar = normalized_adjacency(rep, R_RADIUS) if theta_sp != 0 else np.zeros((n, n))
            for ch in range(2):
                seq = test_obs[s:e, :, ch] - rep[:, ch:ch + 1]
                filled_centered = run_kf_scene(seq, Abar, theta_tm, theta_sp, Q)
                filled[s:e, :, ch] = filled_centered + rep[:, ch:ch + 1]
        filled_by_model[model_name] = filled
        per_model[model_name] = metrics(filled, ground_truth, test_mask)

    persist = persistence_baseline(test_obs)
    filled_by_model['persistence'] = persist
    per_model['persistence'] = metrics(persist, ground_truth, test_mask)

    return {'fit': fit, 'overall': per_model, 'n_test_rows': int(test_obs.shape[0])}


def main():
    all_results = {'radius': R_RADIUS}
    for mode in MODES:
        mode_dir = os.path.join(GEOLIFE_ROOT, mode, 'Easy')
        if not os.path.exists(mode_dir):
            continue
        all_results[mode] = {}
        for protocol in ['Easy', 'Hard']:
            res = run_mode_protocol(mode, protocol)
            if res is None:
                print(f'{mode}/{protocol}: skipped (no data)')
                continue
            all_results[mode][protocol] = res
            print(f'\n=== {mode} / {protocol} (n_test_rows={res["n_test_rows"]}) ===')
            print('  fit:', json.dumps(res['fit']))
            for model_name, m in res['overall'].items():
                if m['n'] == 0:
                    print(f'  {model_name:12s} n=0 (no masked entries)')
                    continue
                print(f'  {model_name:12s} MAE={m["MAE"]:.4f}  MSE={m["MSE"]:.4f}  RMSE={m["RMSE"]:.4f}  MRE={m["MRE"]:.4f}  n={m["n"]}')

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results_GeoLife.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nSaved', out_path)


if __name__ == '__main__':
    main()
