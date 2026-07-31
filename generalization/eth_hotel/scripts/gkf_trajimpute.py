"""
Graph Kalman Filter (GKF) generalisation check: ETH-M (TrajImpute), Easy protocol.

This adapts the sensor-network graph Kalman filter from the ICST paper
(s_t = theta_tm*s_{t-1} + theta_sp*Abar@s_{t-1} + noise, node = mote) to
pedestrian trajectory imputation, where:
    - "node"  = a pedestrian present in the same scene (seq_start_end group)
    - "state" = the pedestrian's x (or y) coordinate, filtered independently
                per channel with SHARED theta_tm/theta_sp
    - "graph" = spatial-proximity adjacency among co-present pedestrians,
                built from each pedestrian's mean observed position
                (fixed for the scene, analogous to the published mote
                coordinates used in the ICST paper)

Fitting: theta_tm, theta_sp are fit by pooled least squares on consecutive
frame pairs of pred_traj (the fully-clean 12-frame future segment) across
every scene in the TRAIN split -- this avoids fitting on data that itself
has missing values.

Evaluation: the Easy-protocol TEST split stores each of the 181 underlying
trajectories 5 times (0,1,2,3,4 dropped coordinates respectively), with the
first 181 rows being the clean (m=0) copy of the same trajectories in the
same order -- confirmed empirically (non-NaN entries are bit-identical
across blocks). That gives genuine ground truth for every masked entry,
exactly the protocol TrajImpute's own Table 2 uses.

NOTE: the released seq_start_end for the test split only covers rows
0-180 (block m=0); we reuse that scene structure, offset by 181*i, for
the other four blocks, since the row order/identity is confirmed
unchanged across blocks. This is documented as a modelling assumption.
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import torchless_load as tl

DATASET = os.environ.get('GKF_DATASET', 'ETH-M')
DATA_ROOT = os.environ.get('GKF_DATA_ROOT', os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(DATA_ROOT, DATASET, 'Easy')
R_RADIUS = 2.0        # metres; pedestrian interaction radius for the graph
R_MEAS = 1e-4         # measurement noise (coordinates are ~exact)


def load_split(name):
    return tl.load(os.path.join(DATA_DIR, f'data_{name}.pkl'))


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
    """Representative (fixed) position per pedestrian in a scene: mean of
    whatever observed frames exist (matches the ICST paper's use of a
    fixed, geometry-only position to build the graph)."""
    seg = obs_traj[s:e]  # (n, 8, 2)
    return np.nanmean(seg, axis=1)  # (n, 2)


def fit_theta(train_pred_traj, train_sse):
    """Pooled least squares: state_t ~ w0*state_{t-1} + w1*(Abar@state_{t-1}),
    over every scene, every consecutive frame pair, both x and y channels."""
    X1_diag, X1_graph, Y = [], [], []
    for (s, e) in train_sse:
        n = e - s
        if n < 1:
            continue
        rep = scene_positions_from_pred(train_pred_traj, s, e)
        Abar = normalized_adjacency(rep, R_RADIUS)
        for ch in range(2):
            seq = train_pred_traj[s:e, :, ch]  # (n, 12)
            for t in range(1, seq.shape[1]):
                x1 = seq[:, t - 1]
                y = seq[:, t]
                X1_diag.append(x1)
                X1_graph.append(Abar @ x1)
                Y.append(y)
    X1_diag = np.concatenate(X1_diag)
    X1_graph = np.concatenate(X1_graph)
    Y = np.concatenate(Y)

    # diagonal-only (no-graph) fit
    A_diag = X1_diag[:, None]
    w_diag, *_ = np.linalg.lstsq(A_diag, Y, rcond=None)
    resid_diag = Y - A_diag @ w_diag
    q_diag = float(resid_diag.var())

    # graph fit
    A_graph = np.stack([X1_diag, X1_graph], axis=1)
    w_graph, *_ = np.linalg.lstsq(A_graph, Y, rcond=None)
    resid_graph = Y - A_graph @ w_graph
    q_graph = float(resid_graph.var())

    return {
        'no_graph': {'theta_tm': float(w_diag[0]), 'theta_sp': 0.0, 'Q': q_diag},
        'graph': {'theta_tm': float(w_graph[0]), 'theta_sp': float(w_graph[1]), 'Q': q_graph},
    }


def scene_positions_from_pred(pred_traj, s, e):
    seg = pred_traj[s:e]
    return np.nanmean(seg, axis=1)


def run_kf_scene(seq_obs, Abar, theta_tm, theta_sp, Q, R=R_MEAS):
    """seq_obs: (n, T) with NaN at missing entries. Returns filled (n, T)."""
    n, T = seq_obs.shape
    F = theta_tm * np.eye(n) + theta_sp * Abar
    obs_mask = ~np.isnan(seq_obs)

    # initial state: first observed value per node, else column-mean of
    # whatever is observed anywhere in the scene, else 0.
    s0 = np.full(n, np.nan)
    for i in range(n):
        row = seq_obs[i]
        obs_idx = np.where(~np.isnan(row))[0]
        if len(obs_idx):
            s0[i] = row[obs_idx[0]]
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
            innov = seq_obs[:, t][o] - s[o]
            s = s + K @ innov
            I = np.eye(n) - K @ H
            P = I @ P @ I.T + K @ (np.eye(o.sum()) * R) @ K.T
        filled[:, t] = s
    return filled


def metrics(pred, target, mask):
    pred = pred[mask]
    target = target[mask]
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mre = float(np.sum(np.abs(err)) / np.sum(np.abs(target)))
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MRE': mre, 'n': int(mask.sum())}


def get_test_scene_list(block0_sse, n_blocks=5, block_size=181):
    """Offset the block-0 scene structure for blocks 1..4 (see module
    docstring for why this is needed)."""
    all_sse = []
    for b in range(n_blocks):
        off = b * block_size
        all_sse.extend([(s + off, e + off) for (s, e) in block0_sse])
    return all_sse


def persistence_baseline(test_obs):
    persist = test_obs.copy()
    N_test = persist.shape[0]
    for i in range(N_test):
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


def run_protocol(protocol, fit, clean_block, block0_sse, block_size):
    """protocol: 'Easy' or 'Hard'. clean_block/block0_sse come from Easy
    (Hard's test split has no clean copy of its own, but shares the same
    underlying trajectories/scene structure -- verified bit-identical
    on overlapping non-NaN entries)."""
    data_dir = os.path.join(DATA_ROOT, DATASET, protocol)
    test = tl.load(os.path.join(data_dir, 'data_test.pkl'))
    test_obs = test['obs_traj']
    test_mask = test['missing_mask']
    N_test = test_obs.shape[0]
    n_blocks = N_test // block_size
    assert N_test % block_size == 0, (protocol, N_test)
    ground_truth = np.tile(clean_block, (n_blocks, 1, 1))
    assert not np.isnan(ground_truth).any()

    test_sse_full = get_test_scene_list(block0_sse, n_blocks=n_blocks, block_size=block_size)

    per_model = {}
    filled_by_model = {}
    for model_name in ['no_graph', 'graph']:
        theta_tm = fit[model_name]['theta_tm']
        theta_sp = fit[model_name]['theta_sp']
        Q = fit[model_name]['Q']
        filled = np.full_like(test_obs, np.nan)
        for (s, e) in test_sse_full:
            n = e - s
            rep = scene_positions(test_obs, s, e)
            Abar = normalized_adjacency(rep, R_RADIUS) if theta_sp != 0 else np.zeros((n, n))
            for ch in range(2):
                seq = test_obs[s:e, :, ch]
                filled[s:e, :, ch] = run_kf_scene(seq, Abar, theta_tm, theta_sp, Q)
        filled_by_model[model_name] = filled
        per_model[model_name] = metrics(filled, ground_truth, test_mask)

    persist = persistence_baseline(test_obs)
    filled_by_model['persistence'] = persist
    per_model['persistence'] = metrics(persist, ground_truth, test_mask)

    # breakdown by exact number of missing coordinates (m), one block per m
    m_values = sorted(set(int(v) for v in (np.isnan(test_obs).sum(axis=(1, 2)) // 2)))
    breakdown = {mv: {} for mv in m_values}
    nan_per_row = np.isnan(test_obs).sum(axis=(1, 2)) // 2
    for mv in m_values:
        row_idx = np.where(nan_per_row == mv)[0]
        sub_mask = test_mask[row_idx]
        sub_gt = ground_truth[row_idx]
        for model_name, filled in filled_by_model.items():
            breakdown[mv][model_name] = metrics(filled[row_idx], sub_gt, sub_mask)

    return {
        'overall': per_model,
        'by_m': breakdown,
        'n_test_rows': int(N_test),
    }


def main():
    print(f'Dataset: {DATASET}')
    print('Loading Easy/train ...')
    train = load_split('train')

    print('Fitting theta_tm / theta_sp on train pred_traj (Easy split; pred_traj has no missing values in either protocol) ...')
    fit = fit_theta(train['pred_traj'], train['seq_start_end'])
    print(json.dumps(fit, indent=2))

    print('Loading Easy/test (source of ground truth: the m=0 rows are the clean, 0-missing copy) ...')
    easy_test = load_split('test')
    nan_per_row = np.isnan(easy_test['obs_traj']).sum(axis=(1, 2)) // 2
    block_size = int((nan_per_row == 0).sum())
    print(f'Auto-detected block_size (unique test trajectories) = {block_size}')
    clean_block = easy_test['obs_traj'][:block_size].copy()
    block0_sse = easy_test['seq_start_end']
    assert np.array(block0_sse).max() == block_size, (np.array(block0_sse).max(), block_size)

    all_results = {'dataset': DATASET, 'fit': fit, 'radius': R_RADIUS, 'block_size': block_size}
    for protocol in ['Easy', 'Hard']:
        print(f'\n=== Running {protocol} protocol ===')
        res = run_protocol(protocol, fit, clean_block, block0_sse, block_size)
        all_results[protocol] = res
        print(f"{protocol} overall:")
        for model_name, m in res['overall'].items():
            print(f"  {model_name:12s} MAE={m['MAE']:.4f}  MSE={m['MSE']:.4f}  RMSE={m['RMSE']:.4f}  MRE={m['MRE']:.4f}  n={m['n']}")
        print(f"{protocol} by missing-count m:")
        for mv, models in res['by_m'].items():
            print(f"  m={mv}:")
            for model_name, m in models.items():
                print(f"    {model_name:12s} MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  n={m['n']}")

    out_path = os.path.join(os.path.dirname(__file__), '..', f'results_{DATASET}.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nSaved', out_path)


if __name__ == '__main__':
    main()
