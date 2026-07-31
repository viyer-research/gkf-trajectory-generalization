"""
Build a TrajImpute-style multi-agent trajectory-imputation dataset out of raw
GeoLife GPS logs, bounded to a first-pass scope:
  - only the 69 users with transportation-mode labels (labels.txt)
  - only points inside a generous Beijing bounding box
  - only the ~50 busiest days (by distinct-labeled-user overlap)
  - resampled onto one GLOBAL absolute 5-second grid (so different users'
    windows line up in time -- this is what makes a "scene"/graph possible
    at all, since GeoLife users log independently and there is no
    pre-built seq_start_end like TrajImpute ships)
  - windows of 8 observed + 12 predicted steps (100 s), non-overlapping
  - one dataset PER transportation mode (fit/evaluate separately per mode)
  - Easy (0-4 missing of 8 observed) / Hard (4-7 missing) protocols,
    mirroring TrajImpute's own simulation exactly
  - 70/15/15 user-disjoint train/val/test split

Reads directly from the GeoLife zip (no full extraction -- 18,740 tiny
files made on-disk extraction impractically slow in this sandbox).
"""
import zipfile
import re
import os
import json
import numpy as np

ZIP_PATH = '/sessions/sweet-dreamy-brahmagupta/mnt/Space-IoT-Proposal/ICST-2026-submission/Geo-life.zip'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'prepared')
os.makedirs(OUT_DIR, exist_ok=True)

BJ_LAT = (39.4, 41.1)
BJ_LON = (115.6, 117.6)
REF_LAT, REF_LON = 39.9042, 116.4074  # Beijing centre, for local projection

N_BUSY_DAYS = 50
GRID_SEC = 5          # global resampling step
N_OBS = 8
N_PRED = 12
WIN_LEN = N_OBS + N_PRED     # 20 steps = 100 s
MAX_GAP_SEC = 30      # don't trust interpolation across a gap bigger than this
MODE_MERGE = {}  # keep taxi separate from private car -- see cab-data follow-up


def project(lat, lon):
    x = (lon - REF_LON) * 111320.0 * np.cos(np.radians(REF_LAT))
    y = (lat - REF_LAT) * 110540.0
    return x, y


def parse_labels(z, uid):
    """Returns list of (start_epoch, end_epoch, mode) sorted by start."""
    path = f'Geolife Trajectories 1.3/Data/{uid}/labels.txt'
    try:
        with z.open(path) as f:
            lines = f.read().decode(errors='ignore').splitlines()[1:]
    except KeyError:
        return []
    out = []
    for l in lines:
        parts = l.split('\t')
        if len(parts) < 3:
            continue
        try:
            import datetime
            st = datetime.datetime.strptime(parts[0], '%Y/%m/%d %H:%M:%S')
            et = datetime.datetime.strptime(parts[1], '%Y/%m/%d %H:%M:%S')
        except ValueError:
            continue
        mode = parts[2].strip().lower()
        mode = MODE_MERGE.get(mode, mode)
        out.append((st.timestamp(), et.timestamp(), mode))
    out.sort()
    return out


def mode_at(intervals, t):
    # linear scan is fine: per-user interval counts are small (tens-hundreds)
    for st, et, mode in intervals:
        if st <= t <= et:
            return mode
    return None


def load_plt_points(z, name):
    """Returns list of (epoch_seconds, lat, lon)."""
    import datetime
    with z.open(name) as f:
        raw = f.read().decode(errors='ignore').splitlines()[6:]
    out = []
    for l in raw:
        parts = l.split(',')
        if len(parts) < 7:
            continue
        try:
            lat = float(parts[0]); lon = float(parts[1])
            date_s, time_s = parts[5], parts[6]
            dt = datetime.datetime.strptime(date_s + ' ' + time_s, '%Y-%m-%d %H:%M:%S')
        except Exception:
            continue
        out.append((dt.timestamp(), lat, lon))
    return out


def select_busy_days(z, labeled_users):
    pat = re.compile(r'Data/(\d+)/Trajectory/(\d{14})\.plt$')
    from collections import defaultdict
    user_day_files = defaultdict(lambda: defaultdict(list))
    for n in z.namelist():
        m = pat.search(n)
        if not m:
            continue
        uid, ts = m.groups()
        if uid not in labeled_users:
            continue
        user_day_files[uid][ts[:8]].append(n)
    day_users = defaultdict(set)
    for uid, days in user_day_files.items():
        for d in days:
            day_users[d].add(uid)
    ranked = sorted(day_users.items(), key=lambda kv: -len(kv[1]))

    def first_point_ok(name):
        with z.open(name) as f:
            for i, line in enumerate(f):
                if i < 6:
                    continue
                parts = line.decode(errors='ignore').strip().split(',')
                if len(parts) >= 2:
                    try:
                        lat, lon = float(parts[0]), float(parts[1])
                    except Exception:
                        return False
                    return BJ_LAT[0] <= lat <= BJ_LAT[1] and BJ_LON[0] <= lon <= BJ_LON[1]
                break
        return False

    selected = []
    for d, users in ranked:
        uid0 = sorted(users)[0]
        f0 = user_day_files[uid0][d][0]
        if first_point_ok(f0):
            selected.append(d)
        if len(selected) >= N_BUSY_DAYS:
            break
    return selected, user_day_files


def main():
    z = zipfile.ZipFile(ZIP_PATH)
    names = z.namelist()
    labeled_users = sorted(set(re.search(r'Data/(\d+)/labels.txt', n).group(1)
                                for n in names if n.endswith('labels.txt')))
    print(f'{len(labeled_users)} labeled users')

    busy_days, user_day_files = select_busy_days(z, set(labeled_users))
    print(f'{len(busy_days)} busy Beijing days selected: {busy_days[:10]}...')

    # per-user label intervals
    user_intervals = {uid: parse_labels(z, uid) for uid in labeled_users}

    # collect raw (t, x, y, mode) sequences per user, restricted to busy days
    # and Beijing bbox
    per_user_points = {}
    n_files = 0
    for uid in labeled_users:
        pts = []
        for d in busy_days:
            for fname in user_day_files.get(uid, {}).get(d, []):
                n_files += 1
                for t, lat, lon in load_plt_points(z, fname):
                    if not (BJ_LAT[0] <= lat <= BJ_LAT[1] and BJ_LON[0] <= lon <= BJ_LON[1]):
                        continue
                    mode = mode_at(user_intervals[uid], t)
                    if mode is None:
                        continue
                    x, y = project(lat, lon)
                    pts.append((t, x, y, mode))
        if pts:
            pts.sort()
            per_user_points[uid] = pts
    print(f'read {n_files} plt files; {len(per_user_points)} users have usable labeled points on busy days')
    total_pts = sum(len(v) for v in per_user_points.values())
    print(f'total labeled points in scope: {total_pts}')

    with open(os.path.join(OUT_DIR, 'stage1_summary.json'), 'w') as f:
        json.dump({
            'labeled_users': labeled_users,
            'busy_days': busy_days,
            'n_files_read': n_files,
            'users_with_points': list(per_user_points.keys()),
            'total_points': total_pts,
            'points_per_user': {u: len(v) for u, v in per_user_points.items()},
        }, f, indent=2)

    # stash raw per-user points to disk (npz) for the next stage, since
    # re-parsing the zip each time is the expensive part
    raw_dir = os.path.join(OUT_DIR, 'raw_points')
    os.makedirs(raw_dir, exist_ok=True)
    for uid, pts in per_user_points.items():
        arr_t = np.array([p[0] for p in pts])
        arr_x = np.array([p[1] for p in pts])
        arr_y = np.array([p[2] for p in pts])
        arr_mode = np.array([p[3] for p in pts])
        np.savez(os.path.join(raw_dir, f'{uid}.npz'), t=arr_t, x=arr_x, y=arr_y, mode=arr_mode)
    print('Saved raw per-user points to', raw_dir)


if __name__ == '__main__':
    main()
