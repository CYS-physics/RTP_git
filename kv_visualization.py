from pathlib import Path

import numpy as np


def periodic(x, L):
    return -L / 2 + (x + L / 2) % L


def trajectory_shape(load):
    return {
        key: load[key].shape
        for key in load.files
        if key.endswith("_traj") or key == "time"
    }


def time_by_replica(arr, name):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")


def moving_average_1d(arr, window_size):
    arr = np.asarray(arr).reshape(-1)
    if window_size <= 1:
        return arr
    if arr.size < window_size:
        return np.array([])
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(arr, weights, mode="valid")


def available_values(base, key):
    prefix = f"{key}="
    return sorted(
        float(path.name.split(prefix, 1)[1])
        for path in base.glob(f"{prefix}*")
        if path.is_dir()
    )


def available_sizes(base, M):
    m_dir = base / f"M={M}"
    return sorted(
        int(path.name.split("N=", 1)[1])
        for path in m_dir.glob("N=*")
        if path.is_dir()
    )


def available_velocities(base, k_s):
    velocities = set()
    for state in base.glob("M=*/N=*/*.npz"):
        parts = state.stem.split("_")
        if len(parts) >= 3 and np.isclose(float(parts[0]), k_s):
            velocities.add(float(parts[1]))
    return sorted(velocities)


def load_phase_trajectory(load, replica=0, timewin=10, time_init=0, time_end=None):
    time = np.asarray(load["time"])
    X = time_by_replica(load["X_traj"], "X_traj")
    X_s = time_by_replica(load["X_s_traj"], "X_s_traj")

    if X.shape != X_s.shape:
        raise ValueError(f"X_traj shape {X.shape} != X_s_traj shape {X_s.shape}")
    if X.shape[0] != time.size:
        raise ValueError(f"time length {time.size} != trajectory length {X.shape[0]}")
    if not 0 <= replica < X.shape[1]:
        raise IndexError(f"replica {replica} out of range for {X.shape[1]} replicas")

    L = float(load["L"]) if "L" in load else 200.0
    dt = float(np.median(np.diff(time))) if time.size > 1 else 1.0
    X_rep = X[:, replica]
    X_s_rep = X_s[:, replica]

    dX = periodic(X_rep[1:] - X_rep[:-1], L)
    dX_s = periodic(X_s_rep[1:] - X_s_rep[:-1], L)
    rel_x = np.cumsum(dX - dX_s)

    if "v_traj" in load:
        v = time_by_replica(load["v_traj"], "v_traj")[1:, replica]
    else:
        v = dX / dt

    x = moving_average_1d(rel_x, timewin)
    v = moving_average_1d(v, timewin)
    t = time[1 + timewin - 1 :]

    n = min(x.size, v.size, t.size)
    if time_end is None:
        time_end = n
    sl = slice(time_init, min(time_end, n))
    return x[sl], v[sl], t[sl]


def load_kdx_trajectory(load, k_s, replica=None):
    time = np.asarray(load["time"])
    X = time_by_replica(load["X_traj"], "X_traj")
    X_s = time_by_replica(load["X_s_traj"], "X_s_traj")

    if X.shape != X_s.shape:
        raise ValueError(f"X_traj shape {X.shape} != X_s_traj shape {X_s.shape}")
    if X.shape[0] != time.size:
        raise ValueError(f"time length {time.size} != trajectory length {X.shape[0]}")

    L = float(load["L"]) if "L" in load else 200.0
    kdx = k_s * periodic(X - X_s, L)

    if replica is None:
        return time, kdx
    if not 0 <= replica < kdx.shape[1]:
        raise IndexError(f"replica {replica} out of range for {kdx.shape[1]} replicas")
    return time, kdx[:, replica]


def add_colored_trajectory(ax, x, y, t, linewidth=0.1, cmap="rainbow"):
    from matplotlib.collections import LineCollection

    if len(x) < 2 or len(y) < 2:
        return None
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=linewidth)
    lc.set_array(np.asarray(t)[: len(segments)])
    ax.add_collection(lc)
    return lc


def default_data_dir(name="kv_under4"):
    base_dirs = [Path(f"/data/{name}"), Path.cwd() / "data" / name]
    return next((path for path in base_dirs if path.exists()), base_dirs[-1])
