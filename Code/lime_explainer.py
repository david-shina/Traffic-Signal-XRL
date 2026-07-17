#!/usr/bin/env python3
"""
LIME Explainer for DDQN Traffic Signal Control Policy.

Loads the trained policy SavedModel, reconstructs the Q-network,
collects states via SUMO rollouts, and runs LIME analysis.
Saves visualisations to lime_outputs/.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import traci
import matplotlib
matplotlib.use("Agg")
matplotlib.rcdefaults()
import matplotlib.pyplot as plt
import subprocess
from lime.lime_tabular import LimeTabularExplainer

from tf_agents.networks import sequential

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

BASE         = r"C:\Users\Dell\Desktop\Project\Implementation\Sim1"
SUMO_CFG     = os.path.join(BASE, "SUMO_FILES", "sim.sumocfg")
ADDITIONAL   = os.path.join(BASE, "SUMO_FILES", "sim.add.xml")
TLS_ID       = "Node2"
POLICY_PATH  = os.path.join(BASE, "policy_checkpoints_phase2", "best_policy")
OUT_DIR      = os.path.join(BASE, "lime_outputs")
N_EPISODES   = 8
MAX_STEPS    = 6000
ACTIONS      = 2

DETECTORS = [
    "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2",
    "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2",
]

MIN_GREEN_TIME = 25
TRIP_END_STEP  = 4200
MAX_QUEUE     = 50.0
MAX_OCCUPANCY = 100.0
MAX_SPEED     = 20.0
MAX_WAIT      = 300.0
MAX_EV        = 5.0
MAX_TIME_IN   = 300.0

os.makedirs(OUT_DIR, exist_ok=True)
NUM_PHASES    = None
FEATURE_NAMES = None


def query_num_phases() -> int:
    sumo_dir = os.path.join(BASE, "SUMO_FILES")
    net_xml  = os.path.join(sumo_dir, "sim.net.xml")
    traci.start(["sumo", "-n", net_xml, "--no-step-log", "--no-warnings"])
    n = len(traci.trafficlight.getAllProgramLogics(TLS_ID)[0].phases)
    traci.close()
    print(f"  Detected {n} phases for TLS '{TLS_ID}'")
    return n


def build_feature_names(num_phases: int) -> list:
    names = []
    for det in DETECTORS:
        for feat in ["queue", "occ", "speed", "mean_wait", "ev_count"]:
            names.append(f"{det}_{feat}")
    names.append("time_since_switch")
    for p in range(num_phases):
        names.append(f"phase_{p}")
    names.extend(["pressure", "demand"])
    return names


def gather_detector_data():
    data = []
    for det in DETECTORS:
        veh_ids   = traci.lanearea.getLastStepVehicleIDs(det)
        queue     = traci.lanearea.getJamLengthVehicle(det)
        occ       = traci.lanearea.getLastStepOccupancy(det)
        speed     = traci.lanearea.getLastStepMeanSpeed(det)
        veh_count = traci.lanearea.getLastStepVehicleNumber(det)
        ev_count, wait_times = 0.0, []
        for vid in veh_ids:
            try:
                wait_times.append(traci.vehicle.getWaitingTime(vid))
                if traci.vehicle.getVehicleClass(vid) == "emergency":
                    ev_count += 1.0
            except traci.TraCIException:
                continue
        mean_wait = float(np.mean(wait_times)) if wait_times else 0.0
        data.append(dict(queue=queue, occ=occ, speed=speed, veh_count=veh_count,
                         mean_wait=mean_wait, ev_count=ev_count))
    return data


def get_state(det_data, current_phase, time_since_switch, num_phases):
    features = []
    for d in det_data:
        features.extend([d["queue"], d["occ"], d["speed"], d["mean_wait"], d["ev_count"]])
    features.append(float(time_since_switch))
    max_vals = np.array(
        [MAX_QUEUE, MAX_OCCUPANCY, MAX_SPEED, MAX_WAIT, MAX_EV] * len(DETECTORS)
        + [MAX_TIME_IN], dtype=np.float32)
    continuous = np.array(features, dtype=np.float32)
    normalized = np.clip(continuous / (max_vals + 1e-5), 0.0, 1.0)
    total_queue = sum(d["queue"] for d in det_data)
    pressure    = np.clip(total_queue / (MAX_QUEUE * len(DETECTORS)), 0.0, 1.0)
    total_veh   = sum(d["veh_count"] for d in det_data)
    demand      = np.clip(total_veh / (10.0 * len(DETECTORS)), 0.0, 1.0)
    phase_oh = np.zeros(num_phases, dtype=np.float32)
    if 0 <= current_phase < num_phases:
        phase_oh[current_phase] = 1.0
    return np.concatenate([normalized, phase_oh, [pressure, demand]]).astype(np.float32)


def build_q_network(state_dim):
    net = sequential.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(state_dim,)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(ACTIONS),
    ])
    net(tf.zeros((1, state_dim)))
    return net


def _safe_sort_key(name):
    for segment in name.split("/"):
        try:
            return int(segment)
        except ValueError:
            continue
    return 999


def load_policy_q_network(saved_model_path, state_dim):
    q_net = build_q_network(state_dim)
    ckpt_path = os.path.join(saved_model_path, "variables", "variables")
    if not os.path.exists(ckpt_path + ".index"):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.index")
    print(f"  Loading weights from: {ckpt_path}")
    var_list = tf.train.list_variables(ckpt_path)
    _EXCLUDE = ("train_step", "CHECKPOINTABLE", "optimizer", "Adam")
    weight_vars = [(n, tuple(s)) for n, s in var_list
                   if len(s) > 0 and not any(ex in n for ex in _EXCLUDE)]
    weight_vars.sort(key=lambda x: _safe_sort_key(x[0]))
    print(f"  Found {len(weight_vars)} weight tensors in checkpoint")
    vi = 0
    for layer in q_net.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            name, shape = weight_vars[vi]
            assert shape == tuple(layer.kernel.shape)
            layer.kernel.assign(tf.train.load_variable(ckpt_path, name))
            vi += 1
            name, shape = weight_vars[vi]
            assert shape == tuple(layer.bias.shape)
            layer.bias.assign(tf.train.load_variable(ckpt_path, name))
            vi += 1
        elif isinstance(layer, tf.keras.layers.LayerNormalization):
            name, shape = weight_vars[vi]
            assert shape == tuple(layer.gamma.shape)
            layer.gamma.assign(tf.train.load_variable(ckpt_path, name))
            vi += 1
            name, shape = weight_vars[vi]
            assert shape == tuple(layer.beta.shape)
            layer.beta.assign(tf.train.load_variable(ckpt_path, name))
            vi += 1
    print(f"  Loaded {vi} / {2 * 4 + 2 * 2} expected weight tensors")
    return q_net


def generate_traffic(seed):
    sumo_dir = os.path.join(BASE, "SUMO_FILES")
    os.makedirs(os.path.join(sumo_dir, "traffic_files"), exist_ok=True)
    if "SUMO_HOME" not in os.environ:
        raise RuntimeError("SUMO_HOME not set")
    tools  = os.path.join(os.environ["SUMO_HOME"], "tools")
    script = os.path.join(tools, "randomTrips.py")
    net_xml = os.path.join(sumo_dir, "sim.net.xml")
    subprocess.run([
        sys.executable, script, "-n", net_xml,
        "-o", os.path.join(sumo_dir, "traffic_files", "passenger.rou.xml"),
        "-e", str(TRIP_END_STEP), "-p", "4",
        "--prefix", "pass_", "--fringe-factor", "10", "--random",
        "--seed", str(seed),
    ], check=True, capture_output=True)
    subprocess.run([
        sys.executable, script, "-n", net_xml,
        "-o", os.path.join(sumo_dir, "traffic_files", "ev.rou.xml"),
        "-e", str(TRIP_END_STEP), "-p", "150",
        "--vehicle-class", "emergency",
        "--prefix", "ev_", "--fringe-factor", "10", "--random",
        "--seed", str(seed + 1),
    ], check=True, capture_output=True)


def collect_rollout(q_net, seed, num_phases):
    generate_traffic(seed)
    sumo_dir = os.path.join(BASE, "SUMO_FILES")
    route_files = (f"{os.path.join(sumo_dir, 'traffic_files', 'ev.rou.xml')},"
                   f"{os.path.join(sumo_dir, 'traffic_files', 'passenger.rou.xml')}")
    sumo_cmd = ["sumo", "-c", SUMO_CFG, "--route-files", route_files,
                "--additional-files", ADDITIONAL, "--no-step-log", "--no-warnings",
                "--seed", str(seed)]
    traci.start(sumo_cmd)
    states, actions = [], []
    try:
        current_phase, time_in_phase = 0, 0
        traci.trafficlight.setPhase(TLS_ID, current_phase)
        traci.simulationStep()
        for _ in range(MAX_STEPS):
            det_data  = gather_detector_data()
            state_vec = get_state(det_data, current_phase, time_in_phase, num_phases)
            obs_t     = tf.constant(state_vec.reshape(1, -1), dtype=tf.float32)
            q_vals, _ = q_net(obs_t, training=False)
            action    = int(np.argmax(q_vals.numpy().flatten()))
            states.append(state_vec)
            actions.append(action)
            if action == 1 and time_in_phase >= MIN_GREEN_TIME:
                current_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(TLS_ID, current_phase)
                time_in_phase = 0
            else:
                time_in_phase += 1
            traci.simulationStep()
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
    finally:
        traci.close()
    return np.array(states), np.array(actions)


def collect_states(q_net, num_phases, n_episodes=N_EPISODES):
    all_states, all_actions = [], []
    for ep in range(n_episodes):
        seed = 42 + ep * 7
        print(f"  Episode {ep+1}/{n_episodes} (seed={seed}) ...", end=" ", flush=True)
        s, a = collect_rollout(q_net, seed, num_phases)
        all_states.append(s)
        all_actions.append(a)
        print(f"{len(s)} steps, action_1={a.mean()*100:.1f}%")
    X = np.vstack(all_states).astype(np.float32)
    y = np.concatenate(all_actions)
    print(f"\n  Total: {len(X)} states, action_1={y.mean()*100:.1f}%")
    return X, y

# ----------------------- LIME EXPLANATION FUNCTION ---------------------------
def predict_switch_prob(q_net, states):
    """Returns P(stay), P(switch) via softmax over Q-values. Shape: (N, 2)."""
    if states.ndim == 1:
        states = states[np.newaxis, :]
    states_t = tf.constant(states, dtype=tf.float32)
    q_vals, _ = q_net(states_t, training=False)
    probs = tf.nn.softmax(q_vals, axis=-1)
    return probs.numpy()


def run_lime(q_net, X_train, X_test, feature_names, num_phases):
    """Fit LIME TabularExplainer and explain test instances."""
    print(f"  Fitting LIME TabularExplainer with {len(X_train)} training samples ...")
    explainer = LimeTabularExplainer(
        training_data    = X_train,
        feature_names    = feature_names,
        class_names      = ["stay (0)", "switch (1)"],
        mode             = "classification",
        discretize_continuous = True,
        random_state     = 42,
    )
    print(f"  Explaining {len(X_test)} test instances ...")
    explanations = []
    for i in range(len(X_test)):
        exp = explainer.explain_instance(
            data_row     = X_test[i],
            predict_fn   = lambda x: predict_switch_prob(q_net, x),
            num_features = len(feature_names),
            top_labels    = 1,
            num_samples  = 1000,
        )
        explanations.append(exp)
        if (i + 1) % 10 == 0 or i == len(X_test) - 1:
            print(f"    {i+1}/{len(X_test)} done")
    return explanations, explainer


# ----------------------- VISUALISATION ---------------------------------------
def _extract_weights(explanations, feature_names):
    """Build (n_samples, n_features) array of LIME weights for label=1."""
    all_w = np.zeros((len(explanations), len(feature_names)))
    for idx, exp in enumerate(explanations):
        labels = exp.available_labels()
        if not labels:
            continue
        target = max(labels)
        for feat_name, weight in exp.as_list(target):
            for fi, fn in enumerate(feature_names):
                if fn in feat_name:
                    all_w[idx, fi] = weight
                    break
    return all_w


def plot_global_feature_weights(explanations, feature_names, out_dir):
    """Mean |LIME weight| per feature, ranked. Positive=switch, Negative=stay."""
    all_w = _extract_weights(explanations, feature_names)
    mean_wt = np.mean(all_w, axis=0)
    abs_wt  = np.abs(mean_wt)
    idx     = np.argsort(abs_wt)[::-1]

    fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.25)))
    colors = ["coral" if mean_wt[i] > 0 else "steelblue" for i in idx]
    ax.barh(range(len(idx)), abs_wt[idx], color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Mean |LIME weight|")
    ax.set_title("LIME Global Feature Importance\n(Positive=favours switch, Negative=favours stay)",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(x=0, color="gray", linewidth=0.5)
    fig.tight_layout()
    path = os.path.join(out_dir, "lime_feature_importance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_detector_heatmap(explanations, feature_names, out_dir):
    """6 detectors x 5 features grid: mean LIME weight."""
    n_det, n_feat = len(DETECTORS), 5
    all_w = _extract_weights(explanations, feature_names)
    det_ws = all_w[:, :n_det * n_feat].reshape(-1, n_det, n_feat)
    det_mean = np.mean(det_ws, axis=0)
    abs_mean = np.abs(det_mean)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(abs_mean, cmap="RdYlGn_r", aspect="auto", vmin=0)
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(["Queue", "Occ", "Speed", "Wait", "EV"], fontsize=10)
    ax.set_yticks(range(n_det))
    ax.set_yticklabels(DETECTORS, fontsize=9)
    ax.set_title("Mean |LIME weight| per Detector x Feature\n(Higher = More Influence)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Mean |LIME weight|", shrink=0.8)
    for i in range(n_det):
        for j in range(n_feat):
            val = det_mean[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=7,
                    color="white" if abs_mean[i,j] > abs_mean.max() * 0.5 else "black")
    plt.tight_layout()
    path = os.path.join(out_dir, "lime_detector_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_global_features(explanations, feature_names, out_dir):
    """Bar chart for non-detector features."""
    gs = len(DETECTORS) * 5
    all_w = _extract_weights(explanations, feature_names)
    g_wt  = all_w[:, gs:]
    g_names = feature_names[gs:]
    g_mean  = np.mean(g_wt, axis=0)
    g_abs   = np.abs(g_mean)
    idx     = np.argsort(g_abs)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["coral" if g_mean[i] > 0 else "steelblue" for i in idx]
    ax.barh(range(len(idx)), g_abs[idx], color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([g_names[i] for i in idx])
    ax.set_xlabel("Mean |LIME weight|")
    ax.set_title("Global / Non-Detector Feature Influence", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(out_dir, "lime_global_features.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_single_waterfall(explanations, X_test, feature_names, out_dir, q_net):
    """Waterfall for the test sample with highest P(switch)."""
    probs = predict_switch_prob(q_net, X_test)[:, 1]
    best  = int(np.argmax(probs))
    if probs[best] < 0.5:
        print(f"  Waterfall skipped (max P(switch)={probs[best]:.3f} < 0.5)")
        return

    exp = explanations[best]
    target = max(exp.available_labels())
    exp_list = exp.as_list(target)

    feats, weights = zip(*exp_list)
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(feats))
    colors = ["coral" if w > 0 else "steelblue" for w in weights]
    ax.barh(y_pos, weights, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("LIME weight")
    ax.set_title(f"LIME Waterfall -- Sample where P(switch)={probs[best]:.3f}",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(out_dir, "lime_waterfall_switch.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_detector_dominance(explanations, feature_names, out_dir):
    """Which detector contributes the most total |LIME weight| per sample."""
    n_det, n_feat = len(DETECTORS), 5
    all_w = _extract_weights(explanations, feature_names)
    det_abs = np.abs(all_w[:, :n_det * n_feat].reshape(-1, n_det, n_feat))
    det_total = det_abs.sum(axis=2)
    dominant  = np.argmax(det_total, axis=1)
    counts    = np.bincount(dominant, minlength=n_det).tolist()
    x_pos     = list(range(n_det))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [plt.cm.Set2(i / n_det) for i in range(n_det)]
    bars = ax.bar(x_pos, counts, color=colors, edgecolor="gray")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(DETECTORS, fontsize=8, rotation=15)
    ax.set_ylabel("Count of decisions where detector dominates")
    ax.set_title("Which Detector Dominates LIME Attention?",
                 fontsize=13, fontweight="bold")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(int(c)), ha="center", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "lime_detector_dominance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def verify_paths():
    missing = []
    for label, path in [("SUMO config", SUMO_CFG), ("Additional", ADDITIONAL),
                         ("Network", os.path.join(BASE, "SUMO_FILES", "sim.net.xml"))]:
        if not os.path.exists(path):
            missing.append(f"  {label}: {path}")
    ckpt_index = os.path.join(POLICY_PATH, "variables", "variables.index")
    if not os.path.exists(ckpt_index):
        missing.append(f"  Policy ckpt: {ckpt_index}")
    if missing:
        print("ERROR: Required files not found. BASE =", BASE)
        for m in missing:
            print(m)
        sys.exit(1)


def main():
    global NUM_PHASES, FEATURE_NAMES

    print("=" * 60)
    print("LIME Explainer for DDQN Traffic Signal Policy")
    print("=" * 60)
    print(f"BASE = {BASE}")

    verify_paths()
    NUM_PHASES    = query_num_phases()
    FEATURE_NAMES = build_feature_names(NUM_PHASES)
    state_dim     = len(FEATURE_NAMES)
    print(f"  State dimension: {state_dim}")

    q_net = load_policy_q_network(POLICY_PATH, state_dim)
    q_net.trainable = False

    dummy = np.random.randn(1, state_dim).astype(np.float32)
    q_vals, _ = q_net(dummy, training=False)
    print(f"  Q-value sanity (random input): {q_vals.numpy()}")

    print("\nCollecting rollouts ...")
    X, y = collect_states(q_net, NUM_PHASES, n_episodes=N_EPISODES)

    sample_probs = predict_switch_prob(q_net, X[:500])[:, 1]
    print(f"  P(switch) on first 500 states: mean={sample_probs.mean():.3f}  "
          f"min={sample_probs.min():.3f}  max={sample_probs.max():.3f}")

    if y.mean() < 0.005:
        print("\n  WARNING: action_1 rate < 0.5%. Policy collapsed to always-stay.")

    n_bg   = min(100, len(X) // 3)
    n_test = min(50, len(X) // 4)
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    bg_idx   = idx[:n_bg]
    test_idx = idx[n_bg:n_bg + n_test]

    action_1_idx = np.where(y == 1)[0]
    if len(action_1_idx) > 3 and n_test > 10:
        n_act1 = max(3, int(n_test * 0.15))
        act1   = np.random.choice(action_1_idx, min(n_act1, len(action_1_idx)), replace=False)
        rest   = np.setdiff1d(test_idx, act1)
        if len(rest) >= n_test - len(act1):
            test_idx = np.concatenate([act1, rest[:n_test - len(act1)]])
        test_idx = test_idx[:n_test]

    bg       = X[bg_idx]
    test     = X[test_idx]
    test_act = y[test_idx]
    print(f"\n  Background: {bg.shape}, Test: {test.shape}")
    print(f"  Test set: action_1={test_act.mean()*100:.1f}%")

    explanations, lime_explainer = run_lime(q_net, bg, test, FEATURE_NAMES, NUM_PHASES)

    print("\nGenerating visualisations ...")
    plot_global_feature_weights(explanations, FEATURE_NAMES, OUT_DIR)
    plot_detector_heatmap(explanations, FEATURE_NAMES, OUT_DIR)
    plot_global_features(explanations, FEATURE_NAMES, OUT_DIR)
    plot_single_waterfall(explanations, test, FEATURE_NAMES, OUT_DIR, q_net)
    plot_detector_dominance(explanations, FEATURE_NAMES, OUT_DIR)

    np.savez_compressed(
        os.path.join(OUT_DIR, "lime_raw_data.npz"),
        test_states      = test,
        test_actions     = test_act,
        background_states = bg,
        feature_names    = FEATURE_NAMES,
        num_phases       = np.array(NUM_PHASES),
    )
    print(f"\n  Saved raw data: {os.path.join(OUT_DIR, 'lime_raw_data.npz')}")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {OUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
