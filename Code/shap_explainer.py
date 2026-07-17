#!/usr/bin/env python3
"""
SHAP KernelExplainer for DDQN Traffic Signal Control Policy.

Strategy:
  - Loads the trained policy SavedModel (best_policy)
  - Reconstructs the Q-network with identical architecture
  - Copies weights from the SavedModel using shape-based variable matching
  - Runs SUMO evaluation rollouts to collect observation states
  - Uses SHAP KernelExplainer to explain P(switch | state)
  - Saves visualisations to shap_outputs/

State vector (37 dims):
  6 detectors x 5 features (queue, occ, speed, mean_wait, ev_count) = 30
  + time_since_switch (1)
  + phase one-hot (NUM_PHASES, queried at runtime)
  + pressure, demand (2)

Fixes applied vs original:
  1. safe_sort_key() replaces int(name.split("/")[1]) — handles non-numeric path segments
  2. Optimizer/Adam slots filtered out of checkpoint variable list
  3. NUM_PHASES derived from a live SUMO query at startup, not hardcoded
  4. FEATURE_NAMES rebuilt after the live phase count is known
  5. plot_waterfall re-uses the main explainer instead of creating a second one
  6. plot_dependence uses interaction_index="auto" instead of hardcoded second-best
  7. generate_and_collect() wrapper removed; main() calls collect_states() directly
  8. matplotlib.rcdefaults() moved to module startup; removed from every plot fn
  9. verify_paths() checks variables/variables.index instead of saved_model.pb
  10. policy_switch_prob() logs mean prob to help catch softmax saturation
"""

import os
import sys
import numpy as np
import tensorflow as tf
import traci
import matplotlib
matplotlib.use("Agg")
matplotlib.rcdefaults()          # FIX 8: single call at module level
import matplotlib.pyplot as plt
import subprocess

from tf_agents.networks import sequential
import shap

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# ─────────────────────────── CONFIG ──────────────────────────────────────────
BASE         = r"C:\Users\Dell\Desktop\Project\Implementation\Sim1"

SUMO_CFG     = os.path.join(BASE, "SUMO_FILES", "sim.sumocfg")
ADDITIONAL   = os.path.join(BASE, "SUMO_FILES", "sim.add.xml")
TLS_ID       = "Node2"
POLICY_PATH  = os.path.join(BASE, "policy_checkpoints_phase2", "best_policy")
OUT_DIR      = os.path.join(BASE, "shap_outputs")
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

# FIX 3: NUM_PHASES is populated at runtime by query_num_phases().
# Everything that used to depend on the hardcoded 4 now waits until
# after that function is called in main().
NUM_PHASES    = None   # set by query_num_phases()
FEATURE_NAMES = None   # set by build_feature_names() after NUM_PHASES is known


# ─────────────────────── RUNTIME PHASE COUNT ──────────────────────────────────
def query_num_phases() -> int:
    """
    FIX 3: Start SUMO briefly to read the real phase count for TLS_ID.
    This replaces the hardcoded NUM_PHASES = 4, which would silently produce
    a shape mismatch if the network has a different number of signal phases.
    """
    sumo_dir = os.path.join(BASE, "SUMO_FILES")
    net_xml  = os.path.join(sumo_dir, "sim.net.xml")
    # Minimal SUMO call — no route files needed, just the network
    traci.start([
        "sumo", "-n", net_xml,
        "--no-step-log", "--no-warnings",
    ])
    n = len(traci.trafficlight.getAllProgramLogics(TLS_ID)[0].phases)
    traci.close()
    print(f"  Detected {n} phases for TLS '{TLS_ID}'")
    return n


# ─────────────────────── FEATURE NAMES ───────────────────────────────────────
def build_feature_names(num_phases: int) -> list[str]:
    """
    FIX 3 (cont.): Feature names are built after the live phase count is known,
    so the phase one-hot block always has the correct length.
    """
    names = []
    for det in DETECTORS:
        for feat in ["queue", "occ", "speed", "mean_wait", "ev_count"]:
            names.append(f"{det}_{feat}")
    names.append("time_since_switch")
    for p in range(num_phases):
        names.append(f"phase_{p}")
    names.extend(["pressure", "demand"])
    return names


# ─────────────────────── HELPER: detector data ───────────────────────────────
def gather_detector_data() -> list[dict]:
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
                # FIX (back-ported from original): vehicle may leave between
                # getLastStepVehicleIDs and the per-vehicle queries
                continue

        mean_wait = float(np.mean(wait_times)) if wait_times else 0.0
        data.append(dict(
            queue=queue, occ=occ, speed=speed, veh_count=veh_count,
            mean_wait=mean_wait, ev_count=ev_count,
        ))
    return data


def get_state(det_data: list[dict],
              current_phase: int,
              time_since_switch: int,
              num_phases: int) -> np.ndarray:
    features = []
    for d in det_data:
        features.extend([d["queue"], d["occ"], d["speed"], d["mean_wait"], d["ev_count"]])
    features.append(float(time_since_switch))

    max_vals = np.array(
        [MAX_QUEUE, MAX_OCCUPANCY, MAX_SPEED, MAX_WAIT, MAX_EV] * len(DETECTORS)
        + [MAX_TIME_IN],
        dtype=np.float32,
    )
    continuous = np.array(features, dtype=np.float32)
    normalized = np.clip(continuous / (max_vals + 1e-5), 0.0, 1.0)

    total_queue = sum(d["queue"] for d in det_data)
    pressure    = np.clip(total_queue / (MAX_QUEUE * len(DETECTORS)), 0.0, 1.0)

    total_veh  = sum(d["veh_count"] for d in det_data)
    demand     = np.clip(total_veh / (10.0 * len(DETECTORS)), 0.0, 1.0)

    phase_oh = np.zeros(num_phases, dtype=np.float32)
    if 0 <= current_phase < num_phases:
        phase_oh[current_phase] = 1.0

    return np.concatenate([normalized, phase_oh, [pressure, demand]]).astype(np.float32)


# ─────────────────────── Q-NETWORK: build + weight loading ───────────────────
def build_q_network(state_dim: int) -> tf.keras.Model:
    net = sequential.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(state_dim,)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(ACTIONS),
    ])
    net(tf.zeros((1, state_dim)))   # materialise weights
    return net


def _safe_sort_key(name: str) -> int:
    """
    FIX 1: Robustly extract the numeric index from a TF checkpoint variable
    name like 'model_variables/0/kernel/.ATTRIBUTES/VARIABLE_VALUE'.
    Falls back to 999 if no numeric segment is found, so non-weight variables
    sort to the end rather than crashing.
    """
    for segment in name.split("/"):
        try:
            return int(segment)
        except ValueError:
            continue
    return 999


def load_policy_q_network(saved_model_path: str, state_dim: int) -> tf.keras.Model:
    """
    Reconstruct the Q-network and load trained weights from the policy SavedModel.

    FIX 1: Uses _safe_sort_key() instead of int(name.split("/")[1]).
    FIX 2: Filters out optimizer/Adam slots in addition to train_step /
            CHECKPOINTABLE_OBJECT_GRAPH entries.
    """
    q_net = build_q_network(state_dim)

    ckpt_path = os.path.join(saved_model_path, "variables", "variables")
    if not os.path.exists(ckpt_path + ".index"):
        raise FileNotFoundError(
            f"Checkpoint index not found at {ckpt_path}.index\n"
            f"Expected a Policy SavedModel with a variables/variables checkpoint."
        )

    print(f"  Loading weights from: {ckpt_path}")
    var_list = tf.train.list_variables(ckpt_path)

    # FIX 1 + 2: filter to model weight tensors only.
    # Exclusions:
    #   train_step         — scalar training counter
    #   CHECKPOINTABLE     — object-graph metadata
    #   optimizer / Adam   — FIX 2: optimizer momentum/velocity slots
    _EXCLUDE = ("train_step", "CHECKPOINTABLE", "optimizer", "Adam")
    weight_vars = [
        (n, tuple(s)) for n, s in var_list
        if len(s) > 0 and not any(ex in n for ex in _EXCLUDE)
    ]

    # FIX 1: sort by the first numeric path segment (the layer index)
    weight_vars.sort(key=lambda x: _safe_sort_key(x[0]))

    print(f"  Found {len(weight_vars)} weight tensors in checkpoint")
    for n, s in weight_vars:
        print(f"    {n:70s}  {s}")

    vi = 0
    for layer in q_net.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            name, shape = weight_vars[vi]
            assert shape == tuple(layer.kernel.shape), (
                f"Dense kernel @ idx {vi}: expected {tuple(layer.kernel.shape)}, got {shape}"
            )
            layer.kernel.assign(tf.train.load_variable(ckpt_path, name))
            vi += 1

            name, shape = weight_vars[vi]
            assert shape == tuple(layer.bias.shape), (
                f"Dense bias @ idx {vi}: expected {tuple(layer.bias.shape)}, got {shape}"
            )
            layer.bias.assign(tf.train.load_variable(ckpt_path, name))
            vi += 1

        elif isinstance(layer, tf.keras.layers.LayerNormalization):
            name, shape = weight_vars[vi]
            assert shape == tuple(layer.gamma.shape), (
                f"LN gamma @ idx {vi}: expected {tuple(layer.gamma.shape)}, got {shape}"
            )
            layer.gamma.assign(tf.train.load_variable(ckpt_path, name))
            vi += 1

            name, shape = weight_vars[vi]
            assert shape == tuple(layer.beta.shape), (
                f"LN beta @ idx {vi}: expected {tuple(layer.beta.shape)}, got {shape}"
            )
            layer.beta.assign(tf.train.load_variable(ckpt_path, name))
            vi += 1

    n_loaded   = vi
    n_expected = 2 * 4 + 2 * 2   # 4 Dense (k+b) + 2 LayerNorm (g+b)
    print(f"  Loaded {n_loaded} / {n_expected} expected weight tensors")
    assert n_loaded == n_expected, (
        f"Weight count mismatch: loaded {n_loaded}, expected {n_expected}"
    )

    return q_net


# ─────────────────────── TRAFFIC GENERATION ──────────────────────────────────
def generate_traffic(seed: int):
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


# ─────────────────────── STATE COLLECTION ────────────────────────────────────
def collect_rollout(q_net: tf.keras.Model, seed: int, num_phases: int) -> tuple:
    """
    Run one seeded SUMO episode under the greedy Q-network policy and
    collect (states, actions).

    num_phases is passed in (derived from query_num_phases at startup) so
    get_state() always uses the correct phase one-hot length.
    """
    generate_traffic(seed)
    sumo_dir = os.path.join(BASE, "SUMO_FILES")
    route_files = (
        f"{os.path.join(sumo_dir, 'traffic_files', 'ev.rou.xml')},"
        f"{os.path.join(sumo_dir, 'traffic_files', 'passenger.rou.xml')}"
    )
    sumo_cmd = [
        "sumo", "-c", SUMO_CFG,
        "--route-files", route_files,
        "--additional-files", ADDITIONAL,
        "--no-step-log", "--no-warnings", "--seed", str(seed),
    ]
    traci.start(sumo_cmd)

    states, actions = [], []
    try:
        current_phase = 0
        time_in_phase = 0

        traci.trafficlight.setPhase(TLS_ID, current_phase)
        traci.simulationStep()

        for _ in range(MAX_STEPS):
            det_data  = gather_detector_data()
            state_vec = get_state(det_data, current_phase, time_in_phase, num_phases)

            obs_t          = tf.constant(state_vec.reshape(1, -1), dtype=tf.float32)
            q_vals, _      = q_net(obs_t, training=False)
            q_vals_np      = q_vals.numpy().flatten()
            action         = int(np.argmax(q_vals_np))

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


def collect_states(q_net: tf.keras.Model,
                   num_phases: int,
                   n_episodes: int = N_EPISODES) -> tuple:
    """
    FIX 7: generate_and_collect() wrapper removed — this is called directly
    from main(). num_phases is threaded through to collect_rollout so the
    phase one-hot always has the right length.
    """
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


# ─────────────────────── SHAP ANALYSIS ───────────────────────────────────────
def policy_switch_prob(q_net: tf.keras.Model, states: np.ndarray) -> np.ndarray:
    """
    Returns P(action=1 | state) via softmax over Q-values. Shape: (N, 1).

    Note: softmax over Q-values is not a calibrated probability — if Q-values
    diverge (|Q| >> 10), softmax saturates and SHAP variation compresses.
    The mean prob is logged in main() to help detect saturation early.
    """
    if states.ndim == 1:
        states = states[np.newaxis, :]
    states_t = tf.constant(states, dtype=tf.float32)
    q_vals, _ = q_net(states_t, training=False)
    probs = tf.nn.softmax(q_vals, axis=-1)
    return probs[:, 1:2].numpy()   # (N, 1)


def run_shap(q_net: tf.keras.Model,
             background_states: np.ndarray,
             test_states: np.ndarray) -> tuple:
    """
    Run SHAP KernelExplainer and return (shap_values, explainer).

    The explainer object is returned so plot_waterfall can re-use it
    (FIX 5) rather than creating a second, inconsistent explainer.

    sv shape: (N_test, N_features) — one SHAP value per feature per sample.
    """
    f = lambda x: policy_switch_prob(q_net, x)

    print(f"  Initialising KernelExplainer with {len(background_states)} background samples ...")
    explainer = shap.KernelExplainer(f, background_states)

    print(f"  Computing SHAP values for {len(test_states)} samples (nsamples=150) ...")
    shap_values = explainer.shap_values(test_states, nsamples=150, silent=True)

    # KernelExplainer returns a list when the function outputs (N, 1);
    # extract the single-output array so sv is always (N, n_features).
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    print(f"  SHAP values shape: {sv.shape}  (expected ({len(test_states)}, {test_states.shape[1]}))")
    return sv, explainer


# ─────────────────────── VISUALISATION ───────────────────────────────────────
def plot_shap_summary(sv, X_test, feature_names, out_dir):
    """Beeswarm summary plot."""
    plt.close("all")
    shap.summary_plot(sv, X_test, feature_names=feature_names, show=False, max_display=20)
    fig = plt.gcf()
    fig.set_size_inches(12, max(6, len(feature_names) * 0.3))
    fig.axes[0].set_title(
        "SHAP Summary — Feature Influence on P(switch)", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "shap_summary_beeswarm.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_shap_bar(sv, X_test, feature_names, out_dir):
    """Mean |SHAP| bar chart, ranked."""
    plt.close("all")
    mean_abs = np.mean(np.abs(sv), axis=0)
    # Convert to plain Python lists before passing to matplotlib.
    # After shap.summary_plot calls np.random.seed internally, numpy fancy-
    # indexed arrays can carry unexpected metadata that makes matplotlib's
    # barh attempt float(array) and raise "only length-1 arrays" TypeError.
    idx    = np.argsort(mean_abs)[::-1].tolist()
    values = [float(mean_abs[i]) for i in idx]
    labels = [feature_names[i] for i in idx]
    y_pos  = list(range(len(idx)))

    fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.25)))
    ax.barh(y_pos, values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (Mean |SHAP|)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(out_dir, "shap_feature_importance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_detector_heatmap(sv, out_dir):
    """6 detectors x 5 features grid: mean |SHAP|."""
    plt.close("all")
    n_det  = len(DETECTORS)
    n_feat = 5
    det_sv   = sv[:, :n_det * n_feat].reshape(-1, n_det, n_feat)
    det_mean = np.mean(np.abs(det_sv), axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(det_mean, cmap="RdYlGn_r", aspect="auto", vmin=0)
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(["Queue", "Occ", "Speed", "Wait", "EV"], fontsize=10)
    ax.set_yticks(range(n_det))
    ax.set_yticklabels(DETECTORS, fontsize=9)
    ax.set_title(
        "Mean |SHAP| per Detector x Feature\n(Higher = More Influence on Switch Decision)",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Mean |SHAP|", shrink=0.8)
    for i in range(n_det):
        for j in range(n_feat):
            val = det_mean[i, j]
            ax.text(
                j, i, f"{val:.4f}", ha="center", va="center",
                fontsize=7, color="white" if val > det_mean.max() * 0.5 else "black",
            )
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_detector_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_global_features(sv, X_test, feature_names, out_dir):
    """
    Bar chart for non-detector features: time_since_switch, phase_*, pressure, demand.
    global_start = 6 detectors × 5 features = 30.
    """
    global_start = len(DETECTORS) * 5
    if sv.shape[1] <= global_start:
        return

    plt.close("all")
    g_sv    = sv[:, global_start:]
    g_names = feature_names[global_start:]
    g_mean  = np.mean(np.abs(g_sv), axis=0)
    idx     = np.argsort(g_mean).tolist()
    values  = [float(g_mean[i]) for i in idx]
    labels  = [g_names[i] for i in idx]
    y_pos   = list(range(len(idx)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(y_pos, values, color="teal")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global / Non-Detector Feature Influence", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(out_dir, "shap_global_features.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_waterfall(sv, X_test, feature_names, out_dir, explainer):
    """
    FIX 5: Waterfall plot re-uses the main explainer (passed in) instead of
    creating a second KernelExplainer with a different background.
    This ensures expected_value and SHAP values are on the same scale as
    the beeswarm / bar plots — all charts share a single reference point.

    Test-set probabilities are retrieved by calling explainer.model directly
    (KernelExplainer stores the wrapped prediction function as .model), so
    no second SUMO simulation or second explainer is needed.
    """
    plt.close("all")

    # KernelExplainer stores the wrapped prediction function as .model
    test_probs = explainer.model(X_test).flatten()

    best_idx = int(np.argmax(test_probs))
    if test_probs[best_idx] < 0.5:
        print(f"  Waterfall skipped (max P(switch) = {test_probs[best_idx]:.3f} < 0.5)")
        return

    # expected_value may be an array when output is (N,1); flatten to scalar
    expected = explainer.expected_value
    if isinstance(expected, np.ndarray):
        expected = float(expected.flatten()[0])
    elif isinstance(expected, list):
        expected = float(expected[0])

    shap.waterfall_plot(
        shap.Explanation(
            values        = sv[best_idx],
            base_values   = expected,
            data          = X_test[best_idx],
            feature_names = feature_names,
        ),
        max_display=15,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.axes[0].set_title(
        f"SHAP Waterfall — Sample where P(switch)={test_probs[best_idx]:.3f}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "shap_waterfall_switch.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dependence(sv, X_test, feature_names, out_dir):
    """
    FIX 6: Uses interaction_index='auto' so SHAP selects the feature with the
    strongest interaction, rather than always using the second-most-important
    feature regardless of actual correlation.
    """
    plt.close("all")
    mean_abs   = np.mean(np.abs(sv), axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    top3       = sorted_idx[:3]

    for fi in top3:
        name = feature_names[fi]
        shap.dependence_plot(
            fi, sv, X_test,
            feature_names     = feature_names,
            interaction_index = "auto",   # FIX 6
            show              = False,
        )
        fig = plt.gcf()
        fig.set_size_inches(7, 5)
        title_line2 = "(colour = auto-selected interaction feature)"
        fig.axes[0].set_title(
            f"SHAP Dependence: {name}\n{title_line2}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        safe_name = name.replace("|", "_").replace(" ", "_")
        path = os.path.join(out_dir, f"shap_dependence_{safe_name}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_detector_dominance(sv, out_dir):
    """Per-sample: which detector contributes the most total |SHAP| influence."""
    plt.close("all")
    n_det  = len(DETECTORS)
    n_feat = 5
    det_abs   = np.abs(sv[:, :n_det * n_feat].reshape(-1, n_det, n_feat))
    det_total = det_abs.sum(axis=2)
    dominant  = np.argmax(det_total, axis=1)
    counts    = np.bincount(dominant, minlength=n_det).tolist()   # plain list
    x_pos     = list(range(n_det))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [plt.cm.Set2(i / n_det) for i in range(n_det)]
    bars   = ax.bar(x_pos, counts, color=colors, edgecolor="gray")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(DETECTORS, fontsize=8, rotation=15)
    ax.set_ylabel("Count of decisions where detector dominates")
    ax.set_title(
        "Which Detector Dominates the Agent's Attention?",
        fontsize=13, fontweight="bold",
    )
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(int(c)), ha="center", fontsize=9,
        )
    fig.tight_layout()
    path = os.path.join(out_dir, "shap_detector_dominance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────── PATH VERIFICATION ────────────────────────────────────
def verify_paths():
    """
    FIX 9: Check variables/variables.index (the actual file load_policy_q_network
    reads) instead of saved_model.pb, which is absent in raw checkpoint exports.
    """
    missing = []
    if not os.path.exists(SUMO_CFG):
        missing.append(f"  SUMO config : {SUMO_CFG}")
    if not os.path.exists(ADDITIONAL):
        missing.append(f"  Additional  : {ADDITIONAL}")
    net_xml = os.path.join(BASE, "SUMO_FILES", "sim.net.xml")
    if not os.path.exists(net_xml):
        missing.append(f"  Network     : {net_xml}")

    # FIX 9: check the checkpoint index file, not saved_model.pb
    ckpt_index = os.path.join(POLICY_PATH, "variables", "variables.index")
    if not os.path.exists(ckpt_index):
        missing.append(
            f"  Policy ckpt : {ckpt_index}\n"
            f"    (expected variables/variables.index inside {POLICY_PATH})"
        )

    if missing:
        print("ERROR: Required files not found. BASE =", BASE)
        for m in missing:
            print(m)
        sys.exit(1)


# ─────────────────────── MAIN ─────────────────────────────────────────────────
def main():
    global NUM_PHASES, FEATURE_NAMES

    print("=" * 60)
    print("SHAP KernelExplainer for DDQN Traffic Signal Policy")
    print("=" * 60)
    print(f"BASE = {BASE}")

    # FIX 9: verify the right files before anything else
    verify_paths()

    # FIX 3: derive phase count from a live SUMO query
    NUM_PHASES    = query_num_phases()
    FEATURE_NAMES = build_feature_names(NUM_PHASES)
    state_dim     = len(FEATURE_NAMES)
    print(f"  State dimension: {state_dim}  (6 det × 5 feat + 1 + {NUM_PHASES} phase + 2)")

    # 1. Load Q-network
    q_net = load_policy_q_network(POLICY_PATH, state_dim)
    q_net.trainable = False

    # 2. Sanity-check Q-value range
    dummy    = np.random.randn(1, state_dim).astype(np.float32)
    q_vals, _ = q_net(dummy, training=False)
    print(f"  Q-value sanity (random input): {q_vals.numpy()}")

    # 3. Collect rollouts (FIX 7: no generate_and_collect wrapper)
    print("\nCollecting rollouts ...")
    X, y = collect_states(q_net, NUM_PHASES, n_episodes=N_EPISODES)

    # FIX 10 (new): log mean switch probability to detect softmax saturation
    sample_probs = policy_switch_prob(q_net, X[:500])
    print(f"  P(switch) on first 500 states: mean={sample_probs.mean():.3f}  "
          f"min={sample_probs.min():.3f}  max={sample_probs.max():.3f}")
    if sample_probs.mean() < 0.005 or sample_probs.mean() > 0.995:
        print("  WARNING: P(switch) is near-constant — softmax may be saturated. "
              "SHAP values will still be computed but may show low variance.")

    if y.mean() < 0.005:
        print("\n  WARNING: action_1 rate < 0.5%. The policy appears to have "
              "collapsed to always-stay.")

    # 4. Prepare background and stratified test sets
    n_bg   = min(100, len(X) // 3)
    n_test = min(100, len(X) // 3)
    np.random.seed(42)
    idx = np.random.permutation(len(X))

    bg_idx   = idx[:n_bg]
    test_idx = idx[n_bg:n_bg + n_test]

    action_1_idx = np.where(y == 1)[0]
    if len(action_1_idx) > 5 and n_test > 10:
        n_act1_test = max(5, int(n_test * 0.2))
        act1_test   = np.random.choice(
            action_1_idx, min(n_act1_test, len(action_1_idx)), replace=False
        )
        rest_test = np.setdiff1d(test_idx, act1_test)
        if len(rest_test) >= n_test - len(act1_test):
            test_idx = np.concatenate([act1_test, rest_test[:n_test - len(act1_test)]])
        test_idx = test_idx[:n_test]

    bg            = X[bg_idx]
    test          = X[test_idx]
    test_actions  = y[test_idx]
    print(f"\n  Background: {bg.shape}, Test: {test.shape}")
    print(f"  Test set: action_1={test_actions.mean()*100:.1f}%")

    # 5. Run SHAP — returns (sv, explainer); explainer re-used in plot_waterfall
    sv, explainer = run_shap(q_net, bg, test)

    # 6. Plots
    print("\nGenerating visualisations ...")
    plot_shap_summary(sv, test, FEATURE_NAMES, OUT_DIR)
    plot_shap_bar(sv, test, FEATURE_NAMES, OUT_DIR)
    plot_detector_heatmap(sv, OUT_DIR)
    plot_global_features(sv, test, FEATURE_NAMES, OUT_DIR)
    plot_waterfall(sv, test, FEATURE_NAMES, OUT_DIR, explainer)   # FIX 5: pass explainer
    plot_dependence(sv, test, FEATURE_NAMES, OUT_DIR)             # FIX 6: auto interaction
    plot_detector_dominance(sv, OUT_DIR)

    # 7. Save raw data
    np.savez_compressed(
        os.path.join(OUT_DIR, "shap_raw_data.npz"),
        shap_values      = sv,
        test_states      = test,
        test_actions     = test_actions,
        background_states = bg,
        feature_names    = FEATURE_NAMES,
        expected_value   = explainer.expected_value,
        num_phases       = np.array(NUM_PHASES),
    )
    print(f"\n  Saved raw data: {os.path.join(OUT_DIR, 'shap_raw_data.npz')}")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {OUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()