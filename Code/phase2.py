import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import subprocess
import traci

from tf_agents.environments   import py_environment, tf_py_environment
from tf_agents.specs          import array_spec
from tf_agents.trajectories   import time_step as ts, trajectory
from tf_agents.policies       import policy_saver
from tf_agents.networks       import sequential
from tf_agents.agents.dqn     import dqn_agent
from tf_agents.utils          import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer

# ─────────────────────────── CONFIG ──────────────────────────────────────────
SUMO_CFG   = "SUMO_FILES/sim.sumocfg"
ADDITIONAL = "SUMO_FILES/sim.add.xml"
TLS_ID     = "Node2"



DETECTORS = [
    "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2",
    "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2",
]

CHECKPOINT_DIR      = "./checkpoints_phase2"
POLICY_CKPT_DIR     = "./policy_checkpoints_phase2"
CHECKPOINT_INTERVAL = 10

ACTIONS        = 2
MIN_GREEN_TIME = 25
MAX_STEPS      = 6000
NUM_EPISODES   = 700
EARLY_STOP_PATIENCE = 50    # episodes with no improvement
EARLY_STOP_MIN_EPS  = 350   # don't stop before exploration is done

# Trip generation vs simulation end times
# ─────────────────────────────────────────
# TRIP_END_STEP controls how long randomTrips.py injects vehicles (-e flag).
# Setting it equal to MAX_STEPS means new vehicles spawn until step 5999,
# so the network never drains and early termination never fires.
#
# Fix: stop injecting new trips at 70% of MAX_STEPS (step 4200).
# The remaining 30% (1800 steps) is a DRAIN window where no new vehicles
# enter, the agent can actually clear the intersection, and early
# termination via getMinExpectedNumber() <= 0 becomes reachable.
#
# You can tune TRIP_END_RATIO:
#   • Higher (e.g. 0.80) → more traffic pressure, longer drain needed
#   • Lower  (e.g. 0.60) → lighter load, agent clears faster
TRIP_END_RATIO = 0.70
TRIP_END_STEP  = int(MAX_STEPS * TRIP_END_RATIO)   # 4200

# ─────────────────────────── HYPERPARAMETERS ─────────────────────────────────
MEMORY_SIZE     = 100_000
BATCH_SIZE      = 128


GAMMA           = 0.985

TRAIN_FREQ      = 4

# FIX 2: Raised LEARN_START from 1_000 → 8_000
# With ~42 features and stochastic traffic, the replay buffer needs richer
# diversity before training starts. 8k transitions ≈ 1-2 full episodes.
LEARN_START     = 8_000

TARGET_UPDATE   = 1_000
GRAD_CLIP       = 1.0

EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY_STEPS = 1_500_000

# Normalization maximums
MAX_QUEUE     = 50.0
MAX_OCCUPANCY = 100.0
MAX_SPEED     = 20.0
MAX_WAIT      = 300.0
MAX_EV        = 5.0
MAX_TIME_IN   = 300.0

# ─────────────────────────── CSV LOGGING ─────────────────────────────────────
csv_filename = "training_metrics.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Env_Step", "Reward", "Avg_Reward", "Loss", "Epsilon"])

# ─────────────────────── TRAFFIC GENERATION ──────────────────────────────────
def generate_traffic():
    os.makedirs("SUMO_FILES/traffic_files", exist_ok=True)
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Error: Please declare environment variable 'SUMO_HOME'")
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    randomTrips_script = os.path.join(tools, 'randomTrips.py')

    # Use TRIP_END_STEP (not MAX_STEPS) so vehicle injection stops at 70% of
    # the simulation. The remaining 30% lets the network drain naturally,
    # making early termination via getMinExpectedNumber() reachable.
    subprocess.run([
        sys.executable, randomTrips_script,
        "-n", "SUMO_FILES/sim.net.xml",
        "-o", "SUMO_FILES/traffic_files/passenger.rou.xml",
        "-e", str(TRIP_END_STEP), "-p", "4",
        "--prefix", "pass_", "--fringe-factor", "10", "--random"
    ], check=True, capture_output=True)

    subprocess.run([
        sys.executable, randomTrips_script,
        "-n", "SUMO_FILES/sim.net.xml",
        "-o", "SUMO_FILES/traffic_files/ev.rou.xml",
        "-e", str(TRIP_END_STEP), "-p", "150",
        "--vehicle-class", "emergency",
        "--prefix", "ev_", "--fringe-factor", "10", "--random"
    ], check=True, capture_output=True)


# ─────────────────────── DETECTOR DATA (single traci pass) ───────────────────
# FIX 3: Gather all detector data in ONE function, called once per step.
# Previously get_intersection_state() and compute_reward() each looped over
# DETECTORS independently — doubling traci calls and risking drift between
# the two reads within the same simulation step.

def gather_detector_data() -> list[dict]:
    """
    Returns a list of per-detector dicts with all fields needed by both
    get_intersection_state() and compute_reward().
    Only one traci loop per simulation step.
    """
    data = []
    for det in DETECTORS:
        veh_ids  = traci.lanearea.getLastStepVehicleIDs(det)
        queue    = traci.lanearea.getJamLengthVehicle(det)
        occ      = traci.lanearea.getLastStepOccupancy(det)
        speed    = traci.lanearea.getLastStepMeanSpeed(det)
        veh_count = traci.lanearea.getLastStepVehicleNumber(det)

        ev_count        = 0.0
        ev_waiting      = 0
        wait_times      = []

        for vid in veh_ids:
            wait_times.append(traci.vehicle.getWaitingTime(vid))
            if traci.vehicle.getVehicleClass(vid) == 'emergency':
                ev_count += 1.0
                if traci.vehicle.getSpeed(vid) < 1.0:
                    ev_waiting += 1

        mean_wait = float(np.mean(wait_times)) if wait_times else 0.0

        data.append({
            "queue":      queue,
            "occ":        occ,
            "speed":      speed,
            "veh_count":  veh_count,
            "mean_wait":  mean_wait,
            "ev_count":   ev_count,
            "ev_waiting": ev_waiting,
        })
    return data


# ─────────────────────── STATE LOGIC ─────────────────────────────────────────
def get_intersection_state(detector_data: list[dict],
                           current_phase: int,
                           time_since_switch: int,
                           num_phases: int) -> tuple[np.ndarray, int]:
    """
    FIX 4: Removed redundant time_until_can_switch feature.
    time_until_can_switch = max(0, MIN_GREEN_TIME - time_since_switch),
    so it carries zero additional information once time_since_switch is present.
    Keeping both inflates the state and adds a correlated feature pair.
    """
    features     = []
    ev_waiting_total = 0

    for d in detector_data:
        features.extend([
            d["queue"],
            d["occ"],
            d["speed"],
            d["mean_wait"],
            d["ev_count"],
        ])
        ev_waiting_total += d["ev_waiting"]

    # Single temporal feature: how long we have been in this phase
    features.append(float(time_since_switch))

    max_vals = np.array(
        [MAX_QUEUE, MAX_OCCUPANCY, MAX_SPEED, MAX_WAIT, MAX_EV] * len(DETECTORS)
        + [MAX_TIME_IN],
        dtype=np.float32,
    )

    continuous = np.array(features, dtype=np.float32)
    normalized = np.clip(continuous / (max_vals + 1e-5), 0.0, 1.0)

    # Summary globals
    total_queue = sum(d["queue"] for d in detector_data)
    pressure    = np.clip(total_queue / (MAX_QUEUE * len(DETECTORS)), 0.0, 1.0)

    total_veh   = sum(d["veh_count"] for d in detector_data)
    demand      = np.clip(total_veh / (10.0 * len(DETECTORS)), 0.0, 1.0)

    phase_one_hot = np.zeros(num_phases, dtype=np.float32)
    if 0 <= current_phase < num_phases:
        phase_one_hot[current_phase] = 1.0

    state = np.concatenate([normalized, phase_one_hot, [pressure, demand]]).astype(np.float32)
    return state, ev_waiting_total


# ─────────────────────── REWARD LOGIC ────────────────────────────────────────
def compute_reward(detector_data: list[dict], action: int,
                   blocked: bool, ev_waiting_count: int,
                   step_count: int, episode_ended: bool) -> float:
    """
    FIX 5: Flow metric changed from speed×(occ/100) to veh_count×speed.
    The old proxy scored empty detectors ambiguously (occ=0 zeroed the bonus
    regardless of speed). veh_count×speed approximates actual throughput:
    more vehicles moving faster = higher reward.

    FIX 6: Penalty logic corrected.
    OLD:  switch=-1.0,  blocked=-0.5
          → trying-and-being-blocked costs LESS than switching successfully,
            so the agent preferred to spam action=1 and get blocked.
    NEW:  switch=-0.5 (small friction for phase churn),
          blocked=-2.0 (clear disincentive for issuing premature switch commands).
    This hierarchy: blocked > switch > stay, which is the intended behaviour.

    FIX 7: EV weight reduced from 20→10.
    With W_EV=20, even 2 waiting EVs produce -40 before other terms, which can
    dwarf normal traffic signal during early exploration and destabilise learning.
    10 keeps EVs safety-critical while staying proportional to other penalties.

    FIX 10: Clearance bonus added.
    When the episode ends early (network drained before MAX_STEPS), the agent
    receives a bonus proportional to how many steps it saved. This directly
    rewards the behaviour we want — clearing the intersection quickly — and
    gives a strong learning signal that was completely absent before.
    """
    W_EV      = 10.0
    W_QUEUE   = 1.0
    W_WAIT    = 0.5
    W_FLOW    = 2.0
    W_SWITCH  = 0.5
    W_BLOCKED = 2.0

    queues, waits, flows = [], [], []

    for d in detector_data:
        queues.append(d["queue"])
        waits.append(d["mean_wait"])
        flows.append(d["veh_count"] * d["speed"])

    mean_queue = float(np.mean(queues))
    mean_wait  = float(np.mean(waits))
    mean_flow  = float(np.mean(flows))
    norm_flow  = mean_flow / 100.0

    ev_penalty      = -W_EV      * ev_waiting_count
    queue_penalty   = -W_QUEUE   * mean_queue
    wait_penalty    = -W_WAIT    * mean_wait
    flow_bonus      =  W_FLOW    * norm_flow
    switch_penalty  = -W_SWITCH  if (action == 1 and not blocked) else 0.0
    blocked_penalty = -W_BLOCKED if blocked else 0.0

    # Clearance bonus: reward early network drain proportional to steps saved.
    # Only fires on the terminal step and only during the drain window
    # (step > TRIP_END_STEP), where early termination is actually possible.
    # Scaled to [0, +15] so it's meaningful but doesn't dominate the reward.
    clearance_bonus = 0.0
    if episode_ended and step_count > TRIP_END_STEP:
        steps_saved     = MAX_STEPS - step_count
        clearance_bonus = 15.0 * (steps_saved / (MAX_STEPS - TRIP_END_STEP))

    raw = (ev_penalty + queue_penalty + wait_penalty + flow_bonus
           + switch_penalty + blocked_penalty + clearance_bonus)
    return float(np.clip(raw, -30.0, 30.0))


# ─────────────────────── TF-AGENTS ENVIRONMENT ───────────────────────────────
class SUMOTrafficEnv(py_environment.PyEnvironment):
    def __init__(self, sumo_cmd: list, tls_id: str = TLS_ID):
        super().__init__()
        self._sumo_cmd = sumo_cmd
        self._tls_id   = tls_id
        self._started  = False

        traci.start(sumo_cmd)
        self._num_phases = len(traci.trafficlight.getAllProgramLogics(self._tls_id)[0].phases)
        # Dummy data for spec sizing (no live detectors needed at init)
        dummy_data  = [{"queue": 0, "occ": 0, "speed": 0, "veh_count": 0,
                        "mean_wait": 0, "ev_count": 0, "ev_waiting": 0}] * len(DETECTORS)
        dummy_state, _ = get_intersection_state(dummy_data, 0, 0, self._num_phases)
        traci.close()

        self._n_features = len(dummy_state)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=ACTIONS - 1, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._n_features,), dtype=np.float32,
            minimum=0.0, maximum=1.0, name="observation",
        )

        self._episode_ended = False
        self._current_phase = 0
        self._time_in_phase = 0
        self._step_count    = 0

    def action_spec(self):    return self._action_spec
    def observation_spec(self): return self._observation_spec

    def _reset(self):
        generate_traffic()
        print("Reset")

        if not self._started:
            traci.start(self._sumo_cmd)
            self._started = True
        else:
            traci.load([
                "-c", SUMO_CFG,
                "--route-files",
                "SUMO_FILES/traffic_files/ev.rou.xml,SUMO_FILES/traffic_files/passenger.rou.xml",
                "--additional-files", ADDITIONAL,
                "--no-step-log", "--no-warnings",
            ])

        self._episode_ended = False
        self._current_phase = 0
        self._time_in_phase = 0
        self._step_count    = 0
        self._num_phases    = len(
            traci.trafficlight.getAllProgramLogics(self._tls_id)[0].phases
        )
        traci.trafficlight.setPhase(self._tls_id, self._current_phase)
        traci.simulationStep()

        det_data = gather_detector_data()
        obs, _   = get_intersection_state(det_data, self._current_phase,
                                          self._time_in_phase, self._num_phases)
        return ts.restart(obs)

    def _step(self, action: int):
        if self._episode_ended:
            return self._reset()

        blocked = False

        if action == 1 and self._time_in_phase >= MIN_GREEN_TIME:
            self._current_phase = (self._current_phase + 1) % self._num_phases
            traci.trafficlight.setPhase(self._tls_id, self._current_phase)
            self._time_in_phase = 0
        elif action == 1:
            blocked = True
            self._time_in_phase += 1
        else:
            self._time_in_phase += 1

        traci.simulationStep()
        self._step_count += 1

        # Single detector pass — shared by state and reward
        det_data = gather_detector_data()

        obs, ev_waiting = get_intersection_state(
            det_data, self._current_phase, self._time_in_phase, self._num_phases
        )

        episode_ended = (
            traci.simulation.getMinExpectedNumber() <= 0
            or self._step_count >= MAX_STEPS
        )
        reward = compute_reward(det_data, action, blocked, ev_waiting,
                                self._step_count, episode_ended)

        if episode_ended:
            self._episode_ended = True
            return ts.termination(obs, reward)

        return ts.transition(obs, reward=reward, discount=1.0)

    def close(self):
        if self._started:
            traci.close()
            self._started = False


# ──────────────────────── BUILD DDQN AGENT ───────────────────────────────────
def build_agent(tf_env, global_step, env_step_var, epsilon_var):
    n_features = tf_env.observation_spec().shape[0]

    # FIX 8: Reduced network from 256→128→64 to 128→64→32.
    # The original 256-128-64 (~50k params) is oversized for a ~42-feature
    # binary-action problem. The smaller network:
    #   • converges faster on sparse early experience
    #   • reduces overfitting risk across diverse traffic seeds
    #   • still has enough capacity for non-linear Q-value surfaces
    #
    # FIX 9: Moved LayerNorm AFTER activation (post-norm pattern).
    # Original order: Dense → LayerNorm → ReLU
    # Fixed order:    Dense → ReLU → LayerNorm
    # Post-norm is the standard for feed-forward blocks in practice and
    # prevents the normalisation from collapsing pre-activation distributions.
    # Since inputs are already clipped to [0,1], the first LayerNorm is removed
    # entirely — it was redundant given the upstream normalisation in state.
    q_net = sequential.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(n_features,)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(ACTIONS),
    ])

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=5e-4,
        decay_steps=EPS_DECAY_STEPS,
        end_learning_rate=1e-5,
        power=1,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=GRAD_CLIP)

    def epsilon_fn():
        return float(epsilon_var.numpy())

    agent = dqn_agent.DdqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network            = q_net,
        optimizer            = optimizer,
        td_errors_loss_fn    = common.element_wise_huber_loss,
        gamma                = GAMMA,
        target_update_period = TARGET_UPDATE,
        train_step_counter   = global_step,
        epsilon_greedy       = epsilon_fn,
    )
    agent.initialize()
    return agent, agent.collect_policy


# ──────────────────────── REPLAY BUFFER ──────────────────────────────────────
def build_replay_buffer(agent):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec  = agent.collect_data_spec,
        batch_size = 1,
        max_length = MEMORY_SIZE,
    )
    dataset = replay_buffer.as_dataset(
        num_parallel_calls = 1,
        sample_batch_size  = BATCH_SIZE,
        num_steps          = 2,
    ).prefetch(3)
    return replay_buffer, iter(dataset)


# ──────────────────────── TRAINING LOOP ──────────────────────────────────────
def collect_step(tf_env, policy, replay_buffer, time_step):
    action_step    = policy.action(time_step)
    next_time_step = tf_env.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)
    return next_time_step, int(action_step.action.numpy().flat[0])


def train():
    sumo_cmd = [
        "sumo", "-c", SUMO_CFG,
        "--route-files",
        "SUMO_FILES/traffic_files/ev.rou.xml,SUMO_FILES/traffic_files/passenger.rou.xml",
        "--additional-files", ADDITIONAL,
        "--no-step-log", "--no-warnings",
    ]

    generate_traffic()

    py_env = SUMOTrafficEnv(sumo_cmd)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    global_step  = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")
    env_step_var = tf.Variable(0, trainable=False, dtype=tf.int64, name="env_step")
    epsilon_var  = tf.Variable(EPS_START, trainable=False, dtype=tf.float32, name="epsilon")

    agent, collect_policy = build_agent(tf_env, global_step, env_step_var, epsilon_var)
    replay_buffer, dataset_iter = build_replay_buffer(agent)

    episode_rewards, episode_steps_hist, avg_rewards, loss_history = [], [], [], []
    q_max_history, switches_history, act0_counts, act1_counts = [], [], [], []

    checkpointer = common.Checkpointer(
        ckpt_dir      = CHECKPOINT_DIR,
        max_to_keep   = 5,
        agent         = agent,
        policy        = agent.policy,
        replay_buffer = replay_buffer,
        global_step   = global_step,
        env_step      = env_step_var,
        epsilon       = epsilon_var,
    )
    tf_policy_saver = policy_saver.PolicySaver(agent.policy, batch_size=None)

    best_avg_reward = -float('inf')
    checkpointer.initialize_or_restore()

    restored_env_step = int(env_step_var.numpy())
    if restored_env_step > 0:
        new_eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * restored_env_step / EPS_DECAY_STEPS)
        epsilon_var.assign(new_eps)
        print(f"Resumed: global_step={int(global_step.numpy())}, env_step={restored_env_step}, ε={new_eps:.4f}")
    else:
        print("Starting fresh training run.")

    for episode in range(NUM_EPISODES):
        time_step     = tf_env.reset()
        total_reward  = 0.0
        step          = 0
        ep_losses     = []
        action_counts = {0: 0, 1: 0}
        switches      = 0

        while not time_step.is_last():
            time_step, action = collect_step(tf_env, collect_policy, replay_buffer, time_step)
            total_reward += float(time_step.reward)
            action_counts[action] += 1
            step += 1

            env_step_var.assign_add(1)
            current_env_step = int(env_step_var.numpy())
            current_eps = max(
                EPS_END,
                EPS_START - (EPS_START - EPS_END) * current_env_step / EPS_DECAY_STEPS,
            )
            epsilon_var.assign(current_eps)

            if current_env_step >= LEARN_START and current_env_step % TRAIN_FREQ == 0:
                experience, _ = next(dataset_iter)
                loss_info = agent.train(experience)

                if current_env_step % 50 == 0:
                    q_values, _ = agent._q_network(time_step.observation)
                    q_max_history.append(float(np.max(q_values.numpy())))

                if current_env_step % 5000 == 0:
                    dummy_data  = [{"queue": 0, "occ": 0, "speed": 0, "veh_count": 0,
                                    "mean_wait": 0, "ev_count": 0, "ev_waiting": 0}] * len(DETECTORS)
                    dummy_state, _ = get_intersection_state(dummy_data, 0, 0, py_env._num_phases)
                    dummy_obs = tf.expand_dims(tf.constant(dummy_state), 0)
                    q_vals, _ = agent._q_network(dummy_obs)
                    q_arr = q_vals.numpy()
                    q_max = float(np.max(q_arr))
                    q_min = float(np.min(q_arr))
                    print(f"  Q-values: {q_arr}  diff={abs(q_arr[0,0]-q_arr[0,1]):.4f}")
                    print(f"  Q-range: [{q_min:.2f}, {q_max:.2f}]")
                    if abs(q_max) > 500 or np.isnan(q_max):
                        print("  WARNING: Q-values diverging.")

                ep_losses.append(float(loss_info.loss))

            if action == 1 and py_env._time_in_phase == 0:
                switches += 1

        episode_rewards.append(total_reward)
        episode_steps_hist.append(step)
        avg_reward = float(np.mean(episode_rewards[-10:]))
        avg_rewards.append(avg_reward)
        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        loss_history.append(avg_loss)
        switches_history.append(switches)
        act0_counts.append(action_counts[0])
        act1_counts.append(action_counts[1])

        current_eps = float(epsilon_var.numpy())

        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            checkpointer.save(global_step)
            tf_policy_saver.save(f"{POLICY_CKPT_DIR}/policy_step_{int(global_step.numpy())}")
            print(f"  [ckpt] Saved at episode {episode+1}")

        if avg_reward > best_avg_reward:
            best_avg_reward  = avg_reward
            no_improve_count = 0
            tf_policy_saver.save(f"{POLICY_CKPT_DIR}/best_policy")
            print("  [BEST] Policy updated")
        else:
            no_improve_count += 1

        exploration_done = int(env_step_var.numpy()) >= EPS_DECAY_STEPS
        if (exploration_done
                and episode + 1 >= EARLY_STOP_MIN_EPS
                and no_improve_count >= EARLY_STOP_PATIENCE):
            print(f"Early stop at episode {episode+1}: "
                f"no improvement for {EARLY_STOP_PATIENCE} episodes.")
            break

        print(
            f"Ep {episode+1:>3}/{NUM_EPISODES} | "
            f"Steps: {step:>5} | "
            f"Switches: {switches:>3} (act0={action_counts[0]}, act1={action_counts[1]}) | "
            f"Reward: {total_reward:>8.2f} | "
            f"AvgR(10): {avg_reward:>8.2f} | "
            f"Loss: {avg_loss:.4f} | "
            f"ε: {current_eps:.4f} | "
            f"env_step: {int(env_step_var.numpy())}"
        )

        with open(csv_filename, 'a', newline='') as file:
            csv.writer(file).writerow([
                episode, int(env_step_var.numpy()),
                total_reward, avg_reward, avg_loss, current_eps,
            ])

    tf_env.close()
    checkpointer.save(global_step)
    tf_policy_saver.save("final_saved_policy_phase2")
    print("Policy saved to ./final_saved_policy_phase2/")

    _plot(episode_rewards, avg_rewards, loss_history, episode_steps_hist,
          q_max_history, switches_history, act0_counts, act1_counts)


def _plot(rewards, avg_rewards, losses, steps, q_max, switches, act0, act1):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("DDQN Traffic Control – Training Analytics", fontsize=16, fontweight='bold')

    axes[0, 0].plot(rewards, label="Episode Reward", alpha=0.3, color='royalblue')
    axes[0, 0].plot(avg_rewards, label="Moving Avg (10)", linewidth=2, color='blue')
    axes[0, 0].set(title="Cumulative Reward", xlabel="Episode", ylabel="Score")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, color="orange", alpha=0.8)
    axes[0, 1].set(title="Episode Duration", xlabel="Episode", ylabel="Steps to Clear")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(losses, color="red", label="Huber Loss", alpha=0.8)
    axes[1, 0].set(title="Model Loss (TD Error)", xlabel="Episode", ylabel="Loss")
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(q_max, color='purple', linewidth=1.5)
    axes[1, 1].set(title="Max Q-Value (Agent Confidence)", xlabel="Steps (sampled)", ylabel="Q-Value")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(switches, color='green', marker='o', markersize=2, linestyle='')
    axes[2, 0].set(title="Total Signal Switches", xlabel="Episode", ylabel="Count")
    axes[2, 0].grid(True, alpha=0.3)

    episodes = range(len(act0))
    axes[2, 1].bar(episodes, act0, label='Stay (0)', color='skyblue', alpha=0.7)
    axes[2, 1].bar(episodes, act1, bottom=act0, label='Switch (1)', color='coral', alpha=0.7)
    axes[2, 1].set(title="Action Distribution", xlabel="Episode", ylabel="Total Actions")
    axes[2, 1].legend(loc='upper right')
    axes[2, 1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("training_report.png", dpi=200)
    plt.show()
    print("Dashboard saved to training_report.png")


if __name__ == "__main__":
    train()