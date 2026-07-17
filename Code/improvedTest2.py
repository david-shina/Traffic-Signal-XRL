#!/usr/bin/env python3
"""
Comprehensive evaluation of DDQN vs Actuated vs Fixed-Time traffic signal control.

Tests three traffic scenarios:
  1. Balanced   — symmetric demand (mirrors training)
  2. Asymmetric — one corridor 3× heavier than the other
  3. EV-Heavy   — frequent emergency vehicles injected mid-run

Seeding guarantees:
  - The same .rou.xml files are generated for all controllers per episode.
  - SUMO internal RNG is fixed per episode for full reproducibility.

Outputs:
  - Console summary table
  - evaluate_results.csv   (raw per-episode data)
  - evaluate_summary.csv   (aggregated means ± std per scenario)
  - evaluate_report.png    (12-panel comparison dashboard)
"""

import os
import sys
import csv
import hashlib
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import subprocess
import traci
import xml.etree.ElementTree as ET

from tf_agents.trajectories import time_step as ts

# ─────────────────────────── CONFIG ──────────────────────────────────────────
SUMO_CFG   = "SUMO_FILES/sim.sumocfg"
ADDITIONAL = "SUMO_FILES/sim.add.xml"
TLS_ID     = "Node2"

DETECTORS = [
    "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2",
    "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2",
]

# Must match training values
MAX_STEPS      = 6000
TRIP_END_RATIO = 0.70
TRIP_END_STEP  = int(MAX_STEPS * TRIP_END_RATIO)
MIN_GREEN_TIME = 25
ACTIONS        = 2
#
MAX_QUEUE     = 50.0
MAX_OCCUPANCY = 100.0
MAX_SPEED     = 20.0
MAX_WAIT      = 300.0
MAX_EV        = 5.0
MAX_TIME_IN   = 300.0

# ─────────────────────────── EVALUATION SETTINGS ─────────────────────────────
EPISODES_PER_SCENARIO = 10

# FIXED-TIME CONTROLLER: duration (seconds) for each phase index
# Phase 0: 45 s | Phase 1: 5 s | Phase 2: 45 s | Phase 3: 5 s
FIXED_PHASE_DURATIONS = [45, 5, 45, 5]

# ACTUATED CONTROLLER SETTINGS
# Green phases can extend up to this limit if demand is present.
# Transition phases (1, 3) keep their fixed short duration.
ACTUATED_MAX_GREEN = [60, 5, 60, 5]
# Map each green phase to the detector indices that serve it.
# Phase 0 (EB/WB green) → EB detectors 0-2
# Phase 2 (NB/SB green) → SB detectors 3-5
ACTUATED_PHASE_DETECTORS = {
    0: [0, 1, 2],
    2: [3, 4, 5],
}

POLICY_PATH           = "./policy_checkpoints_phase2/best_policy"

# Asymmetric scenario: source edge IDs for the heavy corridor.
# Example: HEAVY_CORRIDOR_EDGES = ["1to2_0", "1to2_1"]
# Leave empty to fall back to elevated uniform demand.
HEAVY_CORRIDOR_EDGES = []

# Suppress TF spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ─────────────────────────── SEEDING ───────────────────────────────────────
def make_seed(scenario: str, episode: int) -> int:
    """
    Deterministic 31-bit seed derived from scenario name + episode number.
    Guarantees the same traffic files for all controllers.
    """
    digest = hashlib.md5(f"{scenario}:{episode}".encode()).hexdigest()
    return int(digest, 16) % (2**31)


# ─────────────────────── DETECTOR DATA (safe version) ──────────────────────
def gather_detector_data() -> list[dict]:
    """
    Returns per-detector metrics.  Vehicle-level queries are wrapped in
    try/except because getLastStepVehicleIDs() may contain vehicles that
    arrived or teleported out between the simulation step and this call.
    """
    data = []
    for det in DETECTORS:
        veh_ids   = traci.lanearea.getLastStepVehicleIDs(det)
        queue     = traci.lanearea.getJamLengthVehicle(det)
        occ       = traci.lanearea.getLastStepOccupancy(det)
        speed     = traci.lanearea.getLastStepMeanSpeed(det)
        veh_count = traci.lanearea.getLastStepVehicleNumber(det)

        ev_count   = 0.0
        ev_waiting = 0
        wait_times = []

        for vid in veh_ids:
            try:
                wait_times.append(traci.vehicle.getWaitingTime(vid))
                if traci.vehicle.getVehicleClass(vid) == 'emergency':
                    ev_count += 1.0
                    if traci.vehicle.getSpeed(vid) < 1.0:
                        ev_waiting += 1
            except traci.TraCIException:
                # Vehicle left the network between the step and this query
                continue

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


# ─────────────────────── STATE LOGIC (from phase2.py) ──────────────────────────
def get_intersection_state(detector_data: list[dict],
                           current_phase: int,
                           time_since_switch: int,
                           num_phases: int) -> tuple[np.ndarray, int]:
    features         = []
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

    features.append(float(time_since_switch))

    max_vals = np.array(
        [MAX_QUEUE, MAX_OCCUPANCY, MAX_SPEED, MAX_WAIT, MAX_EV] * len(DETECTORS)
        + [MAX_TIME_IN],
        dtype=np.float32,
    )

    continuous = np.array(features, dtype=np.float32)
    normalized = np.clip(continuous / (max_vals + 1e-5), 0.0, 1.0)

    total_queue = sum(d["queue"] for d in detector_data)
    pressure    = np.clip(total_queue / (MAX_QUEUE * len(DETECTORS)), 0.0, 1.0)

    total_veh = sum(d["veh_count"] for d in detector_data)
    demand    = np.clip(total_veh / (10.0 * len(DETECTORS)), 0.0, 1.0)

    phase_one_hot = np.zeros(num_phases, dtype=np.float32)
    if 0 <= current_phase < num_phases:
        phase_one_hot[current_phase] = 1.0

    state = np.concatenate([normalized, phase_one_hot, [pressure, demand]]).astype(np.float32)
    return state, ev_waiting_total


# ─────────────────────── REWARD LOGIC (from phase2.py) ───────────────────────
def compute_reward(detector_data: list[dict], action: int,
                   blocked: bool, ev_waiting_count: int,
                   step_count: int, episode_ended: bool) -> float:
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

    clearance_bonus = 0.0
    if episode_ended and step_count > TRIP_END_STEP:
        steps_saved     = MAX_STEPS - step_count
        clearance_bonus = 15.0 * (steps_saved / (MAX_STEPS - TRIP_END_STEP))

    raw = (ev_penalty + queue_penalty + wait_penalty + flow_bonus
           + switch_penalty + blocked_penalty + clearance_bonus)
    return float(np.clip(raw, -30.0, 30.0))


# ─────────────────────── TRAFFIC GENERATION HELPERS ──────────────────────────
def _randomTrips(output: str, end: int, period: float, prefix: str,
                 vclass: Optional[str] = None, begin: int = 0, seed: Optional[int] = None):
    if 'SUMO_HOME' not in os.environ:
        raise RuntimeError("SUMO_HOME not declared")
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    script = os.path.join(tools, 'randomTrips.py')

    cmd = [
        sys.executable, script,
        "-n", "SUMO_FILES/sim.net.xml",
        "-o", output,
        "-e", str(end), "-p", str(period),
        "--prefix", prefix, "--fringe-factor", "10", "--random"
    ]
    if vclass:
        cmd.extend(["--vehicle-class", vclass])
    if begin > 0:
        cmd.extend(["--begin", str(begin)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    subprocess.run(cmd, check=True, capture_output=True)


def _merge_route_files(inputs: list[str], output: str):
    """Merge SUMO route files (vehicle or trip elements) into one file."""
    root = ET.Element('routes')
    for inp in inputs:
        if not os.path.exists(inp):
            continue
        tree = ET.parse(inp)
        for child in tree.getroot():
            root.append(child)
    ET.ElementTree(root).write(output, encoding='UTF-8', xml_declaration=True)


def _filter_routes_by_first_edge(input_file: str, output_file: str, allowed_edges: list[str]):
    """Keep only vehicles/trips whose first route edge is in allowed_edges."""
    tree = ET.parse(input_file)
    root = tree.getroot()
    new_root = ET.Element('routes')

    for vehicle in root.findall('vehicle'):
        route = vehicle.find('route')
        if route is not None:
            edges = route.get('edges', '').split()
            if edges and edges[0] in allowed_edges:
                new_root.append(vehicle)
        else:
            new_root.append(vehicle)

    for trip in root.findall('trip'):
        if trip.get('from') in allowed_edges:
            new_root.append(trip)

    ET.ElementTree(new_root).write(output_file, encoding='UTF-8', xml_declaration=True)


# ─────────────────────── SCENARIO GENERATORS (seeded) ─────────────────────────
def generate_traffic_balanced(seed: int):
    os.makedirs("SUMO_FILES/traffic_files", exist_ok=True)
    _randomTrips("SUMO_FILES/traffic_files/passenger.rou.xml",
                 TRIP_END_STEP, 4, "pass_", seed=seed)
    _randomTrips("SUMO_FILES/traffic_files/ev.rou.xml",
                 TRIP_END_STEP, 150, "ev_", vclass="emergency", seed=seed + 1)


def generate_traffic_asymmetric(seed: int):
    os.makedirs("SUMO_FILES/traffic_files", exist_ok=True)

    # Light base traffic (1/3 density of balanced)
    _randomTrips("SUMO_FILES/traffic_files/passenger_light.rou.xml",
                 TRIP_END_STEP, 12, "pass_l_", seed=seed)

    if HEAVY_CORRIDOR_EDGES:
        _randomTrips("SUMO_FILES/traffic_files/passenger_heavy_raw.rou.xml",
                     TRIP_END_STEP, 4, "pass_h_", seed=seed + 1)
        _filter_routes_by_first_edge(
            "SUMO_FILES/traffic_files/passenger_heavy_raw.rou.xml",
            "SUMO_FILES/traffic_files/passenger_heavy.rou.xml",
            HEAVY_CORRIDOR_EDGES
        )
        if not os.path.exists("SUMO_FILES/traffic_files/passenger_heavy.rou.xml") or \
           os.path.getsize("SUMO_FILES/traffic_files/passenger_heavy.rou.xml") < 100:
            print("[WARN] Edge-filtered heavy traffic is empty; falling back to uniform.")
            _randomTrips("SUMO_FILES/traffic_files/passenger_heavy.rou.xml",
                         TRIP_END_STEP, 4, "pass_h_", seed=seed + 1)
    else:
        print("[WARN] HEAVY_CORRIDOR_EDGES is empty; asymmetric scenario uses elevated uniform demand.")
        _randomTrips("SUMO_FILES/traffic_files/passenger_heavy.rou.xml",
                     TRIP_END_STEP, 4, "pass_h_", seed=seed + 1)

    _merge_route_files([
        "SUMO_FILES/traffic_files/passenger_light.rou.xml",
        "SUMO_FILES/traffic_files/passenger_heavy.rou.xml"
    ], "SUMO_FILES/traffic_files/passenger.rou.xml")

    _randomTrips("SUMO_FILES/traffic_files/ev.rou.xml",
                 TRIP_END_STEP, 150, "ev_", vclass="emergency", seed=seed + 2)


def generate_traffic_ev_heavy(seed: int):
    os.makedirs("SUMO_FILES/traffic_files", exist_ok=True)
    _randomTrips("SUMO_FILES/traffic_files/passenger.rou.xml",
                 TRIP_END_STEP, 4, "pass_", seed=seed)
    ev_begin = int(TRIP_END_STEP * 0.33)
    _randomTrips("SUMO_FILES/traffic_files/ev.rou.xml",
                 TRIP_END_STEP, 50, "ev_", vclass="emergency", begin=ev_begin, seed=seed + 1)


def generate_traffic(scenario: str, seed: int):
    """Dispatcher: generate route files for a given scenario with a fixed seed."""
    if scenario == "balanced":
        generate_traffic_balanced(seed)
    elif scenario == "asymmetric":
        generate_traffic_asymmetric(seed)
    elif scenario == "ev_heavy":
        generate_traffic_ev_heavy(seed)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# ─────────────────────── EPISODE RUNNER (no generation) ────────────────────
def run_episode(controller: str, scenario: str, policy=None, seed: int = 0) -> dict:
    """
    Run one evaluation episode using pre-generated route files.
    The same seed is passed to SUMO for reproducible internal RNG.
    """
    sumo_cmd = [
        "sumo-gui", "-c", SUMO_CFG,
        "--route-files",
        "SUMO_FILES/traffic_files/ev.rou.xml,SUMO_FILES/traffic_files/passenger.rou.xml",
        "--additional-files", ADDITIONAL,
        "--no-step-log", "--no-warnings",
        "--seed", str(seed),
    ]

    traci.start(sumo_cmd)
    try:
        num_phases = len(traci.trafficlight.getAllProgramLogics(TLS_ID)[0].phases)
        current_phase = 0
        time_in_phase = 0
        step_count    = 0
        total_reward  = 0.0
        switches      = 0

        # Vehicle registry: vid -> {"departed": int, "class": str, "wait": float}
        vehicle_registry = {}
        arrived_stats    = []   # (wait_time, travel_time, is_ev)
        queue_totals     = []   # sum of queue across all detectors per step

        # ── initial step (mirrors SUMOTrafficEnv._reset) ──
        traci.trafficlight.setPhase(TLS_ID, current_phase)
        traci.simulationStep()
        step_count += 1

        # Process any departures / arrivals from the very first step
        for vid in traci.simulation.getDepartedIDList():
            try:
                vclass = traci.vehicle.getVehicleClass(vid)
            except traci.TraCIException:
                vclass = "passenger"
            vehicle_registry[vid] = {"departed": step_count, "class": vclass, "wait": 0.0}

        for vid in traci.simulation.getArrivedIDList():
            if vid not in vehicle_registry:
                continue
            info = vehicle_registry[vid]
            arrived_stats.append((info["wait"], step_count - info["departed"], info["class"] == "emergency"))
            del vehicle_registry[vid]

        # ── main loop ──
        while step_count < MAX_STEPS:
            # 1. Snapshot waiting times for all vehicles *before* they might arrive
            current_ids = set(traci.vehicle.getIDList())
            for vid in list(vehicle_registry.keys()):
                if vid not in current_ids:
                    continue
                try:
                    vehicle_registry[vid]["wait"] = traci.vehicle.getAccumulatedWaitingTime(vid)
                except (traci.TraCIException, AttributeError):
                    try:
                        vehicle_registry[vid]["wait"] = traci.vehicle.getWaitingTime(vid)
                    except traci.TraCIException:
                        pass

            # 2. Observe state
            det_data = gather_detector_data()

            # 3. Choose action
            if controller == "fixed":
                # ── Phase-specific fixed durations ──
                phase_limit = FIXED_PHASE_DURATIONS[current_phase % len(FIXED_PHASE_DURATIONS)]
                action = 1 if time_in_phase >= phase_limit else 0

            elif controller == "actuated":
                # ── Gap-seeking actuated control ──
                # Transition phases (yellow/all-red) follow fixed duration
                if current_phase in [1, 3]:
                    action = 1 if time_in_phase >= FIXED_PHASE_DURATIONS[current_phase] else 0
                else:
                    # Green phases: extend if demand present, up to max green
                    min_green = FIXED_PHASE_DURATIONS[current_phase]
                    max_green = ACTUATED_MAX_GREEN[current_phase]

                    if time_in_phase < min_green:
                        action = 0  # Hold minimum green
                    elif time_in_phase >= max_green:
                        action = 1  # Max out — must switch
                    else:
                        det_indices = ACTUATED_PHASE_DETECTORS.get(current_phase, [])
                        demand = sum(det_data[i]["veh_count"] for i in det_indices if i < len(det_data))
                        ev_wait = sum(det_data[i]["ev_waiting"] for i in det_indices if i < len(det_data))

                        # Extend if vehicles or EVs are present on current approach
                        if demand > 0 or ev_wait > 0:
                            action = 0
                        else:
                            action = 1  # Gap out

            else:   # ddqn
                obs, _ = get_intersection_state(det_data, current_phase, time_in_phase, num_phases)
                obs_t = tf.expand_dims(tf.constant(obs, dtype=tf.float32), 0)

                # Build a proper TimeStep namedtuple for SavedModel
                time_step = ts.TimeStep(
                    step_type=tf.constant([ts.StepType.MID], dtype=tf.int32),
                    reward=tf.constant([0.0], dtype=tf.float32),
                    discount=tf.constant([1.0], dtype=tf.float32),
                    observation=obs_t,
                )

                action_step = policy.action(time_step)

                if hasattr(action_step, 'action'):
                    action = int(action_step.action.numpy().flat[0])
                else:
                    action = int(action_step['action'].numpy().flat[0])

            # 4. Execute action
            blocked = False
            if action == 1 and time_in_phase >= MIN_GREEN_TIME:
                current_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(TLS_ID, current_phase)
                time_in_phase = 0
                switches += 1
            elif action == 1:
                blocked = True
                time_in_phase += 1
            else:
                time_in_phase += 1

            # 5. Step simulation
            traci.simulationStep()
            step_count += 1

            # 6. Collect detector metrics
            det_data = gather_detector_data()
            queue_totals.append(sum(d["queue"] for d in det_data))

            # 7. Handle departures
            for vid in traci.simulation.getDepartedIDList():
                try:
                    vclass = traci.vehicle.getVehicleClass(vid)
                except traci.TraCIException:
                    vclass = "passenger"
                vehicle_registry[vid] = {"departed": step_count, "class": vclass, "wait": 0.0}

            # 8. Handle arrivals (use stored data — NEVER query arrived vehicles)
            for vid in traci.simulation.getArrivedIDList():
                if vid not in vehicle_registry:
                    continue
                info = vehicle_registry[vid]
                travel_time = step_count - info["departed"]
                is_ev = (info["class"] == "emergency")
                arrived_stats.append((info["wait"], travel_time, is_ev))
                del vehicle_registry[vid]

            # 9. Episode termination & reward
            drained = traci.simulation.getMinExpectedNumber() <= 0
            episode_ended = drained or step_count >= MAX_STEPS

            if controller == "ddqn":
                _, ev_waiting = get_intersection_state(
                    det_data, current_phase, time_in_phase, num_phases
                )
                reward = compute_reward(det_data, action, blocked, ev_waiting,
                                        step_count, episode_ended)
                total_reward += reward

            if episode_ended:
                break

        # ── compute aggregates ──
        n_arrived = len(arrived_stats)
        avg_wait   = np.mean([s[0] for s in arrived_stats]) if arrived_stats else 0.0
        avg_travel = np.mean([s[1] for s in arrived_stats]) if arrived_stats else 0.0
        thruput_rate = n_arrived / step_count if step_count > 0 else 0.0
        avg_queue = np.mean(queue_totals) / len(DETECTORS) if queue_totals else 0.0
        max_queue = max(queue_totals) if queue_totals else 0.0

        ev_stats = [s for s in arrived_stats if s[2]]
        ev_avg_wait = np.mean([s[0] for s in ev_stats]) if ev_stats else 0.0
        ev_max_wait = max([s[0] for s in ev_stats]) if ev_stats else 0.0
        ev_clearance = (sum(1 for s in ev_stats if s[0] < 10.0) / len(ev_stats) * 100
                        if ev_stats else 0.0)

        return {
            "controller":        controller,
            "scenario":          scenario,
            "total_reward":      total_reward if controller == "ddqn" else np.nan,
            "avg_wait_time":     avg_wait,
            "avg_travel_time":   avg_travel,
            "total_throughput":  n_arrived,
            "throughput_rate":   thruput_rate,
            "avg_queue_length":  avg_queue,
            "max_queue_length":  max_queue,
            "ev_avg_wait":       ev_avg_wait,
            "ev_max_wait":       ev_max_wait,
            "ev_clearance_rate": ev_clearance,
            "episode_steps":     step_count,
            "early_termination": step_count < MAX_STEPS,
            "phase_switches":    switches if controller in ["ddqn", "actuated"] else np.nan,
        }

    finally:
        traci.close()


# ─────────────────────── RESULTS I/O ─────────────────────────────────────────
def save_results_csv(results: list[dict]):
    keys = [
        "episode", "scenario", "controller", "seed",
        "total_reward", "avg_wait_time", "avg_travel_time",
        "total_throughput", "throughput_rate", "avg_queue_length",
        "max_queue_length", "ev_avg_wait", "ev_max_wait",
        "ev_clearance_rate", "episode_steps", "early_termination",
        "phase_switches"
    ]
    with open("evaluate_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in keys})
    print("\nSaved raw per-episode data -> evaluate_results.csv")


def compute_summary(results: list[dict]) -> list[dict]:
    summary = []
    scenarios   = sorted({r["scenario"] for r in results})
    controllers = sorted({r["controller"] for r in results})

    metric_keys = [
        "avg_wait_time", "avg_travel_time", "total_throughput",
        "throughput_rate", "avg_queue_length", "max_queue_length",
        "ev_avg_wait", "ev_max_wait", "ev_clearance_rate",
        "episode_steps", "total_reward", "phase_switches"
    ]

    for sc in scenarios:
        for ctrl in controllers:
            rows = [r for r in results if r["scenario"] == sc and r["controller"] == ctrl]
            if not rows:
                continue
            entry = {"scenario": sc, "controller": ctrl, "n_episodes": len(rows)}
            for mk in metric_keys:
                vals = [r[mk] for r in rows if not np.isnan(r.get(mk, np.nan))]
                entry[f"{mk}_mean"] = np.mean(vals) if vals else np.nan
                entry[f"{mk}_std"]  = np.std(vals)  if vals else np.nan
            summary.append(entry)
    return summary


def save_summary_csv(summary: list[dict]):
    if not summary:
        return
    keys = list(summary[0].keys())
    with open("evaluate_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for s in summary:
            writer.writerow(s)
    print("Saved aggregated summary -> evaluate_summary.csv")


def print_console_summary(summary: list[dict]):
    print("\n" + "="*120)
    print("EVALUATION SUMMARY  (mean ± std)")
    print("="*120)

    for sc in sorted({s["scenario"] for s in summary}):
        print(f"\n--- Scenario: {sc.upper()} ---")
        hdr = (f"{'Ctrl':<<10} {'AvgWait':>10} {'AvgTrav':>10} {'Thru':>8} "
               f"{'ThrRate':>8} {'AvgQ':>8} {'MaxQ':>8} {'EVwait':>8} "
               f"{'EVmax':>8} {'EVclr%':>8} {'Steps':>7} {'Reward':>9} {'Sw':>5}")
        print(hdr)
        print("-" * len(hdr))

        for entry in summary:
            if entry["scenario"] != sc:
                continue
            c = entry["controller"]
            def fmt_mean_std(mk):
                m = entry.get(f"{mk}_mean", np.nan)
                s = entry.get(f"{mk}_std", np.nan)
                if np.isnan(m):
                    return "   —   "
                return f"{m:7.1f}±{s:5.1f}"

            line = (f"{c:<10} "
                    f"{fmt_mean_std('avg_wait_time'):>10} "
                    f"{fmt_mean_std('avg_travel_time'):>10} "
                    f"{fmt_mean_std('total_throughput'):>8} "
                    f"{fmt_mean_std('throughput_rate'):>8} "
                    f"{fmt_mean_std('avg_queue_length'):>8} "
                    f"{fmt_mean_std('max_queue_length'):>8} "
                    f"{fmt_mean_std('ev_avg_wait'):>8} "
                    f"{fmt_mean_std('ev_max_wait'):>8} "
                    f"{fmt_mean_std('ev_clearance_rate'):>8} "
                    f"{fmt_mean_std('episode_steps'):>7} "
                    f"{fmt_mean_std('total_reward'):>9} "
                    f"{fmt_mean_std('phase_switches'):>5}")
            print(line)
    print("="*120)


# ─────────────────────── 12-PANEL DASHBOARD ──────────────────────────────────
def plot_dashboard(results: list[dict], summary: list[dict]):
    scenarios   = ["balanced", "asymmetric", "ev_heavy"]
    controllers = ["fixed", "actuated", "ddqn"]
    colors      = {"fixed": "steelblue", "actuated": "seagreen", "ddqn": "coral"}

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle("DDQN vs Actuated vs Fixed-Time Traffic Signal Control — Evaluation Dashboard",
                 fontsize=16, fontweight='bold')

    def grouped_bar(ax, metric: str, ylabel: str, title: str):
        x = np.arange(len(scenarios))
        width = 0.25
        for i, ctrl in enumerate(controllers):
            means, stds = [], []
            for sc in scenarios:
                row = next((s for s in summary
                            if s["scenario"] == sc and s["controller"] == ctrl), None)
                means.append(row.get(f"{metric}_mean", 0) if row else 0)
                stds.append(row.get(f"{metric}_std", 0) if row else 0)
            offset = (i - 1) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=ctrl.upper(), color=colors[ctrl], alpha=0.8, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([s.title() for s in scenarios])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    # Row 0
    grouped_bar(axes[0, 0], "avg_wait_time",     "s",  "Avg Waiting Time")
    grouped_bar(axes[0, 1], "avg_travel_time",   "s",  "Avg Travel Time")
    grouped_bar(axes[0, 2], "total_throughput",  "veh","Total Throughput")
    grouped_bar(axes[0, 3], "throughput_rate",   "veh/step", "Throughput Rate")

    # Row 1
    grouped_bar(axes[1, 0], "avg_queue_length",  "veh/det", "Avg Queue (per detector)")
    grouped_bar(axes[1, 1], "max_queue_length",  "veh",     "Max Queue (total)")
    grouped_bar(axes[1, 2], "ev_avg_wait",       "s",       "EV Avg Wait")
    grouped_bar(axes[1, 3], "ev_max_wait",       "s",       "EV Max Wait")

    # Row 2
    grouped_bar(axes[2, 0], "ev_clearance_rate", "%",       "EV Clearance <10 s")
    grouped_bar(axes[2, 1], "episode_steps",     "steps",   "Episode Steps")

    # DDQN-only reward panel
    ax = axes[2, 2]
    ddqn_rows = [s for s in summary if s["controller"] == "ddqn"]
    sc_labels = [s["scenario"].title() for s in ddqn_rows]
    rewards = [s.get("total_reward_mean", 0) for s in ddqn_rows]
    ax.bar(sc_labels, rewards, color=colors["ddqn"], alpha=0.8)
    ax.set_title("DDQN Total Reward")
    ax.set_ylabel("Reward")
    ax.grid(True, axis='y', alpha=0.3)

    # Phase switches: actuated vs DDQN
    ax = axes[2, 3]
    x = np.arange(len(scenarios))
    width = 0.35
    for i, ctrl in enumerate(["actuated", "ddqn"]):
        ctrl_rows = [s for s in summary if s["controller"] == ctrl]
        means = [next((r.get("phase_switches_mean", 0) for r in ctrl_rows if r["scenario"] == sc), 0) for sc in scenarios]
        stds  = [next((r.get("phase_switches_std", 0) for r in ctrl_rows if r["scenario"] == sc), 0) for sc in scenarios]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=ctrl.upper(), color=colors[ctrl], alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in scenarios])
    ax.set_title("Phase Switches")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("evaluate_report.png", dpi=200)
    plt.show()
    print("Saved 12-panel dashboard -> evaluate_report.png")


# ─────────────────────── MAIN ───────────────────────────────────────────────
def main():
    policy = None
    if os.path.exists(POLICY_PATH):
        print(f"Loading DDQN policy from: {POLICY_PATH}")
        policy = tf.saved_model.load(POLICY_PATH)
    else:
        print(f"[ERROR] Policy not found at {POLICY_PATH}. DDQN evaluation will be skipped.")

    scenarios   = ["balanced", "asymmetric", "ev_heavy"]
    controllers = ["fixed", "actuated", "ddqn"]
    results     = []

    for scenario in scenarios:
        for ep in range(1, EPISODES_PER_SCENARIO + 1):
            seed = make_seed(scenario, ep)

            print(f"\n{'='*60}")
            print(f"Scenario : {scenario.upper():12} | Episode : {ep:>2}")
            print(f"Seed     : {seed}")
            print(f"{'='*60}")

            # Generate traffic ONCE per episode — all controllers see the same files
            generate_traffic(scenario, seed)

            for controller in controllers:
                if controller == "ddqn" and policy is None:
                    continue

                print(f"  Controller: {controller.upper()} ... ", end="", flush=True)
                metrics = run_episode(controller, scenario, policy, seed=seed)
                metrics["episode"] = ep
                metrics["seed"] = seed
                results.append(metrics)
                print(f"Steps={metrics['episode_steps']:>4}  "
                      f"Thru={metrics['total_throughput']:>3}  "
                      f"AvgWait={metrics['avg_wait_time']:>6.1f}s  "
                      f"EVclr={metrics['ev_clearance_rate']:>5.1f}%")

    save_results_csv(results)
    summary = compute_summary(results)
    save_summary_csv(summary)
    print_console_summary(summary)
    plot_dashboard(results, summary)


if __name__ == "__main__":
    main()