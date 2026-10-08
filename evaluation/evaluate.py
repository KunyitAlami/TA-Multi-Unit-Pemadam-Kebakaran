import os
import sys
import csv
import time
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

import libsumo as traci

from environment.env import TrafficEnvironment, MIN_GREEN_DURATION
from environment.state import state
from environment.action import action
from training.mappo import MAPPO
from metric_evaluation import arrivalTimeUPK, waitingTimeUPK, queueLengthNonEMV, waitingTimeNonEMV

SEED = 42
DEMAND_LEVEL = None
JUMLAH_UNIT = None
MAX_SIMULATION_TIME = 3600
MODEL_PATH = os.path.join(PROJECT_ROOT, "hasil_train_model", "mappo", "mappo_final.pt")
RESULT_DIR = os.path.join(PROJECT_ROOT, "hasil_evaluasi", "mappo")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_result_folder():
    os.makedirs(RESULT_DIR, exist_ok=True)


def make_result_files(start_datetime):
    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
    return (os.path.join(RESULT_DIR, f"mappo_{timestamp}.csv"), os.path.join(RESULT_DIR, f"mappo_{timestamp}.txt"))


def process_fire_events(env):
    current_time = traci.simulation.getTime()
    for event in env.daftar_fire_event:
        if not event["status"] and current_time >= event["time_fire_event"]:
            for i in range(event["vehicle_number"]):
                vehicle_id = f"fire_truck_{env.counter}"
                route_id = f"route_fire_truck_{env.counter}"
                entry_edge = event["entry_edges"][i]
                env.spawnFireTruck(
                    entry_edge=entry_edge,
                    destination_edge=event["destination_edge"],
                    route_id=route_id,
                    vehicle_id=vehicle_id,
                )
                env.active_fire_trucks.append({
                    "vehicle_id": vehicle_id,
                    "destination_edge": event["destination_edge"],
                    "waktu_spawn": current_time,
                })
                env._eval_previous_waiting[vehicle_id] = 0.0
                env._eval_total_waiting[vehicle_id] = 0.0
                env.counter += 1
            event["status"] = True

def record_waiting_time_upk(env):
    vehicle_ids = set(traci.vehicle.getIDList())
    for truck in env.active_fire_trucks:
        vehicle_id = truck["vehicle_id"]
        if vehicle_id not in vehicle_ids:
            continue
        try:
            current_waiting = float(traci.vehicle.getWaitingTime(vehicle_id))
        except Exception:
            continue
        previous_waiting = env._eval_previous_waiting.get(vehicle_id, 0.0)
        delta_waiting = current_waiting - previous_waiting
        if delta_waiting > 0:
            env._eval_total_waiting[vehicle_id] += delta_waiting
        env._eval_previous_waiting[vehicle_id] = current_waiting


def update_fire_trucks(env):
    arrived_ids = set(traci.simulation.getArrivedIDList())
    active_trucks = []

    for truck in env.active_fire_trucks:
        vehicle_id = truck["vehicle_id"]

        if vehicle_id in arrived_ids:
            current_time = traci.simulation.getTime()
            env.buat_avg_arrival_time_upk[vehicle_id] = current_time - truck["waktu_spawn"]
            continue

        if vehicle_id in traci.vehicle.getIDList():
            active_trucks.append(truck)

    env.active_fire_trucks = active_trucks


def collect_non_emv_metrics(env):
    for tl in env.lampu_merah:
        env.buat_avg_queue_length_non_upk[tl].append(env.hitung_queue_length_non_upk(tl))
        env.buat_avg_waiting_time_non_upk[tl].append(env.hitung_waiting_time_non_upk(tl))


def get_mappo_action(env, mappo, tl, states):
    upk_terdekat = env.pilih_upk_terdekat(tl)
    if upk_terdekat is None:
        return None

    state_tensor = torch.tensor(states[tl], dtype=torch.float32, device=mappo.device).unsqueeze(0)

    with torch.no_grad():
        logits = mappo.actors[tl](state_tensor)

    action_mask = env.get_action_mask(tl)
    action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=mappo.device).unsqueeze(0)

    if not action_mask_tensor.any():
        return None

    masked_logits = logits.masked_fill(~action_mask_tensor, -1e9)
    action_id = int(torch.argmax(masked_logits, dim=-1).item())
    return action_id


def apply_mappo_actions(env, mappo, states):
    for tl in env.lampu_merah:
        if env.transition_active[tl]:
            continue
        if env.preemption_active[tl] and env.green_elapsed[tl] < MIN_GREEN_DURATION:
            continue

        action_id = get_mappo_action(env, mappo, tl, states)
        if action_id is None or action_id == 0:
            continue

        action_mask = env.get_action_mask(tl)
        if action_id >= len(action_mask) or action_mask[action_id] == 0:
            continue

        phase = action(env.action_space_marl, tl, action_id)
        if phase == "HOLD":
            continue

        env.execute_action(tl, phase)


def load_mappo_model(env, initial_states):
    state_dim = {tl: len(initial_states[tl]) for tl in env.lampu_merah}
    action_dim = {tl: len(env.action_space_marl[tl]) for tl in env.lampu_merah}
    num_agents = len(env.lampu_merah)

    print("\n=== MAPPO MODEL CONFIGURATION ===")
    print(f"Number of agents : {num_agents}")
    for tl in env.lampu_merah:
        print(f"{tl:<25} | State dim = {state_dim[tl]:3d} | Action dim = {action_dim[tl]:3d}")

    mappo = MAPPO(num_agents=num_agents, state_dim=state_dim, action_dim=action_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5, epochs=10, device=DEVICE)

    print("\nLoading model:")
    print(MODEL_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    for tl in mappo.agent_names:
        mappo.actors[tl].load_state_dict(checkpoint["actors"][tl])

    mappo.critic.load_state_dict(checkpoint["critic"])

    for tl in mappo.agent_names:
        mappo.actors[tl].eval()

    mappo.critic.eval()

    print("\nModel berhasil dimuat.")
    return mappo


def save_csv(filepath, start, end, wall_time, simulation_time, total_steps, env, arrival_upk, waiting_upk, queue_non_emv, waiting_non_emv):
    row = {
        "method": "mappo",
        "training_episodes": 400,
        "evaluation_episode": 1,
        "seed": SEED,
        "demand_level": DEMAND_LEVEL,
        "jumlah_unit": JUMLAH_UNIT,
        "start_datetime": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_datetime": end.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_clock_seconds": wall_time,
        "simulation_time_seconds": simulation_time,
        "total_steps": total_steps,
        "active_fire_trucks_at_end": len(env.active_fire_trucks),
        "arrival_time_upk": arrival_upk,
        "waiting_time_upk": waiting_upk,
    }

    for tl, value in queue_non_emv.items():
        row[f"queue_length_non_emv_{tl}"] = value

    for tl, value in waiting_non_emv.items():
        row[f"waiting_time_non_emv_{tl}"] = value

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_txt(filepath, start, end, wall_time, simulation_time, total_steps, env, arrival_upk, waiting_upk, queue_non_emv, waiting_non_emv):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("HASIL EVALUASI MAPPO\n")
        f.write("=" * 80 + "\n\n")
        f.write("Metode              : MAPPO\n")
        f.write("Training episode    : 400\n")
        f.write("Evaluation episode  : 1\n")
        f.write("Policy action       : Greedy / Argmax + Action Mask\n")
        f.write("Evaluation loop     : Custom evaluator\n")
        f.write("Seed                : 42\n")
        f.write(f"Jam mulai           : {start:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Jam selesai         : {end:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Wall-clock runtime  : {wall_time:.6f} detik\n")
        f.write(f"Simulation time     : {simulation_time:.2f} detik\n")
        f.write(f"Total steps         : {total_steps}\n")
        f.write(f"Active UPK at end   : {len(env.active_fire_trucks)}\n\n")

        f.write("SCENARIO SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Jumlah fire event   : {len(env.daftar_fire_event)}\n")
        f.write(f"Total UPK planned   : {sum(e['vehicle_number'] for e in env.daftar_fire_event)}\n")
        f.write(f"Total UPK spawned   : "f"{sum(e['vehicle_number'] for e in env.daftar_fire_event)}\n\n")

        f.write("METRIC UPK\n")
        f.write("-" * 80 + "\n")
        f.write(f"Arrival Time UPK    : {arrival_upk:.6f} s\n")
        f.write(f"Waiting Time UPK    : {waiting_upk:.6f} s\n\n")

        f.write("WAITING TIME PER FIRE TRUCK\n")
        f.write("-" * 80 + "\n")
        for vehicle_id in sorted(env.buat_avg_waiting_time_upk):
            f.write(f"{vehicle_id:<25} : "f"{env.buat_avg_waiting_time_upk[vehicle_id]:.6f} s\n")

        f.write("\nARRIVAL TIME PER FIRE TRUCK\n")
        f.write("-" * 80 + "\n")
        for vehicle_id in sorted(env.buat_avg_arrival_time_upk):
            f.write(f"{vehicle_id:<25} : {env.buat_avg_arrival_time_upk[vehicle_id]:.6f} s\n")

        f.write("\nQUEUE LENGTH NON-EMV\n")
        f.write("-" * 80 + "\n")
        for tl, value in queue_non_emv.items():
            f.write(f"{tl:<30} : {value:.6f}\n")

        f.write("\nWAITING TIME NON-EMV\n")
        f.write("-" * 80 + "\n")
        for tl, value in waiting_non_emv.items():
            f.write(f"{tl:<30} : {value:.6f}\n")

        f.write("\n" + "=" * 80 + "\n")


def run_evaluation():
    prepare_result_folder()
    start_datetime = datetime.now()
    wall_start = time.time()

    print("\n" + "=" * 80)
    print("EVALUASI MAPPO")
    print("=" * 80)
    print(f"Model       : {MODEL_PATH}")
    print("Episode     : 1")
    print(f"Simulation  : {MAX_SIMULATION_TIME} detik")
    print(f"Seed        : {SEED}")
    print("=" * 80)

    env = TrafficEnvironment(demand_level=DEMAND_LEVEL, jumlah_unit=JUMLAH_UNIT)

    try:
        states, info = env.reset(seed=SEED)
        env._eval_previous_waiting = {}
        env._eval_total_waiting = {}
        mappo = load_mappo_model(env, states)
        steps = 0

        while traci.simulation.getTime() < MAX_SIMULATION_TIME:
            process_fire_events(env)
            apply_mappo_actions(env, mappo, states)
            env.simulation_step()
            record_waiting_time_upk(env)
            update_fire_trucks(env)
            collect_non_emv_metrics(env)
            states = state(env.lampu_merah)
            steps += 1

            simulation_time = traci.simulation.getTime()
            if steps % 100 == 0:
                print(f"Step = {steps:5d} | Simulation time = {simulation_time:8.1f} s | Active UPK = {len(env.active_fire_trucks):3d}")

        end_datetime = datetime.now()
        wall_time = time.time() - wall_start
        simulation_time = traci.simulation.getTime()
        env.buat_avg_waiting_time_upk = dict(env._eval_total_waiting)
        arrival_upk = arrivalTimeUPK(env.buat_avg_arrival_time_upk)
        waiting_upk = waitingTimeUPK(env.buat_avg_waiting_time_upk)
        queue_non_emv = queueLengthNonEMV(env.buat_avg_queue_length_non_upk)
        waiting_non_emv = waitingTimeNonEMV(env.buat_avg_waiting_time_non_upk)

        csv_path, txt_path = make_result_files(start_datetime)
        save_csv(csv_path, start_datetime, end_datetime, wall_time, simulation_time, steps, env, arrival_upk, waiting_upk, queue_non_emv, waiting_non_emv)
        save_txt(txt_path, start_datetime, end_datetime, wall_time, simulation_time, steps, env, arrival_upk, waiting_upk, queue_non_emv, waiting_non_emv)

        print("\n" + "=" * 80)
        print("EVALUASI MAPPO SELESAI")
        print("=" * 80)
        print(f"Simulation time  : {simulation_time:.2f} s")
        print(f"Total steps      : {steps}")
        print(f"Total UPK spawned: "f"{sum(e['vehicle_number'] for e in env.daftar_fire_event)}")
        print(f"Active UPK end   : {len(env.active_fire_trucks)}")
        print(f"Arrival Time UPK : {arrival_upk:.4f} s")
        print(f"Waiting Time UPK : {waiting_upk:.4f} s")
        print(f"CSV              : {csv_path}")
        print(f"TXT              : {txt_path}")
        print("=" * 80)

    finally:
        env.close()
        print("\nEnvironment ditutup.")


if __name__ == "__main__":
    run_evaluation()