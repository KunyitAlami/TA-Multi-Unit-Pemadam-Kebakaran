import os
import sys
import csv
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ENV_DIR = os.path.join(PROJECT_ROOT, "environment")
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

import libsumo as traci

from environment.env import TrafficEnvironment
from metric_evaluation import (arrivalTimeUPK,waitingTimeUPK,queueLengthNonEMV,waitingTimeNonEMV)

SEED = 42
DEMAND_LEVEL = None
JUMLAH_UNIT = None
MAX_SIMULATION_TIME = 3600

RESULT_DIR = os.path.join(PROJECT_ROOT, "hasil_evaluasi", "greedy_premption")

def prepare_result_folder():
    os.makedirs(RESULT_DIR, exist_ok=True)


def make_result_files(start_datetime):
    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
    return (os.path.join(RESULT_DIR, f"greedy_premption_{timestamp}.csv"), os.path.join(RESULT_DIR, f"greedy_premption_{timestamp}.txt"))

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

def build_lane_phase_mapping(tl):
    controlled_links = traci.trafficlight.getControlledLinks(tl)
    signal_to_lanes = {}

    for signal_index, links in enumerate(controlled_links):
        for link in links:
            if not link:
                continue

            incoming_lane = link[0]
            signal_to_lanes.setdefault(signal_index, set()).add(incoming_lane)

    controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))
    lane_phase_mapping = {lane: [] for lane in controlled_lanes}
    programs = (traci.trafficlight.getCompleteRedYellowGreenDefinition(tl))

    if not programs:
        return lane_phase_mapping

    active_program = traci.trafficlight.getProgram(tl)
    selected_logic = None

    for logic in programs:
        if logic.programID == active_program:
            selected_logic = logic
            break

    if selected_logic is None:
        selected_logic = programs[0]

    for phase_index, phase in enumerate(selected_logic.phases):
        signal_state = phase.state

        for signal_index, lanes in signal_to_lanes.items():
            if signal_index >= len(signal_state):
                continue

            signal = signal_state[signal_index]

            if signal not in "G":
                continue

            for lane in lanes:
                if lane not in lane_phase_mapping:
                    lane_phase_mapping[lane] = []

                if phase_index not in lane_phase_mapping[lane]:
                    lane_phase_mapping[lane].append(phase_index)

    return lane_phase_mapping


def get_upk_candidates(tl, lane_phase_mapping):
    candidates = []

    incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))
    vehicle_ids = set(traci.vehicle.getIDList())

    for lane in incoming_lanes:
        phases = lane_phase_mapping.get(lane, [])

        if not phases:
            continue

        vehicles = traci.lane.getLastStepVehicleIDs(lane)

        for vehicle_id in vehicles:
            if vehicle_id not in vehicle_ids:
                continue

            try:
                if (traci.vehicle.getVehicleClass(vehicle_id)!= "emergency"):
                    continue

                lane_position = float(traci.vehicle.getLanePosition(vehicle_id))
                lane_length = float(traci.lane.getLength(lane))
                distance_to_stopline = max(lane_length - lane_position,0.0)

                candidates.append({
                    "vehicle_id": vehicle_id,
                    "lane": lane,
                    "distance_to_stopline": distance_to_stopline,
                    "phases": list(phases),
                })

            except Exception:
                continue

    candidates.sort(key=lambda item: (item["distance_to_stopline"], item["vehicle_id"],))

    return candidates


def initialize_greedy_preemption(env):
    env._greedy_preemption_active = {
        tl: False for tl in env.lampu_merah
    }

    env._greedy_target_vehicle = {
        tl: None for tl in env.lampu_merah
    }

    env._greedy_target_lane = {
        tl: None for tl in env.lampu_merah
    }

    env._greedy_target_edge = {
        tl: None for tl in env.lampu_merah
    }

    env._greedy_target_phase = {
        tl: None for tl in env.lampu_merah
    }


def greedy_emv_has_passed_intersection(env, tl):
    vehicle_id = env._greedy_target_vehicle[tl]
    target_edge = env._greedy_target_edge[tl]

    if vehicle_id is None:
        return True

    try:
        if vehicle_id not in set(traci.vehicle.getIDList()):
            return True

        current_edge = traci.vehicle.getRoadID(vehicle_id)

        if current_edge != target_edge:
            return True

        return False

    except Exception:
        return False


def finish_greedy_preemption(env, tl):
    env._greedy_preemption_active[tl] = False
    env._greedy_target_vehicle[tl] = None
    env._greedy_target_lane[tl] = None
    env._greedy_target_edge[tl] = None
    env._greedy_target_phase[tl] = None

    env.preemption_active[tl] = False
    env.active_phase[tl] = None
    env.green_elapsed[tl] = 0


def apply_greedy_preemption(env):
    for tl in env.lampu_merah:

        if env._greedy_preemption_active[tl]:

            if greedy_emv_has_passed_intersection(env, tl):
                finish_greedy_preemption(env, tl)

            continue

        if env.transition_active[tl]:
            continue

        target_vehicle = env._greedy_target_vehicle[tl]
        target_phase = env._greedy_target_phase[tl]

        if (
            target_vehicle is not None
            and target_phase is not None
            and env.preemption_active[tl]
            and traci.trafficlight.getPhase(tl) == target_phase
        ):
            env._greedy_preemption_active[tl] = True
            continue

        candidates = get_upk_candidates(tl, env._greedy_lane_phase_mapping[tl])

        if not candidates:
            continue

        selected = candidates[0]
        target_phases = selected["phases"]

        if not target_phases:
            continue

        target_phase = target_phases[0]
        target_vehicle = selected["vehicle_id"]
        target_lane = selected["lane"]

        try:
            target_edge = traci.lane.getEdgeID(target_lane)
        except Exception:
            continue

        current_phase = traci.trafficlight.getPhase(tl)

        env._greedy_target_vehicle[tl] = target_vehicle
        env._greedy_target_lane[tl] = target_lane
        env._greedy_target_edge[tl] = target_edge
        env._greedy_target_phase[tl] = target_phase

        if current_phase == target_phase:
            env.active_phase[tl] = target_phase
            env.preemption_active[tl] = True
            env.transition_active[tl] = False
            env.green_elapsed[tl] = 0
            env._greedy_preemption_active[tl] = True
            continue

        env.execute_action(tl, target_phase)

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

def save_csv(filepath,
    start,
    end,
    wall_time,
    simulation_time,
    total_steps,
    env,
    arrival_upk,
    waiting_upk,
    queue_non_emv,
    waiting_non_emv,
):
    row = {
        "method": "greedy_premption",
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


def save_txt(
    filepath,
    start,
    end,
    wall_time,
    simulation_time,
    total_steps,
    env,
    arrival_upk,
    waiting_upk,
    queue_non_emv,
    waiting_non_emv,
):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("HASIL EVALUASI GREEDY PREEMPTION\n")
        f.write("=" * 80 + "\n\n")

        f.write("Metode              : Greedy Preemption\n")
        f.write("Agent action        : Rule-based greedy\n")
        f.write("Green duration      : Hold sampai EMV melewati intersection\n")
        f.write("TLS control         : Nearest EMV + MAPPO-like phase transition\n")
        f.write("Waiting Time UPK    : Evaluator akumulasi getWaitingTime()\n")
        f.write(f"Seed                : {SEED}\n")
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

        f.write("\nQUEUE LENGTH NON-EMV\n")
        f.write("-" * 80 + "\n")
        for tl, value in queue_non_emv.items():
            f.write(f"{tl:<30} : {value:.6f}\n")

        f.write("\nWAITING TIME NON-EMV\n")
        f.write("-" * 80 + "\n")
        for tl, value in waiting_non_emv.items():
            f.write(f"{tl:<30} : {value:.6f}\n")

        f.write("\n" + "=" * 80 + "\n")


def run_greedy_premption():
    prepare_result_folder()
    start_datetime = datetime.now()
    wall_start = time.time()

    env = TrafficEnvironment(demand_level=DEMAND_LEVEL, jumlah_unit=JUMLAH_UNIT)

    try:
        env.reset(seed=SEED)
        env._eval_previous_waiting = {}
        env._eval_total_waiting = {}     
        env._greedy_lane_phase_mapping = {}
        initialize_greedy_preemption(env)

        for tl in env.lampu_merah:
            env._greedy_lane_phase_mapping[tl] = (build_lane_phase_mapping(tl))

        print("\n" + "=" * 80)
        print("GREEDY PREEMPTION BASELINE - FAIR EVALUATOR")
        print("=" * 80)
        print(f"Seed         : {SEED}")
        print(f"Demand level : {DEMAND_LEVEL}")
        print(f"Jumlah unit  : {JUMLAH_UNIT}")
        print(f"Simulation   : {MAX_SIMULATION_TIME} detik")
        print(f"Fire events  : {len(env.daftar_fire_event)}")
        print("=" * 80)

        steps = 0

        while traci.simulation.getTime() < MAX_SIMULATION_TIME:
            process_fire_events(env)
            apply_greedy_preemption(env)
            env.simulation_step()
            record_waiting_time_upk(env)
            update_fire_trucks(env)
            collect_non_emv_metrics(env)

            steps += 1

        end_datetime = datetime.now()
        wall_time = time.time() - wall_start
        simulation_time = traci.simulation.getTime()

        env.buat_avg_waiting_time_upk = dict(env._eval_total_waiting)
        arrival_upk = arrivalTimeUPK(env.buat_avg_arrival_time_upk)
        waiting_upk = waitingTimeUPK(env.buat_avg_waiting_time_upk)
        queue_non_emv = queueLengthNonEMV(env.buat_avg_queue_length_non_upk)
        waiting_non_emv = waitingTimeNonEMV(env.buat_avg_waiting_time_non_upk)
        csv_path, txt_path = make_result_files(start_datetime)

        save_csv(
            csv_path,
            start_datetime,
            end_datetime,
            wall_time,
            simulation_time,
            steps,
            env,
            arrival_upk,
            waiting_upk,
            queue_non_emv,
            waiting_non_emv
        )

        save_txt(
            txt_path,
            start_datetime,
            end_datetime,
            wall_time,
            simulation_time,
            steps,
            env,
            arrival_upk,
            waiting_upk,
            queue_non_emv,
            waiting_non_emv
        )

        print("\n" + "=" * 80)
        print("GREEDY PREEMPTION SELESAI")
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
    run_greedy_premption()