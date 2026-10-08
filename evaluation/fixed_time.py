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
from metric_evaluation import (arrivalTimeUPK, waitingTimeUPK, queueLengthNonEMV, waitingTimeNonEMV)

SEED = 42
DEMAND_LEVEL = None
JUMLAH_UNIT = None
MAX_SIMULATION_TIME = 3600
RESULT_DIR = os.path.join(PROJECT_ROOT, "hasil_evaluasi", "fixed_time" )

def prepare_result_folder():
    os.makedirs(RESULT_DIR, exist_ok=True)


def make_result_files(start_datetime):
    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
    return (os.path.join(RESULT_DIR, f"fixed_time_{timestamp}.csv"), os.path.join(RESULT_DIR, f"fixed_time_{timestamp}.txt"))

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
            arrival_time = (current_time - truck["waktu_spawn"])
            env.buat_avg_arrival_time_upk[vehicle_id] = arrival_time
            continue

        if vehicle_id in traci.vehicle.getIDList():
            active_trucks.append(truck)

    env.active_fire_trucks = active_trucks


def collect_non_emv_metrics(env):
    for tl in env.lampu_merah:
        env.buat_avg_queue_length_non_upk[tl].append(env.hitung_queue_length_non_upk(tl))
        env.buat_avg_waiting_time_non_upk[tl].append(env.hitung_waiting_time_non_upk(tl))

def save_csv(filepath, start, end, wall_time, simulation_time, total_steps, env, arrival_upk, waiting_upk, queue_non_emv, waiting_non_emv):
    row = {
        "method": "fixed_time",
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
        f.write("HASIL EVALUASI FIXED-TIME\n")
        f.write("=" * 80 + "\n\n")

        f.write("Metode              : Fixed Time\n")
        f.write("TLS control         : Program bawaan SUMO\n")
        f.write("UPK spawn           : Berdasarkan fire event\n")
        f.write("Waiting Time UPK    : TrafficEnvironment.updateFireTruck()\n")
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

def run_fixed_time():
    prepare_result_folder()
    start_datetime = datetime.now()
    wall_start = time.time()
    env = TrafficEnvironment(demand_level=DEMAND_LEVEL, jumlah_unit=JUMLAH_UNIT)

    try:
        env.reset(seed=SEED)
        env._eval_previous_waiting = {}
        env._eval_total_waiting = {}
        print("\n" + "=" * 80)
        print("FIXED-TIME BASELINE - FAIR EVALUATOR")
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
            traci.simulationStep()
            record_waiting_time_upk(env)
            active_fire_trucks = []
            for truck in env.active_fire_trucks:
                masih_aktif = env.updateFireTruck(
                    truck["vehicle_id"],
                    truck["destination_edge"],
                    truck["waktu_spawn"]
                )
                if masih_aktif:
                    active_fire_trucks.append(truck)
            env.active_fire_trucks = active_fire_trucks
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

        save_csv(csv_path, start_datetime, end_datetime, wall_time, simulation_time, steps, env, arrival_upk, waiting_upk, queue_non_emv, waiting_non_emv)
        save_txt(txt_path, start_datetime, end_datetime, wall_time, simulation_time, steps, env, arrival_upk, waiting_upk, queue_non_emv, waiting_non_emv)

        print("\n" + "=" * 80)
        print("FIXED-TIME SELESAI")
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
    run_fixed_time()