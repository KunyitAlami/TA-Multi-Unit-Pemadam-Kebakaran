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

from environment.env import (
    TrafficEnvironment,
    DECISION_INTERVAL,
    MIN_GREEN_DURATION,
    YELLOW_DURATION,
    ALL_RED_DURATION,
)
from metric_evaluation import (arrivalTimeUPK, waitingTimeUPK, queueLengthNonEMV, waitingTimeNonEMV)

SEED = 42
DEMAND_LEVEL = None
JUMLAH_UNIT = None
MAX_SIMULATION_TIME = 3600

# None = semua EMV di lane masuk dianggap (sama dengan action mask MAPPO).
# Isi angka (meter) jika ingin membatasi jarak deteksi.
DETECTION_DISTANCE = None

RESULT_DIR = os.path.join(PROJECT_ROOT, "hasil_evaluasi", "greedy_preemption")


def prepare_result_folder():
    os.makedirs(RESULT_DIR, exist_ok=True)


def make_result_files(start_datetime):
    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
    return (
        os.path.join(RESULT_DIR, f"greedy_preemption_{timestamp}.csv"),
        os.path.join(RESULT_DIR, f"greedy_preemption_{timestamp}.txt"),
    )


# ----------------------------------------------------------------------
# Spawn UPK
# ----------------------------------------------------------------------
def process_fire_events(env):
    current_time = traci.simulation.getTime()

    for event in env.daftar_fire_event:
        if event["status"] or current_time < event["time_fire_event"]:
            continue

        for i in range(event["vehicle_number"]):
            vehicle_id = f"fire_truck_{env.counter}"
            route_id = f"route_fire_truck_{env.counter}"

            spawned = env.spawnFireTruck(
                entry_edge=event["entry_edges"][i],
                destination_edge=event["destination_edge"],
                route_id=route_id,
                vehicle_id=vehicle_id,
            )

            if spawned:
                env.active_fire_trucks.append({
                    "vehicle_id": vehicle_id,
                    "destination_edge": event["destination_edge"],
                    "waktu_spawn": current_time,
                })
                env._eval_previous_waiting[vehicle_id] = 0.0
                env._eval_total_waiting[vehicle_id] = 0.0
                env._greedy_spawned += 1

            env.counter += 1

        event["status"] = True


# ----------------------------------------------------------------------
# Pemetaan (lane asal, edge tujuan) -> fase hijau, berbasis linkIndex
# ----------------------------------------------------------------------
def build_link_phase_mapping(env, tl):
    """Kunci pemetaan = (incoming lane, outgoing edge) sehingga gerakan lurus
    dan belok tidak tertukar. Fase hijau protected ('G') diprioritaskan;
    'g' (permisif) hanya dipakai jika sebuah gerakan tidak punya fase 'G'."""
    controlled_links = traci.trafficlight.getControlledLinks(tl)
    signal_to_keys = {}

    for signal_index, links in enumerate(controlled_links):
        for link in links:
            if not link:
                continue

            incoming_lane, outgoing_lane = link[0], link[1]
            outgoing_edge = traci.lane.getEdgeID(outgoing_lane)
            signal_to_keys.setdefault(signal_index, set()).add((incoming_lane, outgoing_edge))

    protected = {}
    permissive = {}

    for phase in env.get_phase_information(tl):
        state = phase["state"]

        for signal_index, keys in signal_to_keys.items():
            if signal_index >= len(state):
                continue

            symbol = state[signal_index]

            if symbol == "G":
                target = protected
            elif symbol == "g":
                target = permissive
            else:
                continue

            for key in keys:
                phases = target.setdefault(key, [])
                if phase["index"] not in phases:
                    phases.append(phase["index"])

    all_keys = set(protected) | set(permissive)
    for keys in signal_to_keys.values():
        all_keys |= keys

    return {key: list(protected.get(key) or permissive.get(key, [])) for key in all_keys}


# ----------------------------------------------------------------------
# Kandidat EMV dan pemilihan fase
# ----------------------------------------------------------------------
def get_emv_candidates(env, tl):
    mapping = env._greedy_link_phase_mapping[tl]
    incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))
    candidates = []

    for lane in incoming_lanes:
        lane_length = float(traci.lane.getLength(lane))

        for vehicle_id in traci.lane.getLastStepVehicleIDs(lane):
            try:
                if traci.vehicle.getVehicleClass(vehicle_id) != "emergency":
                    continue

                route = traci.vehicle.getRoute(vehicle_id)
                route_index = traci.vehicle.getRouteIndex(vehicle_id)

                if route_index < 0 or route_index + 1 >= len(route):
                    continue

                next_edge = route[route_index + 1]
                phases = mapping.get((lane, next_edge), [])

                if not phases:
                    continue

                distance = max(lane_length - float(traci.vehicle.getLanePosition(vehicle_id)), 0.0)

                if DETECTION_DISTANCE is not None and distance > DETECTION_DISTANCE:
                    continue

                candidates.append({
                    "vehicle_id": vehicle_id,
                    "lane": lane,
                    "next_edge": next_edge,
                    "distance_to_stopline": distance,
                    "phases": list(phases),
                })
            except Exception:
                continue

    candidates.sort(key=lambda c: (c["distance_to_stopline"], c["vehicle_id"]))
    return candidates


def choose_target_phase(candidates, current_phase):
    """Greedy: layani EMV terdekat. Jika fase saat ini sudah melayaninya, tetap.
    Jika ada beberapa pilihan fase, ambil yang melayani EMV paling banyak."""
    nearest = candidates[0]

    if current_phase in nearest["phases"]:
        return current_phase

    def served_count(phase):
        return sum(1 for c in candidates if phase in c["phases"])

    return max(nearest["phases"], key=lambda p: (served_count(p), -nearest["phases"].index(p)))


# ----------------------------------------------------------------------
# Kontrol greedy (dipanggil setiap DECISION_INTERVAL detik, seperti MAPPO)
# ----------------------------------------------------------------------
def apply_greedy_preemption(env):
    for tl in env.lampu_merah:
        if env.transition_active[tl]:
            continue

        candidates = get_emv_candidates(env, tl)
        current_phase = traci.trafficlight.getPhase(tl)

        # Recovery bersama dengan MAPPO: lepas hold jika tidak ada EMV tersisa
        needed_phases = set()
        for c in candidates:
            needed_phases.update(c["phases"])

        if env.maintain_preemption(tl, needed_phases):
            continue

        if env.preemption_active[tl]:
            # Hold selama masih ada EMV (siapa pun) yang membutuhkan fase ini
            if any(current_phase in c["phases"] for c in candidates):
                continue

            # EMV tersisa butuh fase lain, hormati minimum green dulu
            if env.green_elapsed[tl] < MIN_GREEN_DURATION:
                continue

            env.execute_action(tl, choose_target_phase(candidates, current_phase), hold=True)
            continue

        if not candidates:
            continue

        env.execute_action(tl, choose_target_phase(candidates, current_phase), hold=True)


# ----------------------------------------------------------------------
# Metrik
# ----------------------------------------------------------------------
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
    """Sama dengan MAPPO: arrival dicatat dan UPK dihapus saat mencapai edge tujuan."""
    remaining = []

    for truck in env.active_fire_trucks:
        if env.updateFireTruck(truck["vehicle_id"], truck["destination_edge"], truck["waktu_spawn"]):
            remaining.append(truck)

    env.active_fire_trucks = remaining


def collect_non_emv_metrics(env):
    for tl in env.lampu_merah:
        env.buat_avg_queue_length_non_upk[tl].append(env.hitung_queue_length_non_upk(tl))
        env.buat_avg_waiting_time_non_upk[tl].append(env.hitung_waiting_time_non_upk(tl))


# ----------------------------------------------------------------------
# Penyimpanan hasil
# ----------------------------------------------------------------------
def save_csv(filepath, start, end, wall_time, simulation_time, total_steps, env,
             arrival_upk, waiting_upk, queue_non_emv, waiting_non_emv):
    row = {
        "method": "greedy_preemption",
        "seed": SEED,
        "demand_level": DEMAND_LEVEL,
        "jumlah_unit": JUMLAH_UNIT,
        "start_datetime": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_datetime": end.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_clock_seconds": wall_time,
        "simulation_time_seconds": simulation_time,
        "total_steps": total_steps,
        "upk_spawned": env._greedy_spawned,
        "upk_arrived": len(env.buat_avg_arrival_time_upk),
        "active_fire_trucks_at_end": len(env.active_fire_trucks),
        "arrival_time_upk": arrival_upk,
        "waiting_time_upk": waiting_upk,
        "transition_count": env.get_transition_report()["total_transitions"],
        "transition_green_violations": env.get_transition_report()["total_violations"],
    }

    for tl, value in queue_non_emv.items():
        row[f"queue_length_non_emv_{tl}"] = value

    for tl, value in waiting_non_emv.items():
        row[f"waiting_time_non_emv_{tl}"] = value

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_txt(filepath, start, end, wall_time, simulation_time, total_steps, env,
             arrival_upk, waiting_upk, queue_non_emv, waiting_non_emv):
    planned = sum(e["vehicle_number"] for e in env.daftar_fire_event)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("HASIL EVALUASI GREEDY PREEMPTION\n")
        f.write("=" * 80 + "\n\n")

        f.write("Metode              : Greedy Preemption\n")
        f.write("Agent action        : Rule-based greedy (EMV terdekat, fase berbasis linkIndex)\n")
        f.write("Green duration      : Hold selama masih ada EMV yang membutuhkan fase tersebut\n")
        f.write(f"Transisi            : Green -> Yellow ({YELLOW_DURATION}s) -> All-Red ({ALL_RED_DURATION}s) -> Target Green\n")
        f.write(f"Decision interval   : {DECISION_INTERVAL} detik\n")
        f.write(f"Minimum green       : {MIN_GREEN_DURATION} detik\n")
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
        f.write(f"Total UPK planned   : {planned}\n")
        f.write(f"Total UPK spawned   : {env._greedy_spawned}\n")
        f.write(f"Total UPK arrived   : {len(env.buat_avg_arrival_time_upk)}\n\n")

        f.write("METRIC UPK\n")
        f.write("-" * 80 + "\n")
        f.write(f"Arrival Time UPK    : {arrival_upk:.6f} s\n")
        f.write(f"Waiting Time UPK    : {waiting_upk:.6f} s\n\n")

        f.write("WAITING TIME PER FIRE TRUCK\n")
        f.write("-" * 80 + "\n")
        for vehicle_id in sorted(env.buat_avg_waiting_time_upk):
            f.write(f"{vehicle_id:<25} : {env.buat_avg_waiting_time_upk[vehicle_id]:.6f} s\n")

        report = env.get_transition_report()
        f.write("\nVALIDASI TRANSISI (RUNTIME)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total transisi      : {report['total_transitions']}\n")
        f.write(f"Pelanggaran hijau   : {report['total_violations']}\n")
        for tl, row in report["per_tls"].items():
            f.write(
                f"{tl:<22} : n={row['count']} dari_hijau={row['from_green']} "
                f"dari_kuning={row['from_yellow']} dari_merah={row['from_red']} "
                f"min_kuning={row['min_yellow']} min_allred={row['min_all_red']} "
                f"pelanggaran={row['violations']}\n"
            )

        f.write("\nQUEUE LENGTH NON-EMV\n")
        f.write("-" * 80 + "\n")
        for tl, value in queue_non_emv.items():
            f.write(f"{tl:<30} : {value:.6f}\n")

        f.write("\nWAITING TIME NON-EMV\n")
        f.write("-" * 80 + "\n")
        for tl, value in waiting_non_emv.items():
            f.write(f"{tl:<30} : {value:.6f}\n")

        f.write("\n" + "=" * 80 + "\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run_greedy_preemption():
    prepare_result_folder()
    start_datetime = datetime.now()
    wall_start = time.time()

    env = TrafficEnvironment(demand_level=DEMAND_LEVEL, jumlah_unit=JUMLAH_UNIT)

    try:
        env.reset(seed=SEED)
        env._eval_previous_waiting = {}
        env._eval_total_waiting = {}
        env._greedy_spawned = 0
        env._greedy_link_phase_mapping = {
            tl: build_link_phase_mapping(env, tl) for tl in env.lampu_merah
        }

        print("\n" + "=" * 80)
        print("GREEDY PREEMPTION BASELINE")
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

            for _ in range(env.steps_per_decision):
                env.simulation_step()
                env.update_emv_tracking()
                env.track_non_emv_travel_time()
                record_waiting_time_upk(env)
                update_fire_trucks(env)
                collect_non_emv_metrics(env)
                steps += 1

        end_datetime = datetime.now()
        wall_time = time.time() - wall_start
        simulation_time = traci.simulation.getTime()

        # Hanya UPK yang tiba, sama seperti pencatatan MAPPO
        env.buat_avg_waiting_time_upk = {
            vid: wait for vid, wait in env._eval_total_waiting.items()
            if vid in env.buat_avg_arrival_time_upk
        }

        arrival_upk = arrivalTimeUPK(env.buat_avg_arrival_time_upk)
        waiting_upk = waitingTimeUPK(env.buat_avg_waiting_time_upk)
        queue_non_emv = queueLengthNonEMV(env.buat_avg_queue_length_non_upk)
        waiting_non_emv = waitingTimeNonEMV(env.buat_avg_waiting_time_non_upk)
        csv_path, txt_path = make_result_files(start_datetime)

        args = (start_datetime, end_datetime, wall_time, simulation_time, steps, env,
                arrival_upk, waiting_upk, queue_non_emv, waiting_non_emv)

        save_csv(csv_path, *args)
        save_txt(txt_path, *args)

        print("\n" + "=" * 80)
        print("GREEDY PREEMPTION SELESAI")
        print("=" * 80)
        print(f"Simulation time  : {simulation_time:.2f} s")
        print(f"Total steps      : {steps}")
        print(f"Total UPK spawned: {env._greedy_spawned}")
        print(f"Total UPK arrived: {len(env.buat_avg_arrival_time_upk)}")
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
    run_greedy_preemption()