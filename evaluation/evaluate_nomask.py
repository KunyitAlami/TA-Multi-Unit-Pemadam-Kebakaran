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

GYMNASIUM_ENV_DIR = os.path.join(PROJECT_ROOT, "gymnasium_env")

if GYMNASIUM_ENV_DIR not in sys.path:
    sys.path.insert(0, GYMNASIUM_ENV_DIR)

TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")

if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

import libsumo as traci

from environment.env import (TrafficEnvironment, MIN_GREEN_DURATION,)
from environment.state import state
from environment.action import action
from training.mappo_nomask import MAPPO

from metric_evaluation import (arrivalTimeUPK, waitingTimeUPK, queueLengthNonEMV, waitingTimeNonEMV,)

SEED = 42
DEMAND_LEVEL = None
JUMLAH_UNIT = None
MAX_SIMULATION_TIME = 3600
MODEL_PATH = os.path.join(PROJECT_ROOT, "hasil_train_model", "mappo_nomask", "REPLACE_WITH_RUN_FOLDER", "mappo_nomask_final.pt",)
RESULT_DIR = os.path.join(PROJECT_ROOT, "hasil_evaluasi", "mappo_nomask", )
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_result_folder():
    os.makedirs(RESULT_DIR, exist_ok=True)


def make_result_files(start_datetime):
    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")

    return (os.path.join(RESULT_DIR, f"mappo_{timestamp}.csv"),
        os.path.join(RESULT_DIR, f"mappo_{timestamp}.txt"),
    )

def process_fire_events(env):

    current_time = traci.simulation.getTime()

    for event in env.daftar_fire_event:

        if (
            not event["status"]
            and current_time >= event["time_fire_event"]
        ):

            for _ in range(event["vehicle_number"]):

                vehicle_id = (
                    f"fire_truck_{env.counter}"
                )

                route_id = (
                    f"route_fire_truck_{env.counter}"
                )

                env.spawnFireTruck(
                    entry_edge=event["entry_edge"],
                    destination_edge=event["destination_edge"],
                    route_id=route_id,
                    vehicle_id=vehicle_id,
                )

                env.active_fire_trucks.append({
                    "vehicle_id": vehicle_id,
                    "destination_edge":
                        event["destination_edge"],
                    "waktu_spawn":
                        current_time,
                })

                env._eval_previous_waiting[
                    vehicle_id
                ] = 0.0

                env._eval_total_waiting[
                    vehicle_id
                ] = 0.0

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
        
        previous_waiting = (
            env._eval_previous_waiting.get(
                vehicle_id,
                0.0
            )
        )

        delta_waiting = (
            current_waiting
            - previous_waiting
        )

        if delta_waiting > 0:

            env._eval_total_waiting[
                vehicle_id
            ] += delta_waiting

        env._eval_previous_waiting[
            vehicle_id
        ] = current_waiting


# ============================================================
# UPDATE FIRE TRUCK
# ============================================================

def update_fire_trucks(env):

    active_trucks = []

    for truck in env.active_fire_trucks:

        still_active = env.updateFireTruck(
            truck["vehicle_id"],
            truck["destination_edge"],
            truck["waktu_spawn"],
        )

        if still_active:

            active_trucks.append(
                truck
            )

    env.active_fire_trucks = (
        active_trucks
    )


# ============================================================
# NON EMV METRICS
# ============================================================

def collect_non_emv_metrics(env):

    for tl in env.lampu_merah:

        env.buat_avg_queue_length_non_upk[
            tl
        ].append(
            env.hitung_queue_length_non_upk(
                tl
            )
        )

        env.buat_avg_waiting_time_non_upk[
            tl
        ].append(
            env.hitung_waiting_time_non_upk(
                tl
            )
        )


# ============================================================
# GET MASKED GREEDY ACTION
# ============================================================

def get_mappo_action(
    env,
    mappo,
    tl,
    states,
):

    # --------------------------------------------------------
    # Cek apakah ada UPK
    # --------------------------------------------------------

    upk_terdekat = (
        env.pilih_upk_terdekat(tl)
    )

    if upk_terdekat is None:

        return None


    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    state_tensor = torch.tensor(
        states[tl],
        dtype=torch.float32,
        device=mappo.device,
    ).unsqueeze(0)


    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    with torch.no_grad():

        logits = mappo.actors[tl](
            state_tensor
        )


    # --------------------------------------------------------
    # Greedy / Argmax (tanpa action mask)
    # --------------------------------------------------------

    action_id = int(
        torch.argmax(
            logits,
            dim=-1
        ).item()
    )

    return action_id


# ============================================================
# APPLY MAPPO ACTION
# ============================================================

def apply_mappo_actions(env,mappo,states,):
    for tl in env.lampu_merah:

        # ----------------------------------------------------
        # Jika sedang transition,
        # jangan menerima action baru
        # ----------------------------------------------------

        if env.transition_active[tl]:

            continue


        # ----------------------------------------------------
        # Minimum green duration
        # ----------------------------------------------------

        if (
            env.preemption_active[tl]
            and env.green_elapsed[tl] < MIN_GREEN_DURATION
        ):
            continue

        # ----------------------------------------------------
        # Ambil action MAPPO
        # ----------------------------------------------------

        action_id = get_mappo_action(
            env,
            mappo,
            tl,
            states,
        )
        if action_id is None:
            continue

        if action_id == 0:
            continue

        phase = action(
            env.action_space_marl,
            tl,
            action_id,
        )

        if phase == "HOLD":
            continue

        env.execute_action(
            tl,
            phase,
        )


# ============================================================
# LOAD MAPPO
# ============================================================

def load_mappo_model(
    env,
    initial_states,
):

    state_dim = {
        tl: len(
            initial_states[tl]
        )
        for tl in env.lampu_merah
    }

    action_dim = {
        tl: len(
            env.action_space_marl[tl]
        )
        for tl in env.lampu_merah
    }

    num_agents = len(
        env.lampu_merah
    )


    print(
        "\n=== MAPPO MODEL CONFIGURATION ==="
    )

    print(
        f"Number of agents : {num_agents}"
    )

    for tl in env.lampu_merah:

        print(
            f"{tl:<25} | "
            f"State dim = "
            f"{state_dim[tl]:3d} | "
            f"Action dim = "
            f"{action_dim[tl]:3d}"
        )


    mappo = MAPPO(

        num_agents=num_agents,

        state_dim=state_dim,

        action_dim=action_dim,

        lr_actor=3e-4,

        lr_critic=1e-3,

        gamma=0.99,

        gae_lambda=0.95,

        clip_epsilon=0.2,

        entropy_coef=0.01,

        value_coef=0.5,

        epochs=10,

        device=DEVICE,
    )


    print(
        "\nLoading model:"
    )

    print(
        MODEL_PATH
    )


    checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE,

        weights_only=False,
    )


    for tl in mappo.agent_names:

        mappo.actors[
            tl
        ].load_state_dict(
            checkpoint[
                "actors"
            ][tl]
        )


    mappo.critic.load_state_dict(
        checkpoint["critic"]
    )


    for tl in mappo.agent_names:

        mappo.actors[
            tl
        ].eval()


    mappo.critic.eval()


    print(
        "\nModel berhasil dimuat."
    )


    return mappo


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
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

    row = {

        "method":
            "mappo_nomask",

        "training_episodes":
            400,

        "evaluation_episode":
            1,

        "seed":
            SEED,

        "demand_level":
            DEMAND_LEVEL,

        "jumlah_unit":
            JUMLAH_UNIT,

        "start_datetime":
            start.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "end_datetime":
            end.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "wall_clock_seconds":
            wall_time,

        "simulation_time_seconds":
            simulation_time,

        "total_steps":
            total_steps,

        "active_fire_trucks_at_end":
            len(
                env.active_fire_trucks
            ),

        "arrival_time_upk":
            arrival_upk,

        "waiting_time_upk":
            waiting_upk,
    }


    for tl, value in (
        queue_non_emv.items()
    ):

        row[
            f"queue_length_non_emv_{tl}"
        ] = value


    for tl, value in (
        waiting_non_emv.items()
    ):

        row[
            f"waiting_time_non_emv_{tl}"
        ] = value


    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                row.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(row)


# ============================================================
# SAVE TXT
# ============================================================

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

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "HASIL EVALUASI MAPPO\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            "Metode              : MAPPO\n"
        )

        f.write(
            "Training episode    : 400\n"
        )

        f.write(
            "Evaluation episode  : 1\n"
        )

        f.write(
            "Policy action       : Greedy / Argmax + Action Mask\n"
        )

        f.write(
            "Evaluation loop     : Custom evaluator\n"
        )

        f.write(
            "Seed                : 42\n"
        )

        f.write(
            f"Jam mulai           : "
            f"{start:%Y-%m-%d %H:%M:%S}\n"
        )

        f.write(
            f"Jam selesai         : "
            f"{end:%Y-%m-%d %H:%M:%S}\n"
        )

        f.write(
            f"Wall-clock runtime  : "
            f"{wall_time:.6f} detik\n"
        )

        f.write(
            f"Simulation time     : "
            f"{simulation_time:.2f} detik\n"
        )

        f.write(
            f"Total steps         : "
            f"{total_steps}\n"
        )

        f.write(
            f"Active UPK at end   : "
            f"{len(env.active_fire_trucks)}\n\n"
        )


        f.write(
            "SCENARIO SUMMARY\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(
            f"Jumlah fire event   : "
            f"{len(env.daftar_fire_event)}\n"
        )

        f.write(
            f"Total UPK planned   : "
            f"{sum(e['vehicle_number'] for e in env.daftar_fire_event)}\n"
        )

        f.write(
            f"Total UPK spawned   : "
            f"{len(env._eval_total_waiting)}\n\n"
        )


        f.write(
            "METRIC UPK\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(
            f"Arrival Time UPK    : "
            f"{arrival_upk:.6f} s\n"
        )

        f.write(
            f"Waiting Time UPK    : "
            f"{waiting_upk:.6f} s\n\n"
        )


        f.write(
            "WAITING TIME PER FIRE TRUCK\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        for vehicle_id in sorted(
            env._eval_total_waiting
        ):

            f.write(
                f"{vehicle_id:<25} : "
                f"{env._eval_total_waiting[vehicle_id]:.6f} s\n"
            )


        f.write(
            "\nARRIVAL TIME PER FIRE TRUCK\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        for vehicle_id in sorted(
            env.buat_avg_arrival_time_upk
        ):

            f.write(
                f"{vehicle_id:<25} : "
                f"{env.buat_avg_arrival_time_upk[vehicle_id]:.6f} s\n"
            )


        f.write(
            "\nQUEUE LENGTH NON-EMV\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        for tl, value in (
            queue_non_emv.items()
        ):

            f.write(
                f"{tl:<30} : "
                f"{value:.6f}\n"
            )


        f.write(
            "\nWAITING TIME NON-EMV\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        for tl, value in (
            waiting_non_emv.items()
        ):

            f.write(
                f"{tl:<30} : "
                f"{value:.6f}\n"
            )


        f.write(
            "\n"
            + "=" * 80
            + "\n"
        )


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation():

    prepare_result_folder()

    start_datetime = datetime.now()

    wall_start = time.time()


    print(
        "\n"
        + "=" * 80
    )

    print(
        "EVALUASI MAPPO"
    )

    print(
        "=" * 80
    )

    print(
        f"Model       : {MODEL_PATH}"
    )

    print(
        "Episode     : 1"
    )

    print(
        f"Simulation  : "
        f"{MAX_SIMULATION_TIME} detik"
    )

    print(
        f"Seed        : {SEED}"
    )

    print(
        "=" * 80
    )


    env = TrafficEnvironment(demand_level=DEMAND_LEVEL, jumlah_unit=JUMLAH_UNIT)


    try:

        # ----------------------------------------------------
        # Reset hanya sekali
        # ----------------------------------------------------

        states, info = env.reset(
            seed=SEED
        )


        # ----------------------------------------------------
        # Evaluation storage
        # ----------------------------------------------------

        env._eval_previous_waiting = {}

        env._eval_total_waiting = {}


        # ----------------------------------------------------
        # Load trained MAPPO
        # ----------------------------------------------------

        mappo = load_mappo_model(
            env,
            states,
        )


        steps = 0


        # ----------------------------------------------------
        # Simulation loop
        # ----------------------------------------------------

        while (
            traci.simulation.getTime()
            < MAX_SIMULATION_TIME
        ):


            # 1. Spawn fire event
            process_fire_events(
                env
            )


            # 2. MAPPO control
            apply_mappo_actions(
                env,
                mappo,
                states,
            )


            # 3. One SUMO second
            env.simulation_step()


            # 4. Record waiting time
            record_waiting_time_upk(
                env
            )


            # 5. Arrival / removal
            update_fire_trucks(
                env
            )


            # 6. Non-EMV metrics
            collect_non_emv_metrics(
                env
            )


            # 7. Next state
            states = state(
                env.lampu_merah
            )


            steps += 1


            simulation_time = (
                traci.simulation.getTime()
            )


            if steps % 100 == 0:

                print(
                    f"Step = {steps:5d} | "
                    f"Simulation time = "
                    f"{simulation_time:8.1f} s | "
                    f"Active UPK = "
                    f"{len(env.active_fire_trucks):3d}"
                )


        # ----------------------------------------------------
        # Finish
        # ----------------------------------------------------

        end_datetime = datetime.now()

        wall_time = (
            time.time()
            - wall_start
        )

        simulation_time = (
            traci.simulation.getTime()
        )


        env.buat_avg_waiting_time_upk = dict(
            env._eval_total_waiting
        )


        arrival_upk = arrivalTimeUPK(
            env.buat_avg_arrival_time_upk
        )


        waiting_upk = waitingTimeUPK(
            env.buat_avg_waiting_time_upk
        )


        queue_non_emv = queueLengthNonEMV(
            env.buat_avg_queue_length_non_upk
        )


        waiting_non_emv = waitingTimeNonEMV(
            env.buat_avg_waiting_time_non_upk
        )


        csv_path, txt_path = (
            make_result_files(
                start_datetime
            )
        )


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
            waiting_non_emv,
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
            waiting_non_emv,
        )


        print(
            "\n"
            + "=" * 80
        )

        print(
            "EVALUASI MAPPO SELESAI"
        )

        print(
            "=" * 80
        )

        print(
            f"Simulation time  : "
            f"{simulation_time:.2f} s"
        )

        print(
            f"Total steps      : "
            f"{steps}"
        )

        print(
            f"Total UPK spawned: "
            f"{len(env._eval_total_waiting)}"
        )

        print(
            f"Active UPK end   : "
            f"{len(env.active_fire_trucks)}"
        )

        print(
            f"Arrival Time UPK : "
            f"{arrival_upk:.4f} s"
        )

        print(
            f"Waiting Time UPK : "
            f"{waiting_upk:.4f} s"
        )

        print(
            f"CSV              : "
            f"{csv_path}"
        )

        print(
            f"TXT              : "
            f"{txt_path}"
        )

        print(
            "=" * 80
        )


    finally:

        env.close()

        print(
            "\nEnvironment ditutup."
        )


if __name__ == "__main__":

    run_evaluation()