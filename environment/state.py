import libsumo as traci
import numpy as np
from . import tls_programs

MAX_GREEN_PHASES = 9
MAX_EMV_COUNT = 5.0
MAX_DISTANCE = 500.0
MAX_WAITING = 300.0
MAX_NON_EMV_QUEUE = 50.0
MAX_NON_EMV_WAITING = 300.0
MAX_REMAINING_PHASE = 60.0


def get_green_phases(tl):
    # Program asli (tls_programs), konsisten dengan action.get_green_phases
    return tls_programs.get_green_phase_indices(tl, "G")

def get_emv_target_phases(vehicle_id, tl, green_phases):
    next_tls = traci.vehicle.getNextTLS(vehicle_id)
    phase_states = tls_programs.get_phase_states(tl)

    for tls_id, link_index, distance, signal_state in next_tls:
        if tls_id != tl:
            continue

        return [
            phase_index
            for phase_index in green_phases
            if (
                0 <= link_index < len(phase_states[phase_index])
                and phase_states[phase_index][link_index] == "G"
            )
        ]

    return []

def get_lane_phase_mapping(tl):
    incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))
    controlled_links = (traci.trafficlight.getControlledLinks(tl))
    signal_to_lanes = {}

    for signal_index, links in enumerate(controlled_links):
        for link in links:
            if not link:
                continue

            incoming_lane = link[0]
            signal_to_lanes.setdefault(signal_index, set()).add(incoming_lane)

    lane_phase_mapping = {lane: [] for lane in incoming_lanes}

    for phase in tls_programs.get_phase_information(tl):
        phase_index = phase["index"]
        phase_state = phase["state"]

        for signal_index, lanes in signal_to_lanes.items():
            if signal_index >= len(phase_state):
                continue

            if phase_state[signal_index] != "G":
                continue

            for lane in lanes:
                if lane not in lane_phase_mapping:
                    lane_phase_mapping[lane] = []

                if phase_index not in lane_phase_mapping[lane]:
                    lane_phase_mapping[lane].append(phase_index)

    return lane_phase_mapping

# def get_phase_emv_information(tl):
#     green_phases = get_green_phases(tl)

#     phase_info = {
#         phase: {
#             "count": 0,
#             "nearest_distance": MAX_DISTANCE,
#             "waiting": 0.0,
#         }
#         for phase in green_phases
#     }

#     lane_phase_mapping = get_lane_phase_mapping(tl)
#     incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

#     for lane in incoming_lanes:
#         phases = get_emv_target_phases(vehicle_id, tl, green_phases)

#         if not phases:
#             continue

#         vehicle_ids = (traci.lane.getLastStepVehicleIDs(lane))
#         lane_length = float(traci.lane.getLength(lane)        )

#         for vehicle_id in vehicle_ids:
#             try:

#                 if (traci.vehicle.getVehicleClass(vehicle_id)!= "emergency"):
#                     continue

#                 lane_position = float(traci.vehicle.getLanePosition(vehicle_id))
#                 distance = max(lane_length - lane_position,0.0)
#                 waiting = float(traci.vehicle.getWaitingTime(vehicle_id))

#                 for phase in phases:
#                     if phase not in phase_info:
#                         continue
#                     phase_info[phase]["count"] += 1
#                     phase_info[phase]["nearest_distance"] = min(phase_info[phase]["nearest_distance"], distance)
#                     phase_info[phase]["waiting"] += waiting

#             except Exception:
#                 continue

#     return phase_info

def get_phase_emv_information(tl):
    green_phases = get_green_phases(tl)

    phase_info = {
        phase: {
            "count": 0,
            "nearest_distance": MAX_DISTANCE,
            "waiting": 0.0,
        }
        for phase in green_phases
    }

    incoming_lanes = list(
        dict.fromkeys(traci.trafficlight.getControlledLanes(tl))
    )

    for lane in incoming_lanes:
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
        lane_length = float(traci.lane.getLength(lane))

        for vehicle_id in vehicle_ids:
            try:
                if traci.vehicle.getVehicleClass(vehicle_id) != "emergency":
                    continue

                phases = get_emv_target_phases(
                    vehicle_id, tl, green_phases
                )

                if not phases:
                    continue

                lane_position = float(
                    traci.vehicle.getLanePosition(vehicle_id)
                )

                distance = max(lane_length - lane_position, 0.0)
                waiting = float(
                    traci.vehicle.getWaitingTime(vehicle_id)
                )

                for phase in phases:
                    phase_info[phase]["count"] += 1
                    phase_info[phase]["nearest_distance"] = min(
                        phase_info[phase]["nearest_distance"],
                        distance
                    )
                    phase_info[phase]["waiting"] += waiting

            except traci.TraCIException:
                continue

    return phase_info

def get_non_emv_information(tl):
    queue = 0
    waiting = 0.0

    incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

    for lane in incoming_lanes:
        vehicle_ids = (traci.lane.getLastStepVehicleIDs(lane))

        for vehicle_id in vehicle_ids:
            try:
                if (traci.vehicle.getVehicleClass(vehicle_id)== "emergency"):
                    continue

                speed = float(traci.vehicle.getSpeed(vehicle_id))

                if speed < 0.1:
                    queue += 1

                waiting += float(traci.vehicle.getWaitingTime(vehicle_id))

            except Exception:
                continue

    return queue, waiting


def state(tls):
    all_states = {}

    for tl in tls:
        green_phases = get_green_phases(tl)
        phase_info = get_phase_emv_information(tl)

        state_values = []

        for phase_slot in range(MAX_GREEN_PHASES):
            if phase_slot < len(green_phases):
                sumo_phase = green_phases[phase_slot]
                info = phase_info.get(sumo_phase,
                    {
                        "count": 0,
                        "nearest_distance": MAX_DISTANCE,
                        "waiting": 0.0,
                    }
                )

                emv_count = min(info["count"],MAX_EMV_COUNT) / MAX_EMV_COUNT

                if info["count"] > 0:
                    nearest_distance = min(info["nearest_distance"],MAX_DISTANCE) / MAX_DISTANCE
                else:
                    nearest_distance = 1.0

                emv_waiting = min(info["waiting"],MAX_WAITING) / MAX_WAITING

            else:
                emv_count = 0.0
                nearest_distance = 1.0
                emv_waiting = 0.0

            state_values.extend([emv_count,nearest_distance,emv_waiting,])

        # -1 selama transisi (program 'online'), sehingga bukan dianggap fase hijau
        current_phase = tls_programs.get_current_phase(tl)
        current_phase_one_hot = [0.0
            for _ in range(MAX_GREEN_PHASES)
        ]

        if current_phase in green_phases:
            rl_phase = green_phases.index(current_phase)

            if rl_phase < MAX_GREEN_PHASES:
                current_phase_one_hot[rl_phase] = 1.0

        state_values.extend(current_phase_one_hot)
        is_transition = (0.0 if current_phase in green_phases else 1.0)

        state_values.append(is_transition)

        if tls_programs.is_original_program_active(tl):
            remaining = (traci.trafficlight.getNextSwitch(tl) - traci.simulation.getTime())
        else:
            remaining = 0.0
        remaining = min(max(float(remaining), 0.0), MAX_REMAINING_PHASE)
        state_values.append(remaining / MAX_REMAINING_PHASE)

        non_emv_queue, non_emv_waiting = (get_non_emv_information(tl))
        state_values.append(min(non_emv_queue, MAX_NON_EMV_QUEUE) / MAX_NON_EMV_QUEUE)
        state_values.append(min(non_emv_waiting, MAX_NON_EMV_WAITING) / MAX_NON_EMV_WAITING)
        
        all_states[tl] = np.asarray(state_values, dtype=np.float32)

    return all_states


def get_lane_traffic_info(tl, lane_phase_mapping):
    incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))
    lane_traffic = {}

    for lane in incoming_lanes:
        vehicle_ids = list(traci.lane.getLastStepVehicleIDs(lane))
        queue = 0
        waiting = 0.0

        for vehicle_id in vehicle_ids:
            try:
                speed = float(traci.vehicle.getSpeed(vehicle_id))

                if speed < 0.1:
                    queue += 1

                waiting += float(traci.vehicle.getWaitingTime(vehicle_id))

            except Exception:
                continue

        lane_traffic[lane] = {
            "phase_per_lane": lane_phase_mapping.get(lane, []),
            "queue_per_lane": queue,
            "waiting_time_per_lane": waiting,
            "vehicle_ids": vehicle_ids,
        }

    return lane_traffic


def build_upk_list(tl, lane, vehicle_ids, lane_phase_mapping, phase_aktif, current_time, emv_entry_time, emv_stop_count,):
    green_phases = get_green_phases(tl)
    upk_list = []
    lane_length = float(traci.lane.getLength(lane))
    speed_limit = float(traci.lane.getMaxSpeed(lane))
    intersection_position = traci.lane.getShape(lane)[-1]

    for vehicle_id in vehicle_ids:
        try:
            if (traci.vehicle.getVehicleClass(vehicle_id) != "emergency"):
                continue

            phases = get_emv_target_phases(vehicle_id, tl, green_phases)

            if not phases:
                continue

            lane_position = float(traci.vehicle.getLanePosition(vehicle_id))
            jarak_ke_stopline = max(lane_length - lane_position, 0.0)
            veh_pos = traci.vehicle.getPosition(vehicle_id)

            jarak_ke_intersection = float(np.linalg.norm(np.array(veh_pos) - np.array(intersection_position)))
            speed = float(traci.vehicle.getSpeed(vehicle_id))
            key = (vehicle_id, tl)
            stay_time_upk = current_time - emv_entry_time.get(key, current_time)
            jumlah_stop_upk = emv_stop_count.get(key, 0)

            upk_list.append({
                "vehicle_id": vehicle_id,
                "phase_upk": phases,
                "phase_aktif": phase_aktif,
                "jarak_ke_intersection": jarak_ke_intersection,
                "jarak_ke_stopline": jarak_ke_stopline,
                "stay_time_upk": stay_time_upk,
                "jumlah_stop_upk": jumlah_stop_upk,
                "speed_upk": speed,
                "speed_limit_upk": speed_limit,
            })

        except Exception:
            continue

    return upk_list


def build_reward_info(tls, emv_entry_time, emv_stop_count):
    current_time = traci.simulation.getTime()
    reward_info = {}

    for tl in tls:
        lane_phase_mapping = get_lane_phase_mapping(tl)
        phase_aktif = tls_programs.get_current_phase(tl)

        lane_traffic = get_lane_traffic_info(tl, lane_phase_mapping)
        info_lane = {}

        for lane, data in lane_traffic.items():
            upk_list = build_upk_list(tl, lane, data["vehicle_ids"], lane_phase_mapping, phase_aktif, current_time, emv_entry_time, emv_stop_count,)

            info_lane[lane] = {
                "phase_per_lane": data["phase_per_lane"],
                "queue_per_lane": data["queue_per_lane"],
                "waiting_time_per_lane": data["waiting_time_per_lane"],
                "upk": upk_list,
            }

        reward_info[tl] = {"info_lane": info_lane}

    return reward_info