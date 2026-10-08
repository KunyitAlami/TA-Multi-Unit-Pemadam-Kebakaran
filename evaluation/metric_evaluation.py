import libsumo as traci
import numpy as np

def arrivalTimeUPK(buat_avg_arrival_time_upk):
    if not buat_avg_arrival_time_upk:
        return 0

    return sum(buat_avg_arrival_time_upk.values()) / len(buat_avg_arrival_time_upk)

def waitingTimeUPK(buat_avg_waiting_time_upk):
    if not buat_avg_waiting_time_upk:
        return 0

    return sum(buat_avg_waiting_time_upk.values()) / len(buat_avg_waiting_time_upk)

def queueLengthNonEMV(data_queue):
    rata_rata_queue_per_tl = {}

    for tl, data_queue_tl in data_queue.items():
        if len(data_queue_tl) > 0:
            rata_rata_queue_per_tl[tl] = np.mean(data_queue_tl)
        else:
            rata_rata_queue_per_tl[tl] = 0

    return rata_rata_queue_per_tl

def waitingTimeNonEMV(data_waiting):
    rata_rata_waiting_per_tl = {}

    for tl, data_waiting_tl in data_waiting.items():
        if len(data_waiting_tl) > 0:
            rata_rata_waiting_per_tl[tl] = np.mean(data_waiting_tl)
        else:
            rata_rata_waiting_per_tl[tl] = 0

    return rata_rata_waiting_per_tl

def travelTimeNonEMV(buat_avg_travel_time_non_upk):
    if not buat_avg_travel_time_non_upk:
        return 0

    return sum(buat_avg_travel_time_non_upk.values()) / len(buat_avg_travel_time_non_upk)



