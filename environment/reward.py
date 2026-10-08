import libsumo as traci
import numpy as np

Tmax = 45
Dmin = 50
MAX_STOP_COUNT = 5.0
MAX_QUEUE = 20.0
MAX_WAITING = 300.0
w1 = -5
w2 = -5
w3 = -1
w4 = -0.5
w5 = -0.5
w6 = -0.5
w7 = -0.2
w8 = -0.5
w9 = -0.2

# fungsi untuk mengurutkan unit pemadam kebakaran berdasarkan phase atau arah yang sama
def kelompokan_arah_unit(list_upk):
    kelompok = {}
    for upk in list_upk:
        for phase in upk["phase_upk"]:
            if phase not in kelompok:
                kelompok[phase] = []
            kelompok[phase].append(upk)

    return kelompok

# fungsi untuk menghitung dari list unit pemadam kebakaran di suatu lampu merah, mana unit yang paling dekat dengan lampu merah dari setiap phase
def ambil_yang_paling_dekat(list_upk):
    kelompok = kelompokan_arah_unit(list_upk)
    list_upk_terdekat = {}

    for phase, daftar_upk_per_phase in kelompok.items():
        upk_terdekat = min(daftar_upk_per_phase, key=lambda upk: upk["jarak_ke_intersection"])
        list_upk_terdekat[phase] = upk_terdekat

    return list_upk_terdekat

# 9 komponen bikin 9 fungsi
# unit pemadam kebakaran tidak boleh lebih lama dari Tmax
def c1(upk):
    waktu_upk = upk["stay_time_upk"]
    lebih_dari_tmax = 1
    kurang_dari_tmax = 0

    if waktu_upk > Tmax:
        return lebih_dari_tmax
    else:
        return kurang_dari_tmax

# jarak unit pemadam kebakaran tidak boleh lebih dari Dmin dan kondisinya belum hijau 
def c2(upk):
    jarak_upk_stopline = upk["jarak_ke_stopline"]
    phase_yang_ada_upk = upk["phase_upk"]
    phase_sedang_aktif = upk["phase_aktif"]

    terpenuhi = 0
    tidak_terpenuhi = 1

    if jarak_upk_stopline < Dmin and phase_sedang_aktif not in phase_yang_ada_upk:
        return tidak_terpenuhi
    else:
        return terpenuhi

# seberapa lama unit pemadam kebakaran ada di lampu merah
def TUPK(upk):
    return upk["stay_time_upk"] / Tmax

# stop number dari unit pemadam kebakaran dari awal muncul hingga di lampu merah tersebut
def SNUPK(upk):
    return upk["jumlah_stop_upk"] / MAX_STOP_COUNT

# delay dari unit pemadam kebakaran yang dhitung dari 1-(speed dari unit pemadam kebakaran sekarang/speed limit dari unit pemadam kebakaran)
def DUPK(upk):
    speed_upk = upk["speed_upk"]
    speed_limit_upk = upk["speed_limit_upk"]

    if speed_limit_upk <= 0:
        return 0

    return 1 - (speed_upk / speed_limit_upk)

# queue length dari arah/phase yang ada unit pemadam kebakarannya
def QUPK(upk, info_setiap_lane):
    phase_upk = upk["phase_upk"]
    total_queue = 0

    for lane, info in info_setiap_lane.items():
        phase_lane = info["phase_per_lane"]
        if any(
            phase in phase_upk
            for phase in phase_lane
        ):
            total_queue += info["queue_per_lane"]

    return total_queue / MAX_QUEUE

# queue length dari arah/phase yang tidak ada unit pemadam kebakarannya
def QnonUPK(upk, info_setiap_lane):
    phase_upk = upk["phase_upk"]
    total_queue = 0

    for lane, info in info_setiap_lane.items():
        phase_lane = info["phase_per_lane"]
        if not any(phase in phase_upk
            for phase in phase_lane
        ):
            total_queue += info["queue_per_lane"]

    return total_queue / MAX_QUEUE

# total waiting time kendaraan dari arah/phase yang ada unit pemadam kebakarannya
def WUPK(upk, info_setiap_lane):
    phase_upk = upk["phase_upk"]
    total_waiting_time = 0

    for lane, info in info_setiap_lane.items():
        phase_lane = info["phase_per_lane"]

        if any(
            phase in phase_upk
            for phase in phase_lane
        ):
            total_waiting_time += info["waiting_time_per_lane"]

    return total_waiting_time / MAX_WAITING

# waiting time yang dialami oleh kendaraan di phase yang tidak ada unit pemadam kebakarannya di lampu merah tersebut
def WnonUPK(upk, info_setiap_lane):
    phase_upk = upk["phase_upk"]
    total_waiting_time = 0

    for lane, info in info_setiap_lane.items():
        phase_lane = info["phase_per_lane"]

        if not any(
            phase in phase_upk
            for phase in phase_lane
        ):
            total_waiting_time += info["waiting_time_per_lane"]

    return total_waiting_time / MAX_WAITING

# fungsi untuk menghitung akumulasi dari 9 reward sebelumnya
def calculate_components(upk, info_setiap_lane):
    return {
        "C1": c1(upk),
        "C2": c2(upk),
        "TUPK": TUPK(upk),
        "SNUPK": SNUPK(upk),
        "DUPK": DUPK(upk),
        "QUPK": QUPK(upk, info_setiap_lane),
        "QnonUPK": QnonUPK(upk, info_setiap_lane),
        "WUPK": WUPK(upk, info_setiap_lane),
        "WnonUPK": WnonUPK(upk, info_setiap_lane)
    }

#fungsi untuk menentukan komponen reward maksimal
def max_components(component_list):
    if not component_list:
        return {}

    max_values = {}

    for component in component_list[0]:
        max_values[component] = max(
            components[component]
            for components in component_list
        )

    return max_values

# recal schema
def recal_schema(list_upk, info_setiap_lane):
    if not list_upk:
        return None

    kelompok = kelompokan_arah_unit(list_upk)

    if not kelompok:
        return None

    list_upk_terdekat = {}

    for phase, daftar_upk_per_phase in kelompok.items():
        upk_terdekat = min(daftar_upk_per_phase, key=lambda upk: upk["jarak_ke_intersection"])
        list_upk_terdekat[phase] = upk_terdekat

    component_list = []

    for upk in list_upk_terdekat.values():
        components = calculate_components(upk, info_setiap_lane)
        component_list.append(components)

    if not component_list:
        return None

    max_component = max_components(component_list)

    return max_component

# reward inti
def reward(reward_info):
    all_rewards = {}

    for tl, info in reward_info.items():
        info_setiap_lane = info["info_lane"]
        list_upk = []

        for lane, info_lane in info_setiap_lane.items():
            list_upk.extend(info_lane["upk"])

        if not list_upk:
            all_rewards[tl] = 0
            continue

        max_component = recal_schema(list_upk, info_setiap_lane)

        if max_component is None:
            all_rewards[tl] = 0
            continue

        reward_value = (
            w1 * max_component["C1"] +
            w2 * max_component["C2"] +
            w3 * max_component["TUPK"] +
            w4 * max_component["SNUPK"] +
            w5 * max_component["DUPK"] +
            w6 * max_component["QUPK"] +
            w7 * max_component["QnonUPK"] +
            w8 * max_component["WUPK"] +
            w9 * max_component["WnonUPK"]
        )

        all_rewards[tl] = reward_value

    return all_rewards