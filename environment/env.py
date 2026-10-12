import math

import libsumo as traci
import numpy as np
import gymnasium as gym

from . import tls_programs
from .state import (
    state,
    build_reward_info,
    get_green_phases,
    get_emv_target_phases
)
from .reward import reward
from .action import build_action_space, action

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DECISION_INTERVAL = 5     
MIN_GREEN_DURATION = 5    
YELLOW_DURATION = 1.5      
ALL_RED_DURATION = 1       
HOLD_DURATION = 5     
STOP_SPEED_THRESHOLD = 0.1

DEMAND_FILES = {
    "low": str(PROJECT_ROOT / "sumo" / "demand" / "demand_low.rou.xml"),
    "normal": str(PROJECT_ROOT / "sumo" / "demand" / "demand_normal.rou.xml"),
    "high": str(PROJECT_ROOT / "sumo" / "demand" / "demand_high.rou.xml"),
}

class TrafficEnvironment(gym.Env):

    def __init__(self, demand_level=None, jumlah_unit=None):
        self.sumo_cfg = str(PROJECT_ROOT / "sumo" / "banjarmasintengah.sumocfg")

        if demand_level is not None and demand_level not in DEMAND_FILES:
            raise ValueError(
                f"demand_level tidak dikenal: {demand_level!r}. "
                f"Pilihan valid: {list(DEMAND_FILES.keys())}"
            )

        if jumlah_unit is not None and jumlah_unit < 1:
            raise ValueError(
                f"jumlah_unit harus >= 1, diberikan: {jumlah_unit!r}"
            )

        self.demand_level = demand_level
        self.jumlah_unit = jumlah_unit
        self.fire_events = []
        self.active_fire_trucks = []
        self.counter = 1
        self._sumo_started = False
        self.lampu_merah = ("belitung", "jembatanmerdeka", "kamboja", "mahaji", "mentari", "mulawarman", "pasarlama", "pdam", "simpang3kuripan", "simpang3sugiono", "simpang4antasari", "tarakan")
        self.action_space_marl = None

        self.buat_avg_arrival_time_upk = {}
        self.buat_avg_waiting_time_upk = {}
        self.buat_avg_queue_length_non_upk = {}
        self.buat_avg_waiting_time_non_upk = {}

        self._emv_entry_time = {}
        self._emv_stop_count = {}
        self._emv_prev_speed = {}
        self._non_emv_depart_time = {}
        self.buat_avg_travel_time_non_upk = {}

        self.original_logic = {}
        self.original_program_id = {}
        self.dt = 1.0
        self.steps_per_decision = DECISION_INTERVAL
        self.transition_log = []
        self.transition_violations = []
        self.transition_validation = {}
        self._init_controller_state()

    def _init_controller_state(self):
        self.remaining_duration = {tl: 0 for tl in self.lampu_merah}
        self.green_elapsed = {tl: 0 for tl in self.lampu_merah}
        self.active_phase = {tl: None for tl in self.lampu_merah}
        self.preemption_active = {tl: False for tl in self.lampu_merah}
        self.hold_active = {tl: False for tl in self.lampu_merah}
        self.transition_phase = {tl: None for tl in self.lampu_merah}
        self.transition_active = {tl: False for tl in self.lampu_merah}

    def _store_original_programs(self):
        """Simpan tlLogic asli di tls_programs (dipakai juga oleh state.py dan
        action.py). setRedYellowGreenState() memindahkan TLS ke program 'online',
        jadi program asli harus dipulihkan sebelum setPhase() dipakai lagi."""
        tls_programs.store_original_programs(self.lampu_merah)

        self.original_logic = {tl: tls_programs.get_logic(tl) for tl in self.lampu_merah}
        self.original_program_id = {tl: tls_programs.get_program_id(tl) for tl in self.lampu_merah}

    def _get_logic(self, tl):
        return tls_programs.get_logic(tl)

    # ------------------------------------------------------------------
    # Utilitas state sinyal
    # ------------------------------------------------------------------
    @staticmethod
    def _yellow_state(state_string):
        """Hijau (G/g) -> kuning. Kuning yang sudah ada tetap kuning.
        'u' (merah-kuning, menuju hijau) -> merah karena kita sedang meninggalkan."""
        out = []
        for s in state_string:
            if s in "Gg":
                out.append("y")
            elif s == "u":
                out.append("r")
            else:
                out.append(s)
        return "".join(out)

    @staticmethod
    def _red_state(state_string):
        """Semua sinyal terkontrol -> merah. 's' dan 'O'/'o' (tidak dikontrol) dibiarkan."""
        return "".join(s if s in "sOo" else "r" for s in state_string)

    def _steps_for(self, duration):
        """Durasi (detik) dibulatkan ke atas ke kelipatan step simulasi."""
        if duration <= 0:
            return 0.0
        return math.ceil(duration / self.dt - 1e-9) * self.dt

    def validate_transition_setup(self):
        """Validasi statis tiap TLS: panjang state, fase hijau tersedia, hasil
        konversi kuning tidak memuat hijau dari SEMUA fase awal (termasuk fase
        yang sudah kuning/merah), dan durasi sesuai step simulasi."""
        errors = []
        warnings = []
        report = {}

        for name, value in (("YELLOW_DURATION", YELLOW_DURATION), ("ALL_RED_DURATION", ALL_RED_DURATION)):
            if value < 0:
                errors.append(f"{name} tidak boleh negatif ({value})")
            elif value > 0 and abs(self._steps_for(value) - value) > 1e-6:
                warnings.append(
                    f"{name}={value}s bukan kelipatan step simulasi ({self.dt}s); "
                    f"efektif {self._steps_for(value)}s"
                )

        if abs(DECISION_INTERVAL / self.dt - round(DECISION_INTERVAL / self.dt)) > 1e-6:
            warnings.append(
                f"DECISION_INTERVAL={DECISION_INTERVAL}s bukan kelipatan step ({self.dt}s); "
                f"dipakai {self.steps_per_decision} step"
            )

        for tl in self.lampu_merah:
            phases = tls_programs.get_phase_information(tl)
            n_signals = len(traci.trafficlight.getRedYellowGreenState(tl))
            tl_errors = []

            if not phases:
                errors.append(f"[{tl}] tidak ada fase")
                continue

            green_idx = [p["index"] for p in phases if any(s in "Gg" for s in p["state"])]
            starts_green = []
            starts_yellow = []
            starts_red = []

            if not green_idx:
                tl_errors.append("tidak ada fase hijau")

            for p in phases:
                st = p["state"]

                if len(st) != n_signals:
                    tl_errors.append(f"fase {p['index']}: panjang state {len(st)} != {n_signals}")
                    continue

                if any(s in "Gg" for s in self._yellow_state(st)):
                    tl_errors.append(f"fase {p['index']}: konversi kuning masih memuat hijau")

                if any(s in self._red_state(st) for s in "Ggyu"):
                    tl_errors.append(f"fase {p['index']}: konversi all-red masih memuat hijau/kuning")

                if any(s in "Gg" for s in st):
                    starts_green.append(p["index"])
                elif "y" in st:
                    starts_yellow.append(p["index"])
                else:
                    starts_red.append(p["index"])

            mixed = [
                p["index"] for p in phases
                if any(s in "Gg" for s in p["state"]) and "y" in p["state"]
            ]
            if mixed:
                warnings.append(
                    f"[{tl}] fase campuran hijau+kuning {mixed}: kuning baru memakai "
                    f"YELLOW_DURATION, kuning yang sudah ada ikut diperpanjang"
                )

            for e in tl_errors:
                errors.append(f"[{tl}] {e}")

            report[tl] = {
                "n_signals": n_signals,
                "green_phases": green_idx,
                "start_from_green": starts_green,
                "start_from_yellow": starts_yellow,
                "start_from_red": starts_red,
                "errors": tl_errors,
            }

        self.transition_validation = {"per_tls": report, "warnings": warnings, "errors": errors}

        for w in warnings:
            print(f"[validasi transisi] PERINGATAN: {w}")

        if errors:
            raise RuntimeError("Validasi transisi gagal:\n  " + "\n  ".join(errors))

        return self.transition_validation

    def spawnFireTruck(self, entry_edge, destination_edge, route_id, vehicle_id):
        route = traci.simulation.findRoute(entry_edge, destination_edge, vType="firetruck")

        if not route.edges:
            # tidak ada jalur valid untuk kombinasi ini, batalkan spawn dengan aman
            return False
        
        traci.route.add(routeID=route_id, edges=route.edges)
        traci.vehicle.add(routeID=route_id, vehID=vehicle_id, typeID="firetruck")
        traci.vehicle.setColor(vehicle_id, (255, 0, 0))
        return True

    def track_non_emv_travel_time(self):
        current_time = traci.simulation.getTime()

        for veh_id in traci.simulation.getDepartedIDList():
            if traci.vehicle.getVehicleClass(veh_id) != "emergency":
                self._non_emv_depart_time[veh_id] = current_time

        for veh_id in traci.simulation.getArrivedIDList():
            if veh_id in self._non_emv_depart_time:
                self.buat_avg_travel_time_non_upk[veh_id] = current_time - self._non_emv_depart_time.pop(veh_id)

    def updateFireTruck(self, vehicle_id, destination_edge, waktu_spawn):
        if vehicle_id not in traci.vehicle.getIDList():
            return False
        if traci.vehicle.getRoadID(vehicle_id) == destination_edge:
            # ini arrival time
            current_time = traci.simulation.getTime()
            data_waktu_upk = current_time - waktu_spawn
            self.buat_avg_arrival_time_upk[vehicle_id] = data_waktu_upk

            # ini waiting time
            waiting_time_upk = traci.vehicle.getWaitingTime(vehicle_id)

            self.buat_avg_arrival_time_upk[vehicle_id] = data_waktu_upk
            self.buat_avg_waiting_time_upk[vehicle_id] = waiting_time_upk

            traci.vehicle.remove(vehicle_id)
            return False
        return True

    def sample_jumlah_unit_training(self):
        return int(np.clip(np.random.poisson(6), 1, 12))

    def generateFireEvent(self):
        daftar_entry_edge = ["298510828#0", "712902521#3", "296040723#6", "221685538#0", "1120959902#2", "221685540#2", "792678882#7", "802880890#1", "355845180#1", "-558123018.172", "1113586959#1", "1210663840#0", "552778605#1", "289701307#13", "802880887#16"]

        daftar_destination_edge = ["703025984#1", "1295981206#2", "558907723#2", "-549263053#19", "-549263053#29", "289701307#1", "-773819408#3", "-783268472#3", "726170083#1", "726170083#4", "48242727#2", "355654010#3", "-355654010#5", "221686414#0", "-709268561#6", "-813551429"]

        waktu_kebakaran = int(np.random.randint(100, 2900))
        jumlah_unit_event = self.jumlah_unit if self.jumlah_unit is not None else self.sample_jumlah_unit_training()
        destination_edge = str(np.random.choice(daftar_destination_edge))

        entry_edges = [str(x) for x in np.random.choice(daftar_entry_edge, size=jumlah_unit_event, replace=False)]

        fireEvents = [{
            "entry_edges": [str(entry) for entry in entry_edges],
            "destination_edge": destination_edge,
            "vehicle_number": jumlah_unit_event,
            "time_fire_event": waktu_kebakaran,
            "status": False
        }]
        
        return fireEvents

    def cek_ada_upk(self, tls):
        cek_upk_per_tl = {}

        for tl in tls:
            cek_upk_per_tl[tl] = False
            incomingLanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

            for lane in incomingLanes:
                vehicleID = traci.lane.getLastStepVehicleIDs(lane)

                for veh in vehicleID:
                    if traci.vehicle.getVehicleClass(veh) == "emergency":
                        cek_upk_per_tl[tl] = True
                        break

                if cek_upk_per_tl[tl]:
                    break

        return cek_upk_per_tl
    
    def get_upk_per_lane(self, tl):
        upk_per_lane = {}

        incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))
        lane_phase_mapping = self.get_lane_phase_mapping(tl)

        if not incoming_lanes:
            return upk_per_lane

        posisi_intersection = traci.lane.getShape(incoming_lanes[0])[-1]

        for lane in incoming_lanes:
            upk_per_lane[lane] = []
            vehicles = traci.lane.getLastStepVehicleIDs(lane)

            for veh in vehicles:
                if traci.vehicle.getVehicleClass(veh) != "emergency":
                    continue
                veh_pos = traci.vehicle.getPosition(veh)
                distance = np.linalg.norm(np.array(veh_pos) - np.array(posisi_intersection))

                upk_per_lane[lane].append({
                    "vehicle_id": veh,
                    "distance": float(distance),
                    "phases": lane_phase_mapping.get(lane, [])
                })

        return upk_per_lane

    def get_lane_phase_mapping(self, tl):
        incoming_lanes = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(tl)
        ))

        controlled_links = traci.trafficlight.getControlledLinks(tl)
        signal_to_lanes = {}

        for signal_index, links in enumerate(controlled_links):
            for link in links:
                if not link:
                    continue

                incoming_lane = link[0]

                signal_to_lanes.setdefault(
                    signal_index,
                    set()
                ).add(incoming_lane)

        lane_phases = {
            lane: []
            for lane in incoming_lanes
        }

        selected_logic = self._get_logic(tl)

        if selected_logic is None:
            return lane_phases

        for phase_index, phase in enumerate(selected_logic.phases):
            phase_state = phase.state

            for signal_index, lanes in signal_to_lanes.items():
                if signal_index >= len(phase_state):
                    continue

                if phase_state[signal_index] != "G":
                    continue

                for lane in lanes:
                    if lane not in lane_phases:
                        lane_phases[lane] = []

                    if phase_index not in lane_phases[lane]:
                        lane_phases[lane].append(phase_index)

        return lane_phases
    
    def pilih_upk_terdekat(self, tl):
        upk_per_lane = self.get_upk_per_lane(tl)
        semua_upk = []

        for lane, daftar_upk in upk_per_lane.items():
            for upk in daftar_upk:
                upk_data = upk.copy()
                upk_data["lane"] = lane
                semua_upk.append(upk_data)

        if not semua_upk:
            return None

        upk_terdekat = min(
            semua_upk,
            key=lambda x: x["distance"]
        )
        return upk_terdekat
    
    def get_phase_information(self, tl):
        return tls_programs.get_phase_information(tl)

    def get_emv_phases(self, tl):
        """Himpunan fase hijau (program asli) yang dibutuhkan EMV di lane masuk TLS."""
        green_phases = get_green_phases(tl)
        incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

        phases_with_emv = set()

        for lane in incoming_lanes:
            for vehicle_id in traci.lane.getLastStepVehicleIDs(lane):
                if traci.vehicle.getVehicleClass(vehicle_id) != "emergency":
                    continue

                phases = get_emv_target_phases(vehicle_id, tl, green_phases)
                phases_with_emv.update(phases)

        return phases_with_emv

    def get_action_mask(self, tl):
        action_space_tl = self.action_space_marl[tl]
        mask = np.zeros(len(action_space_tl), dtype=np.float32)
        mask[0] = 1.0

        if self.transition_active[tl]:
            return mask

        # Setelah minimum green, MAPPO bebas memilih fase lain yang dibutuhkan EMV
        if (self.preemption_active[tl] and self.green_elapsed[tl] < MIN_GREEN_DURATION):
            return mask

        phases_with_emv = self.get_emv_phases(tl)

        for action_id, phase in action_space_tl.items():
            if action_id != 0 and phase in phases_with_emv:
                mask[action_id] = 1.0

        return mask
    
    def reset(self, seed=None, option=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        try:
            traci.close()
        except Exception:
            pass

        sumo_cmd = [
            "sumo",
            "-c",
            self.sumo_cfg,
            "--start",
        ]

        if self.demand_level is not None:
            sumo_cmd += ["--route-files", DEMAND_FILES[self.demand_level]]

        traci.start(sumo_cmd)

        self.dt = float(traci.simulation.getDeltaT())
        self.steps_per_decision = max(1, int(round(DECISION_INTERVAL / self.dt)))
        self.transition_log = []
        self.transition_violations = []

        self._sumo_started = True
        self.daftar_fire_event = self.generateFireEvent()
        self.active_fire_trucks = []
        self.counter = 1
        self.buat_avg_arrival_time_upk = {}
        self.buat_avg_waiting_time_upk = {}
        self.buat_avg_queue_length_non_upk = {
            tl: []
            for tl in self.lampu_merah
        }
        self.buat_avg_waiting_time_non_upk = {
            tl: []
            for tl in self.lampu_merah
        }

        self._emv_entry_time = {}
        self._emv_stop_count = {}
        self._emv_prev_speed = {}
        self._non_emv_depart_time = {}
        self.buat_avg_travel_time_non_upk = {}

        self._store_original_programs()
        self._init_controller_state()
        self.validate_transition_setup()

        self.action_space_marl = build_action_space(self.lampu_merah)
        observation = state(self.lampu_merah)
        return observation, {}

    def get_lane_upk(self, tl):
        lane_upk = set()
        incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

        for lane in incoming_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)

            for veh in vehicles:
                if traci.vehicle.getVehicleClass(veh) == "emergency":
                    lane_upk.add(lane)
                    break
        return lane_upk
    
    def hitung_queue_length_non_upk(self, tl):
        queue_length_per_tl = 0
        incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

        # Semua lane dihitung; hanya kendaraan EMV yang dikecualikan (per kendaraan)
        for lane in incoming_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)

            for veh in vehicles:
                if traci.vehicle.getVehicleClass(veh) == "emergency":
                    continue

                if traci.vehicle.getSpeed(veh) < 0.1:
                    queue_length_per_tl += 1

        return queue_length_per_tl

    def hitung_waiting_time_non_upk(self, tl):
        waiting_time_per_tl = 0
        incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

        # Semua lane dihitung; hanya kendaraan EMV yang dikecualikan (per kendaraan)
        for lane in incoming_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)

            for veh in vehicles:
                if traci.vehicle.getVehicleClass(veh) == "emergency":
                    continue
                waiting_time_per_tl += traci.vehicle.getWaitingTime(veh)

        return waiting_time_per_tl

    def update_emv_tracking(self):
        current_time = traci.simulation.getTime()
        present_keys = set()

        for tl in self.lampu_merah:
            incoming_lanes = list(dict.fromkeys(
                traci.trafficlight.getControlledLanes(tl)
            ))

            for lane in incoming_lanes:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)

                for vehicle_id in vehicle_ids:
                    try:
                        if traci.vehicle.getVehicleClass(vehicle_id) != "emergency":
                            continue
                    except Exception:
                        continue

                    key = (vehicle_id, tl)
                    present_keys.add(key)

                    speed = float(traci.vehicle.getSpeed(vehicle_id))

                    if key not in self._emv_entry_time:
                        self._emv_entry_time[key] = current_time
                        self._emv_stop_count[key] = 0
                        self._emv_prev_speed[key] = speed
                    else:
                        prev_speed = self._emv_prev_speed[key]

                        if prev_speed >= STOP_SPEED_THRESHOLD and speed < STOP_SPEED_THRESHOLD:
                            self._emv_stop_count[key] += 1

                        self._emv_prev_speed[key] = speed

        stale_keys = [
            key
            for key in self._emv_entry_time
            if key not in present_keys
        ]

        for key in stale_keys:
            del self._emv_entry_time[key]
            del self._emv_stop_count[key]
            del self._emv_prev_speed[key]

    # ------------------------------------------------------------------
    # Pengendali fase: Current Green -> Yellow -> All-Red -> Target Green
    # ------------------------------------------------------------------
    def _activate_green(self, tl, target_phase, hold):
        """Pulihkan program asli lalu aktifkan fase hijau target."""
        tls_programs.restore_original_program(tl)

        traci.trafficlight.setPhase(tl, target_phase)

        if hold:
            traci.trafficlight.setPhaseDuration(tl, HOLD_DURATION)

        transition = self.transition_phase[tl]
        if transition is not None:
            self.transition_log.append({
                "tls": tl,
                "t_start": transition["t_start"],
                "t_green": traci.simulation.getTime(),
                "entry": transition["entry"],
                "from_phase": transition["from_phase"],
                "to_phase": target_phase,
                "yellow_planned": transition["yellow_planned"],
                "yellow_actual": transition["yellow_actual"],
                "all_red_actual": transition["all_red_actual"],
                "violations": transition["violations"],
            })

        self.active_phase[tl] = target_phase
        self.preemption_active[tl] = True
        self.transition_active[tl] = False
        self.transition_phase[tl] = None
        self.hold_active[tl] = hold
        self.green_elapsed[tl] = 0

    def execute_action(self, tl, target_phase, hold=False):
        """Pindah ke fase hijau target dengan clearance.

        hold=True: hijau dipertahankan sampai release_preemption() dipanggil
        (dipakai Greedy). hold=False: perilaku lama (fase berjalan normal
        setelah aktif)."""
        phases = self.get_phase_information(tl)

        if not phases:
            raise RuntimeError(f"Tidak ada phase untuk TLS {tl}")

        valid_phases = {phase["index"] for phase in phases}

        if target_phase not in valid_phases:
            raise RuntimeError(
                f"Target phase tidak valid | TLS={tl} | "
                f"target={target_phase} | valid={sorted(valid_phases)}"
            )

        target_state = phases[target_phase]["state"]
        if not any(s in "Gg" for s in target_state):
            raise RuntimeError(
                f"Target phase bukan fase hijau | TLS={tl} | "
                f"target={target_phase} | state={target_state}"
            )

        if self.transition_active[tl]:
            return

        current_phase = traci.trafficlight.getPhase(tl)
        self.green_elapsed[tl] = 0

        if current_phase == target_phase:
            self.transition_phase[tl] = None
            self._activate_green(tl, target_phase, hold)
            return

        now = traci.simulation.getTime()
        current_state = traci.trafficlight.getRedYellowGreenState(tl)
        has_green = any(s in "Gg" for s in current_state)
        has_yellow = "y" in current_state

        if has_green:
            # Hijau -> kuning (sinyal kuning yang sudah ada ikut tetap kuning)
            entry = "green"
            stage = "yellow"
            planned = self._steps_for(YELLOW_DURATION)
            remaining = planned
            if planned > 0:
                traci.trafficlight.setRedYellowGreenState(tl, self._yellow_state(current_state))
        elif has_yellow:
            # Fase awal sudah kuning: bekukan kuning dan tunggu sisanya, jangan dipotong
            # dan jangan biarkan program asli maju ke fase hijau lain.
            entry = "yellow"
            stage = "yellow"
            planned = max(traci.trafficlight.getNextSwitch(tl) - now, 0.0)
            remaining = planned
            traci.trafficlight.setRedYellowGreenState(tl, self._yellow_state(current_state))
        else:
            entry = "red"
            stage = "all_red"
            planned = 0.0
            remaining = self._steps_for(ALL_RED_DURATION)
            traci.trafficlight.setRedYellowGreenState(tl, self._red_state(current_state))

        self.transition_active[tl] = True
        self.preemption_active[tl] = False
        self.hold_active[tl] = False

        self.transition_phase[tl] = {
            "stage": stage,
            "remaining": remaining,
            "target_phase": target_phase,
            "hold": hold,
            "n_signals": len(current_state),
            "entry": entry,
            "from_phase": current_phase,
            "t_start": now,
            "yellow_planned": planned,
            "yellow_actual": 0.0,
            "all_red_actual": 0.0,
            "violations": 0,
        }

        # Tahap yang durasinya sudah 0 langsung dilewati
        if remaining <= 1e-9:
            self._advance_transition(tl)

    def _advance_transition(self, tl):
        """Pindah ke tahap berikutnya setelah tahap saat ini selesai."""
        transition = self.transition_phase[tl]

        if transition["stage"] == "yellow":
            all_red = self._steps_for(ALL_RED_DURATION)

            if all_red > 1e-9:
                current_state = traci.trafficlight.getRedYellowGreenState(tl)
                traci.trafficlight.setRedYellowGreenState(tl, self._red_state(current_state))
                transition["stage"] = "all_red"
                transition["remaining"] = all_red
                return

        self._activate_green(tl, transition["target_phase"], transition["hold"])

    def release_preemption(self, tl):
        """Lepas preemption. Program asli sudah aktif, jadi cukup akhiri fase
        hijau yang di-hold agar SUMO melanjutkan siklus normal."""
        if self.transition_active[tl]:
            return

        was_active = self.preemption_active[tl]

        self.preemption_active[tl] = False
        self.hold_active[tl] = False
        self.active_phase[tl] = None
        self.green_elapsed[tl] = 0

        if was_active:
            traci.trafficlight.setPhaseDuration(tl, 0)

    def maintain_preemption(self, tl, needed_phases):
        """Recovery bersama (MAPPO dan Greedy). Dipanggil di awal tiap keputusan.

        Hold dipertahankan selama masih ada EMV yang membutuhkan fase hijau
        mana pun. Jika tidak ada EMV tersisa, preemption dilepas dan program
        asli melanjutkan siklus. Pemilihan fase lain tetap di tangan agent
        (MAPPO lewat action mask setelah minimum green, Greedy lewat aturannya)."""
        if self.transition_active[tl]:
            return False

        if not (self.preemption_active[tl] and self.hold_active[tl]):
            return False

        if not needed_phases:
            self.release_preemption(tl)
            return True

        return False

    def simulation_step(self):
        traci.simulationStep()

        for tl in self.lampu_merah:
            if self.transition_active[tl]:
                transition = self.transition_phase[tl]

                # Validasi runtime: tidak boleh ada hijau selama transisi
                if any(s in "Gg" for s in traci.trafficlight.getRedYellowGreenState(tl)):
                    transition["violations"] += 1
                    self.transition_violations.append({
                        "tls": tl,
                        "time": traci.simulation.getTime(),
                        "stage": transition["stage"],
                    })

                if transition["stage"] == "yellow":
                    transition["yellow_actual"] += self.dt
                else:
                    transition["all_red_actual"] += self.dt

                transition["remaining"] -= self.dt

                if transition["remaining"] > 1e-9:
                    continue

                self._advance_transition(tl)

            elif self.preemption_active[tl]:
                self.green_elapsed[tl] += self.dt

                if self.hold_active[tl]:
                    traci.trafficlight.setPhaseDuration(tl, HOLD_DURATION)

    def get_transition_report(self):
        """Ringkasan validasi runtime: durasi aktual per tahap dan pelanggaran hijau."""
        by_tls = {}

        for entry in self.transition_log:
            row = by_tls.setdefault(entry["tls"], {
                "count": 0, "from_green": 0, "from_yellow": 0, "from_red": 0,
                "min_yellow": None, "min_all_red": None, "violations": 0,
            })
            row["count"] += 1
            row[f"from_{entry['entry']}"] += 1
            row["violations"] += entry["violations"]

            if entry["entry"] == "green":
                y = entry["yellow_actual"]
                row["min_yellow"] = y if row["min_yellow"] is None else min(row["min_yellow"], y)

            a = entry["all_red_actual"]
            row["min_all_red"] = a if row["min_all_red"] is None else min(row["min_all_red"], a)

        return {
            "per_tls": by_tls,
            "total_transitions": len(self.transition_log),
            "total_violations": len(self.transition_violations),
        }

    def step(self, action_marl):
        current_time = traci.simulation.getTime()
        for event in self.daftar_fire_event:
            if (not event["status"] and current_time >= event["time_fire_event"]):
                for i in range(event["vehicle_number"]):
                    vehicle_id = f"fire_truck_{self.counter}"
                    route_id = f"route_fire_truck_{self.counter}"

                    spawned = self.spawnFireTruck(
                        entry_edge=event["entry_edges"][i],
                        destination_edge=event["destination_edge"],
                        route_id=route_id,
                        vehicle_id=vehicle_id
                    )

                    if spawned:
                        self.active_fire_trucks.append({
                            "vehicle_id": vehicle_id,
                            "destination_edge": event["destination_edge"],
                            "waktu_spawn": current_time
                        })
                    self.counter += 1

                event["status"] = True

        for tl in self.lampu_merah:
            # Recovery: lepas hold jika tidak ada EMV tersisa (sama dengan Greedy)
            self.maintain_preemption(tl, self.get_emv_phases(tl))

            if self.transition_active[tl]:
                continue

            if (self.preemption_active[tl] and self.green_elapsed[tl] < MIN_GREEN_DURATION):
                continue

            action_id = action_marl[tl]
            if action_id == 0:
                continue

            # Setelah minimum green, MAPPO boleh pindah ke fase lain; hold dipakai
            # sama seperti Greedy agar hijau tidak berakhir sendiri sebelum EMV lewat.
            target_phase = action(self.action_space_marl, tl, action_id)
            self.execute_action(tl, target_phase, hold=True)

        for _ in range(self.steps_per_decision):
            self.simulation_step()
            self.update_emv_tracking()
            self.track_non_emv_travel_time()
            current_time = traci.simulation.getTime()
            truck_yang_masih_aktif = []
            for truck in self.active_fire_trucks:
                masih_aktif = self.updateFireTruck(
                    truck["vehicle_id"],
                    truck["destination_edge"],
                    truck["waktu_spawn"]
                )

                if masih_aktif:
                    truck_yang_masih_aktif.append(
                        truck
                    )

            self.active_fire_trucks = (truck_yang_masih_aktif)
            for tl in self.lampu_merah:
                nilai_queue_length_per_tl = (self.hitung_queue_length_non_upk(tl))
                nilai_waiting_time_per_tl = (self.hitung_waiting_time_non_upk(tl))
                self.buat_avg_queue_length_non_upk[tl].append(nilai_queue_length_per_tl)
                self.buat_avg_waiting_time_non_upk[tl].append(nilai_waiting_time_per_tl)

        observation = state(self.lampu_merah)

        reward_info = build_reward_info(
            self.lampu_merah,
            self._emv_entry_time,
            self._emv_stop_count
        )

        reward_value = reward(reward_info)

        maksimum_durasi_satu_episode = 3600

        terminated = current_time >= maksimum_durasi_satu_episode
        truncated = False

        info = {}

        return observation, reward_value, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if self._sumo_started:
            traci.close()
            self._sumo_started = False