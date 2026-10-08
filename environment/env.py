import libsumo as traci
import numpy as np
import gymnasium as gym

from .state import state, build_reward_info
from .reward import reward
from .action import build_action_space, action

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MIN_GREEN_DURATION = 5
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

        self.remaining_duration = {
            tl: 0
            for tl in self.lampu_merah
        }

        self.green_elapsed = {
            tl: 0
            for tl in self.lampu_merah
        }

        self.active_phase = {
            tl: None
            for tl in self.lampu_merah
        }

        self.preemption_active = {
            tl: False
            for tl in self.lampu_merah
        }

        self.transition_phase = {
            tl: None
            for tl in self.lampu_merah
        }

        self.transition_active = {
            tl: False
            for tl in self.lampu_merah
        }

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

        programs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)

        lane_phases = {
            lane: []
            for lane in incoming_lanes
        }

        if not programs:
            return lane_phases

        active_program = traci.trafficlight.getProgram(tl)

        selected_logic = None

        for logic in programs:
            if logic.programID == active_program:
                selected_logic = logic
                break

        if selected_logic is None:
            selected_logic = programs[0]

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
        programs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)

        if not programs:
            return []

        active_program = traci.trafficlight.getProgram(tl)

        selected_logic = None

        for logic in programs:
            if logic.programID == active_program:
                selected_logic = logic
                break

        if selected_logic is None:
            selected_logic = programs[0]

        phase_information = []

        for phase_index, phase in enumerate(selected_logic.phases):
            phase_information.append({
                "index": phase_index,
                "state": phase.state,
                "duration": phase.duration
            })

        return phase_information
    
    def get_action_mask(self, tl):
        action_space_tl = (self.action_space_marl[tl])
        mask = np.zeros(len(action_space_tl), dtype=np.float32)
        mask[0] = 1.0

        if self.transition_active[tl]:
            return mask
        if (self.preemption_active[tl] and self.green_elapsed[tl] < MIN_GREEN_DURATION):
            return mask
        
        lane_phase_mapping = (self.get_lane_phase_mapping(tl))
        incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))
        phases_with_emv = set()

        for lane in incoming_lanes:
            vehicles = (traci.lane.getLastStepVehicleIDs(lane))
            phases = lane_phase_mapping.get(lane, [])

            for vehicle_id in vehicles:
                try:
                    if (traci.vehicle.getVehicleClass(vehicle_id)== "emergency"):
                        phases_with_emv.update(phases)
                        break
                except Exception:
                    continue

        for action_id, phase in (action_space_tl.items()):
            if action_id == 0:
                continue
            if phase in phases_with_emv:
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

        self.remaining_duration = {
            tl: 0
            for tl in self.lampu_merah
        }

        self.green_elapsed = {
            tl: 0
            for tl in self.lampu_merah
        }

        self.active_phase = {
            tl: None
            for tl in self.lampu_merah
        }

        self.preemption_active = {
            tl: False
            for tl in self.lampu_merah
        }

        self.transition_phase = {
            tl: None
            for tl in self.lampu_merah
        }

        self.transition_active = {
            tl: False
            for tl in self.lampu_merah
        }

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
        lane_upk = self.get_lane_upk(tl)

        for lane in incoming_lanes:
            if lane in lane_upk:
                continue

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

        lane_upk = self.get_lane_upk(tl)

        for lane in incoming_lanes:
            if lane in lane_upk:
                continue
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
    
    def execute_action(self, tl, target_phase):
        current_phase = traci.trafficlight.getPhase(tl)
        self.green_elapsed[tl] = 0

        if current_phase == target_phase:
            self.active_phase[tl] = target_phase
            self.preemption_active[tl] = True
            self.transition_active[tl] = False
            return

        phases = self.get_phase_information(tl)

        if not phases:
            raise RuntimeError(
                f"Tidak ada phase untuk TLS {tl}"
            )

        jumlah_phase = len(phases)

        valid_phases = {
            phase["index"]
            for phase in phases
        }

        if target_phase not in valid_phases:
            raise RuntimeError(
                f"Target phase tidak valid | "
                f"TLS={tl} | "
                f"current={current_phase} | "
                f"target={target_phase} | "
                f"valid={sorted(valid_phases)}"
            )

        transition_phases = []
        phase = (current_phase + 1) % jumlah_phase

        for _ in range(jumlah_phase):
            if phase == target_phase:
                break

            transition_phases.append(phase)
            phase = (phase + 1) % jumlah_phase
        else:
            raise RuntimeError(
                f"Transition phase gagal | "
                f"TLS={tl} | "
                f"current={current_phase} | "
                f"target={target_phase}"
            )

        if len(transition_phases) == 0:
            traci.trafficlight.setPhase(
                tl,
                target_phase
            )

            self.active_phase[tl] = target_phase
            self.preemption_active[tl] = True
            self.transition_active[tl] = False
            self.green_elapsed[tl] = 0
            return

        first_phase = transition_phases[0]

        self.transition_phase[tl] = {
            "phases": transition_phases,
            "index": 0,
            "target_phase": target_phase,
            "remaining_duration": phases[first_phase]["duration"]
        }

        self.transition_active[tl] = True
        self.preemption_active[tl] = False

        traci.trafficlight.setPhase(
            tl,
            first_phase
        )

        traci.trafficlight.setPhaseDuration(
            tl,
            phases[first_phase]["duration"]
        )

    def simulation_step(self):
        traci.simulationStep()

        for tl in self.lampu_merah:
            if self.transition_active[tl]:
                transition = self.transition_phase[tl]
                phases = self.get_phase_information(tl)
                transition["remaining_duration"] -= 1

                if transition["remaining_duration"] <= 0:
                    transition["index"] += 1

                    if transition["index"] >= len(transition["phases"]):
                        target_phase = transition["target_phase"]
                        traci.trafficlight.setPhase(tl, target_phase)
                        self.active_phase[tl] = target_phase

                        self.preemption_active[tl] = True
                        self.transition_active[tl] = False
                        self.green_elapsed[tl] = 0


                    else:
                        next_phase = transition["phases"][transition["index"]]
                        transition["remaining_duration"] = (phases[next_phase]["duration"])
                        traci.trafficlight.setPhase(tl, next_phase)
                        traci.trafficlight.setPhaseDuration(tl, phases[next_phase]["duration"])

            elif self.preemption_active[tl]:
                self.green_elapsed[tl] += 1

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
            if self.transition_active[tl]:
                continue

            if (self.preemption_active[tl] and self.green_elapsed[tl] < MIN_GREEN_DURATION):
                continue

            action_id = action_marl[tl]
            if action_id == 0:
                continue

            target_phase = action(self.action_space_marl, tl, action_id)
            self.execute_action(tl, target_phase)

        for _ in range(MIN_GREEN_DURATION):
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