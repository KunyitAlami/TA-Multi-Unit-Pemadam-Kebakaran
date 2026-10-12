import os
import sys
import random
from collections import Counter

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError("SUMO_HOME belum diset.")

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import sumolib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_FILE = os.path.join(SCRIPT_DIR, "banjarmasintengah.net.xml")
SIMULATION_TIME = 3600
MIN_EDGE_LENGTH = 20
BASE_SEED = 42

DEMAND_LEVELS = [
    {
        "name": "low",
        "total_vehicle": 300,
        "based_on": "rata-rata jam 03:00",
        "output_file": os.path.join(SCRIPT_DIR, "demand_low.rou.xml"),
        "seed": BASE_SEED,
    },
    {
        "name": "normal",
        "total_vehicle": 7500,
        "based_on": "rata-rata jam 10:00",
        "output_file": os.path.join(SCRIPT_DIR, "demand_normal.rou.xml"),
        "seed": BASE_SEED + 1,
    },
    {
        "name": "high",
        "total_vehicle": 15000,
        "based_on": "rata-rata jam 17:00",
        "output_file": os.path.join(SCRIPT_DIR, "demand_high.rou.xml"),
        "seed": BASE_SEED + 2,
    },
]

ENTRY_EDGES = [
    "221685538#1",
    "221753368#3",
    "1120959902#278",
    "-857085599#269",
    "221762883",
    "-221762883",
    "702704534",
    "-702704534",
    "221685540#4",
    "792678882#8",
    "802880890#4",
    "355845180#1",
    "-558123018#23",
    "552778603#0",
    "549263043#4",
    "1113586959#1",
    "E1",
    "552778605#2",
    "-E1",
    "549263053#00",
    "109846741",
    "-109846741",
    "296040723#644",
    "712902521#4",
    "-722428047#1",
    "-722428047#4",
]

EXCLUDED_SPAWN_EDGES = {f"pos{i}" for i in range(1, 15)}

VEHICLE_TYPES = {
    "motor": {
        "ratio": 51.9,
        "vClass": "motorcycle",
        "attrs": {
            "id": "motor",
            "vClass": "motorcycle",
            "guiShape": "motorcycle",
            "length": "2.0",
            "width": "0.7",
            "minGap": "1.0",
            "accel": "3.0",
            "decel": "6.0",
            "maxSpeed": "30.56",
            "speedFactor": "1.10",
            "speedDev": "0.20"
        }
    },

    "mobil": {
        "ratio": 42.3,
        "vClass": "passenger",
        "attrs": {
            "id": "mobil",
            "vClass": "passenger",
            "guiShape": "passenger",
            "length": "4.5",
            "width": "1.8",
            "minGap": "2.5",
            "accel": "2.6",
            "decel": "4.5",
            "maxSpeed": "27.78",
            "speedFactor": "1.00",
            "speedDev": "0.10"
        }
    },

    "truk": {
        "ratio": 4.2,
        "vClass": "truck",
        "attrs": {
            "id": "truk",
            "vClass": "truck",
            "guiShape": "truck",
            "length": "8.0",
            "width": "2.5",
            "minGap": "2.5",
            "accel": "1.2",
            "decel": "4.0",
            "maxSpeed": "22.22",
            "speedFactor": "0.90",
            "speedDev": "0.05"
        }
    },

    "bus": {
        "ratio": 1.6,
        "vClass": "bus",
        "attrs": {
            "id": "bus",
            "vClass": "bus",
            "guiShape": "bus",
            "length": "12.0",
            "width": "2.5",
            "minGap": "2.5",
            "accel": "1.5",
            "decel": "4.0",
            "maxSpeed": "22.22",
            "speedFactor": "0.95",
            "speedDev": "0.05"
        }
    }
}


def get_valid_destination_edges(net, vclass):
    edges = []
    for edge in net.getEdges():
        if edge.isSpecial():
            continue
        if edge.getLength() < MIN_EDGE_LENGTH:
            continue
        if not edge.allows(vclass):
            continue
        edges.append(edge)

    return edges


def get_entry_edges(net, vclass):
    result = []
    for edge_id in ENTRY_EDGES:

        if edge_id in EXCLUDED_SPAWN_EDGES:
            continue

        try:
            edge = net.getEdge(edge_id)
            if edge.allows(vclass):
                result.append(edge)

        except Exception:
            pass

    return result

def poisson_departure_times(total_vehicle, simulation_time):
    arrivals = []
    rate = total_vehicle / simulation_time
    current_time = 0.0
    while len(arrivals) < total_vehicle:
        interval = random.expovariate(rate)
        current_time += interval
        if current_time > simulation_time:
            break
        arrivals.append(current_time)

    while len(arrivals) < total_vehicle:
        arrivals.append(random.uniform(0, simulation_time))

    arrivals.sort()

    return arrivals


def calculate_vehicle_numbers(total_vehicle):
    numbers = {}
    total = 0
    vehicle_names = list(VEHICLE_TYPES.keys())

    for name in vehicle_names[:-1]:
        n = round(total_vehicle * VEHICLE_TYPES[name]["ratio"] / 100)
        numbers[name] = n
        total += n

    last_vehicle = vehicle_names[-1]
    numbers[last_vehicle] = total_vehicle - total

    return numbers


def choose_destination(net, entry_edge, destination_edges, vclass):
    attempts = 0

    while attempts < 300:
        destination = random.choice(destination_edges)
        if destination.getID() == entry_edge.getID():
            attempts += 1
            continue

        path, cost = net.getShortestPath(entry_edge, destination, vClass=vclass)

        if path is None:
            attempts += 1
            continue

        if len(path) < 5:
            attempts += 1
            continue

        return destination

    return None


def write_vtypes(file):
    for vehicle in VEHICLE_TYPES.values():
        attrs = vehicle["attrs"]
        text = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        file.write(f'    <vType {text}/>\n')
    file.write("\n")


def generate_demand(net, total_vehicle, output_file, seed):
    random.seed(seed)
    statistics = Counter()

    vehicle_numbers = calculate_vehicle_numbers(total_vehicle)
    departure_times = poisson_departure_times(total_vehicle, SIMULATION_TIME)

    trips = []

    for vehicle_name in VEHICLE_TYPES:
        spec = VEHICLE_TYPES[vehicle_name]
        vclass = spec["vClass"]
        total_vehicle_type = vehicle_numbers[vehicle_name]
        destination_edges = get_valid_destination_edges(net, vclass)
        entry_edges = get_entry_edges(net, vclass)

        if len(entry_edges) == 0:
            raise RuntimeError(f"Tidak ada entry edge valid untuk {vehicle_name}")

        generated = 0

        while generated < total_vehicle_type:
            origin = random.choice(entry_edges)
            destination = choose_destination(net, origin, destination_edges, vclass)

            if destination is None:
                continue

            trips.append({
                "id": f"{vehicle_name}_{generated}",
                "type": vehicle_name,
                "from": origin.getID(),
                "to": destination.getID()
            })

            generated += 1
            statistics[vehicle_name] += 1

    random.shuffle(trips)

    for trip, depart in zip(trips, departure_times):
        trip["depart"] = depart

    trips.sort(key=lambda trip: trip["depart"])

    with open(output_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            '<routes '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n\n'
        )

        write_vtypes(f)

        for trip in trips:
            f.write(
                f'    <trip '
                f'id="{trip["id"]}" '
                f'type="{trip["type"]}" '
                f'depart="{trip["depart"]:.2f}" '
                f'departPosLat="random_free" '
                f'from="{trip["from"]}" '
                f'to="{trip["to"]}"'
                f'/>\n'
            )

        f.write("\n")
        f.write("</routes>\n")

    return statistics, len(trips)


def main():
    net = sumolib.net.readNet(NETWORK_FILE)

    print()
    print("=" * 60)
    print("Demand Generator — 3 Level (Low / Normal / High)")
    print("=" * 60)

    for level in DEMAND_LEVELS:
        print()
        print(f"[{level['name'].upper()}] target = {level['based_on']}")
        print(f"  Total kendaraan  : {level['total_vehicle']}")
        print(f"  Seed             : {level['seed']}")
        print(f"  Output           : {level['output_file']}")

        statistics, total_trips = generate_demand(
            net=net,
            total_vehicle=level["total_vehicle"],
            output_file=level["output_file"],
            seed=level["seed"],
        )

        print(f"  Generated trips  : {total_trips}")
        for vehicle_name in VEHICLE_TYPES:
            ratio = VEHICLE_TYPES[vehicle_name]["ratio"]
            total = statistics[vehicle_name]
            print(f"    {vehicle_name:8s}{total:6d} ({ratio:.1f}%)")

    print()
    print("=" * 60)
    print("Semua demand file selesai dibuat.")
    print("=" * 60)

if __name__ == "__main__":
    main()