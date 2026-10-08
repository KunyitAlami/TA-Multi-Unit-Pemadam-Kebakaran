import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.env import TrafficEnvironment

SEEDS_TO_CHECK = [40, 41, 42, 43, 44, 45]
JUMLAH_UNIT_VARIANTS = [None, 3, 6, 9]


def get_fire_event_signature(seed, jumlah_unit):
    env = TrafficEnvironment(demand_level=None, jumlah_unit=jumlah_unit)
    try:
        env.reset(seed=seed)
        event = env.daftar_fire_event[0]
        return {
            "time_fire_event": event["time_fire_event"],
            "destination_edge": event["destination_edge"],
            "vehicle_number": event["vehicle_number"],
            "entry_edges": tuple(event["entry_edges"]),
        }
    finally:
        env.close()


def main():
    semua_lolos = True

    for seed in SEEDS_TO_CHECK:
        print("\n" + "=" * 90)
        print(f"SEED {seed}")
        print("=" * 90)

        hasil_per_variant = {}

        for jumlah_unit in JUMLAH_UNIT_VARIANTS:
            signature = get_fire_event_signature(seed, jumlah_unit)
            hasil_per_variant[jumlah_unit] = signature

            label = "None (training)" if jumlah_unit is None else str(jumlah_unit)
            print(f"\njumlah_unit = {label}")
            print(f"  time_fire_event   : {signature['time_fire_event']}")
            print(f"  destination_edge  : {signature['destination_edge']}")
            print(f"  vehicle_number    : {signature['vehicle_number']}")
            print(f"  entry_edges       : {signature['entry_edges']}")

        waktu_set = {sig["time_fire_event"] for sig in hasil_per_variant.values()}
        destinasi_set = {sig["destination_edge"] for sig in hasil_per_variant.values()}

        waktu_konsisten = len(waktu_set) == 1
        destinasi_konsisten = len(destinasi_set) == 1

        entry_edges_bertingkat = True
        variants_terurut = [v for v in JUMLAH_UNIT_VARIANTS if v is not None]
        variants_terurut.sort()

        for i in range(len(variants_terurut) - 1):
            kecil = hasil_per_variant[variants_terurut[i]]["entry_edges"]
            besar = hasil_per_variant[variants_terurut[i + 1]]["entry_edges"]
            if besar[: len(kecil)] != kecil:
                entry_edges_bertingkat = False

        print(f"\n  -> waktu_fire_event konsisten di semua jumlah_unit : {waktu_konsisten}")
        print(f"  -> destination_edge konsisten di semua jumlah_unit  : {destinasi_konsisten}")
        print(f"  -> entry_edges bertingkat (3 subset 6 subset 9)     : {entry_edges_bertingkat}")

        if not (waktu_konsisten and destinasi_konsisten and entry_edges_bertingkat):
            semua_lolos = False
            print("\n  *** SEED INI GAGAL VALIDASI ***")

    print("\n" + "=" * 90)
    if semua_lolos:
        print("SEMUA SEED LOLOS VALIDASI — geometri fire event konsisten lintas jumlah_unit.")
    else:
        print("ADA SEED YANG GAGAL VALIDASI — cek detail di atas.")
    print("=" * 90)


if __name__ == "__main__":
    main()