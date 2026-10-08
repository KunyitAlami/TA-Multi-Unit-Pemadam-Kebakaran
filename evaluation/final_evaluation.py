import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import fixed_time
import greedy_premption
import evaluate as mappo_masked_evaluation

import evaluate as mappo_masked_evaluation
import evaluate_nomask as mappo_nomask_evaluation

EVALUATION_SEEDS = [42]
DEMAND_LEVELS = ["low", "normal", "high"]
JUMLAH_UNIT_LEVELS = [3, 6, 9]


def run_episode(seed, demand_level, jumlah_unit):

    print("\n")
    print("=" * 90)
    print(f"EPISODE — SEED {seed} — DEMAND {demand_level.upper()} " f"— UNIT {jumlah_unit}")
    print("=" * 90)
    print("\n")

    print("#" * 90)
    print(
        f"# FIXED-TIME | SEED {seed} | "
        f"DEMAND {demand_level.upper()} | UNIT {jumlah_unit}"
    )
    print("#" * 90)

    fixed_time.SEED = seed
    fixed_time.DEMAND_LEVEL = demand_level
    fixed_time.JUMLAH_UNIT = jumlah_unit
    fixed_time.run_fixed_time()

    # 2. GREEDY PREEMPTION
    print("\n")
    print("#" * 90)
    print(
        f"# GREEDY PREEMPTION | SEED {seed} | "
        f"DEMAND {demand_level.upper()} | UNIT {jumlah_unit}"
    )
    print("#" * 90)

    greedy_premption.SEED = seed
    greedy_premption.DEMAND_LEVEL = demand_level
    greedy_premption.JUMLAH_UNIT = jumlah_unit

    greedy_premption.run_greedy_premption()

    # 3. MAPPO + RECAL + ACTION MASKING
    print("\n")
    print("#" * 90)
    print(
        f"# MAPPO (MASKED) | SEED {seed} | "
        f"DEMAND {demand_level.upper()} | UNIT {jumlah_unit}"
    )
    print("#" * 90)

    mappo_masked_evaluation.SEED = seed
    mappo_masked_evaluation.DEMAND_LEVEL = demand_level
    mappo_masked_evaluation.JUMLAH_UNIT = jumlah_unit

    mappo_masked_evaluation.run_evaluation()

    print("\n")
    print("=" * 90)
    print(
        f"EPISODE SEED {seed} — DEMAND {demand_level.upper()} "
        f"— UNIT {jumlah_unit} SELESAI"
    )
    print("=" * 90)


def main():

    start_all = time.time()

    # 3 demand × 3 jumlah unit × 3 metode
    total_simulasi = (
        len(EVALUATION_SEEDS)
        * len(DEMAND_LEVELS)
        * len(JUMLAH_UNIT_LEVELS)
        * 3
    )

    print("\n")
    print("=" * 90)
    print("EVALUASI 3 METODE — MULTI SEED x MULTI DEMAND x MULTI JUMLAH UNIT")
    print("=" * 90)

    print("Metode:")
    print("1. Fixed-Time")
    print("2. Greedy Preemption")
    print("3. MAPPO + RECAL + Action Masking")

    print(f"\nSeed        : {EVALUATION_SEEDS}")
    print(f"Demand      : {DEMAND_LEVELS}")
    print(f"Jumlah unit : {JUMLAH_UNIT_LEVELS}")

    print(f"\nTotal simulasi : {total_simulasi}")

    print("=" * 90)


    episode_number = 0

    total_episode = (
        len(EVALUATION_SEEDS)
        * len(DEMAND_LEVELS)
        * len(JUMLAH_UNIT_LEVELS)
    )

    for demand_level in DEMAND_LEVELS:

        for jumlah_unit in JUMLAH_UNIT_LEVELS:

            for seed in EVALUATION_SEEDS:

                episode_number += 1

                print("\n")
                print(
                    f"================ EPISODE "
                    f"{episode_number}/{total_episode} "
                    f"= SEED {seed} "
                    f"= DEMAND {demand_level.upper()} "
                    f"= UNIT {jumlah_unit} ================"
                )

                run_episode(
                    seed,
                    demand_level,
                    jumlah_unit
                )


    total_time = time.time() - start_all

    print("\n")
    print("=" * 90)
    print("SEMUA EVALUASI SELESAI")
    print("=" * 90)

    print(
        f"Total wall-clock time : "
        f"{total_time:.2f} detik"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()