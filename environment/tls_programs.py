import libsumo as traci

_REGISTRY = {}


def store_original_programs(tls):
    _REGISTRY.clear()

    for tl in tls:
        programs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)

        if not programs:
            raise RuntimeError(f"TLS {tl} tidak punya program")

        active_id = traci.trafficlight.getProgram(tl)
        selected = next((p for p in programs if p.programID == active_id), programs[0])

        _REGISTRY[tl] = {
            "logic": selected,
            "program_id": selected.programID,
            "phases": [
                {"index": i, "state": ph.state, "duration": ph.duration}
                for i, ph in enumerate(selected.phases)
            ],
        }


def _entry(tl):
    if tl not in _REGISTRY:
        raise RuntimeError(
            f"Program asli TLS {tl} belum disimpan. Panggil store_original_programs() "
            f"setelah traci.start() (env.reset() sudah melakukannya)."
        )
    return _REGISTRY[tl]


def get_logic(tl):
    return _entry(tl)["logic"]


def get_program_id(tl):
    return _entry(tl)["program_id"]


def get_phase_information(tl):
    return [dict(p) for p in _entry(tl)["phases"]]


def get_phase_states(tl):
    return [p["state"] for p in _entry(tl)["phases"]]


def get_green_phase_indices(tl, symbols="G"):
    return [p["index"] for p in _entry(tl)["phases"]
        if any(s in symbols for s in p["state"])
    ]


def restore_original_program(tl):
    program_id = get_program_id(tl)

    if traci.trafficlight.getProgram(tl) != program_id:
        traci.trafficlight.setProgramLogic(tl, get_logic(tl))
        traci.trafficlight.setProgram(tl, program_id)

def is_original_program_active(tl):
    return traci.trafficlight.getProgram(tl) == get_program_id(tl)


def get_current_phase(tl, default=-1):
    if not is_original_program_active(tl):
        return default
    return traci.trafficlight.getPhase(tl)