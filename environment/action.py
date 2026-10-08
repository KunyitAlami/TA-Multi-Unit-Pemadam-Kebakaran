import libsumo as traci

HOLD_ACTION = 0

# ambil dari setiap lampu merah fase yang ada "G" (lampu hijau)
def get_green_phases(tls):
    green_phases = {}
    for tl in tls:
        programs = (traci.trafficlight.getCompleteRedYellowGreenDefinition(tl))

        if not programs:
            green_phases[tl] = []
            continue

        active_program = (traci.trafficlight.getProgram(tl))
        selected_logic = None

        for logic in programs:
            if logic.programID == active_program:
                selected_logic = logic
                break

        if selected_logic is None:
            selected_logic = programs[0]

        phases = []

        for phase_index, phase in enumerate(selected_logic.phases):
            if "G" in phase.state:
                phases.append(phase_index)
        green_phases[tl] = phases

    return green_phases


def build_action_space(tls):
    green_phases = get_green_phases(tls)
    action_space = {}

    for tl in tls:
        action_space[tl] = {HOLD_ACTION: "HOLD"}

        for action_id, phase in enumerate(green_phases[tl],start=1):
            action_space[tl][action_id] = phase

    return action_space

def action(action_space, tl, action_id):
    return action_space[tl][action_id]