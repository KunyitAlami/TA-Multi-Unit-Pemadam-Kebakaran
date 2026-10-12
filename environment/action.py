import libsumo as traci
from . import tls_programs

HOLD_ACTION = 0

# ambil dari setiap lampu merah fase yang ada "G" (lampu hijau)
def get_green_phases(tls):
    # Selalu dari program asli (tls_programs), bukan program aktif TraCI yang bisa
    # berubah menjadi 'online' saat transisi. Hanya fase dengan "G" (protected),
    # supaya ukuran action space tidak berubah.
    return {tl: tls_programs.get_green_phase_indices(tl, "G") for tl in tls}


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