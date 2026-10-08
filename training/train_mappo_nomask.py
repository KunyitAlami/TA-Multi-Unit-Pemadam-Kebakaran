import os, sys, csv, time
from datetime import datetime
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
if TRAINING_DIR not in sys.path: sys.path.insert(0, TRAINING_DIR)

from environment.env import TrafficEnvironment
from mappo_nomask import MAPPO

NUM_EPISODES = 500
MAX_STEPS = 3600
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINING_RESULT_ROOT = os.path.join(PROJECT_ROOT, "hasil_training_marl_nomask")
TRAINING_RESULT_DIR = os.path.join(TRAINING_RESULT_ROOT, "mappo_nomask")
MODEL_RESULT_ROOT = os.path.join(PROJECT_ROOT, "hasil_train_model_nomask")
MODEL_RESULT_DIR = os.path.join(MODEL_RESULT_ROOT, "mappo_nomask")


def prepare_result_folders():
    os.makedirs(TRAINING_RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_RESULT_DIR, exist_ok=True)


def create_run_directories(start_datetime):
    timestamp = start_datetime.strftime("%Y%m%d_%H%M%S")
    run_name = f"mappo_nomask_{timestamp}"
    training_run_dir = os.path.join(TRAINING_RESULT_DIR, run_name)
    model_run_dir = os.path.join(MODEL_RESULT_DIR, run_name)
    os.makedirs(training_run_dir, exist_ok=True)
    os.makedirs(model_run_dir, exist_ok=True)
    return run_name, training_run_dir, model_run_dir


def save_checkpoint(filepath, mappo, episode, total_episodes, total_steps, episode_reward, average_reward):
    checkpoint = {
        "algorithm": "MAPPO_NOMASK", "episode": episode, "total_episodes": total_episodes, "total_steps": total_steps,
        "episode_reward": float(episode_reward), "average_reward": float(average_reward),
        "num_agents": mappo.num_agents, "state_dim": mappo.state_dim, "action_dim": mappo.action_dim,
        "agent_names": list(mappo.agent_names), "global_state_dim": mappo.global_state_dim,
        "actors": {tl: actor.state_dict() for tl, actor in mappo.actors.items()},
        "critic": mappo.critic.state_dict(),
        "actor_optimizers": {tl: optimizer.state_dict() for tl, optimizer in mappo.actor_optimizers.items()},
        "critic_optimizer": mappo.critic_optimizer.state_dict(),
        "hyperparameters": {
            "lr_actor": LR_ACTOR, "lr_critic": LR_CRITIC, "gamma": mappo.gamma, "gae_lambda": mappo.gae_lambda,
            "clip_epsilon": mappo.clip_epsilon, "entropy_coef": mappo.entropy_coef, "value_coef": mappo.value_coef,
            "epochs": mappo.epochs, "seed": SEED,
        },
    }
    torch.save(checkpoint, filepath)


def save_training_summary(filepath, start_datetime, end_datetime, wall_time, mappo, total_steps, rewards_history, steps_history, final_model_path):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("HASIL TRAINING MAPPO (TANPA ACTION MASKING)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Algorithm           : MAPPO (no mask)\n")
        f.write(f"Seed                : {SEED}\n")
        f.write(f"Device              : {DEVICE}\n")
        f.write(f"Jumlah episode      : {NUM_EPISODES}\n")
        f.write(f"Max steps/episode   : {MAX_STEPS}\n")
        f.write(f"Learning rate actor : {LR_ACTOR}\n")
        f.write(f"Learning rate critic: {LR_CRITIC}\n")
        f.write(f"Gamma               : {mappo.gamma}\n")
        f.write(f"GAE lambda          : {mappo.gae_lambda}\n")
        f.write(f"Clip epsilon        : {mappo.clip_epsilon}\n")
        f.write(f"Entropy coefficient : {mappo.entropy_coef}\n")
        f.write(f"Value coefficient   : {mappo.value_coef}\n")
        f.write(f"Update epochs       : {mappo.epochs}\n\n")
        f.write(f"Jam mulai           : {start_datetime:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Jam selesai         : {end_datetime:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Wall-clock runtime  : {wall_time:.6f} detik\n")
        f.write(f"Total training steps: {total_steps}\n\n")

        f.write("AGENT CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of agents    : {mappo.num_agents}\n")
        for tl in mappo.agent_names:
            f.write(f"{tl:<30} | State dim = {mappo.state_dim[tl]} | Action dim = {mappo.action_dim[tl]}\n")
        f.write("\n")

        f.write("EPISODE SUMMARY\n")
        f.write("-" * 80 + "\n")
        for index in range(len(rewards_history)):
            avg_reward = rewards_history[index] / max(steps_history[index], 1)
            f.write(f"Episode {index + 1:4d} | Steps = {steps_history[index]:5d} | Total Reward = {rewards_history[index]:.6f} | Average Reward = {avg_reward:.6f}\n")
        f.write("\n")

        if rewards_history:
            f.write(f"First episode reward: {rewards_history[0]:.6f}\n")
            f.write(f"Last episode reward : {rewards_history[-1]:.6f}\n")
            f.write(f"Best episode reward : {max(rewards_history):.6f}\n")
            f.write(f"Worst episode reward: {min(rewards_history):.6f}\n")
        f.write("\n")

        f.write("FINAL MODEL\n")
        f.write("-" * 80 + "\n")
        f.write(f"{final_model_path}\n\n")
        f.write("=" * 80 + "\n")


def run_training():
    prepare_result_folders()
    start_datetime = datetime.now()
    run_name, training_run_dir, model_run_dir = create_run_directories(start_datetime)
    csv_path = os.path.join(training_run_dir, "mappo_nomask_training.csv")
    txt_path = os.path.join(training_run_dir, "mappo_nomask_training.txt")

    print("\n" + "=" * 80)
    print("MAPPO TRAINING (TANPA ACTION MASKING)")
    print("=" * 80)
    print(f"Episode           : {NUM_EPISODES}")
    print(f"Max steps         : {MAX_STEPS}")
    print(f"Seed              : {SEED}")
    print(f"Device            : {DEVICE}")
    print(f"Training result   : {training_run_dir}")
    print(f"Model result      : {model_run_dir}")
    print("=" * 80)

    env = TrafficEnvironment()
    csv_file = None

    try:
        states, info = env.reset(seed=SEED)
        num_agents = len(env.lampu_merah)
        state_dim = {tl: len(states[tl]) for tl in env.lampu_merah}
        action_dim = {tl: len(env.action_space_marl[tl]) for tl in env.lampu_merah}

        print("\n=== MAPPO CONFIGURATION ===")
        print(f"Number of agents : {num_agents}")
        for tl in env.lampu_merah:
            print(f"{tl} | State dim = {state_dim[tl]} | Action dim = {action_dim[tl]}")
        print(f"Device           : {DEVICE}")
        print(f"CSV              : {csv_path}")

        mappo = MAPPO(num_agents=num_agents, state_dim=state_dim, action_dim=action_dim, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, device=DEVICE)

        csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["episode", "steps", "total_reward", "average_reward", "terminated", "truncated"])
        csv_file.flush()

        rewards_history = []
        steps_history = []
        total_training_steps = 0
        training_wall_start = time.time()

        for episode in range(NUM_EPISODES):
            states, info = env.reset(seed=SEED + episode)
            episode_reward = 0.0
            terminated = False
            truncated = False
            steps_taken = 0

            for step in range(MAX_STEPS):
                actions, log_probs, values = mappo.get_action(states)
                next_states, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated

                mappo.buffer.store(states, actions, rewards, log_probs, values, done)
                episode_reward += sum(rewards.values())
                states = next_states
                steps_taken = step + 1

                if done: break

            mappo.update()
            total_training_steps += steps_taken
            average_reward = episode_reward / max(steps_taken, 1)
            rewards_history.append(episode_reward)
            steps_history.append(steps_taken)

            csv_writer.writerow([episode + 1, steps_taken, episode_reward, average_reward, terminated, truncated])
            csv_file.flush()

            episode_model_path = os.path.join(model_run_dir, f"mappo_nomask_episode_{episode + 1:03d}.pt")
            save_checkpoint(episode_model_path, mappo, episode + 1, NUM_EPISODES, total_training_steps, episode_reward, average_reward)

            print(f"Episode {episode + 1}/{NUM_EPISODES} | Steps: {steps_taken} | Reward: {episode_reward:.2f} | Avg Reward: {average_reward:.4f}")

        csv_file.close()
        csv_file = None

        final_model_path = os.path.join(model_run_dir, "mappo_nomask_final.pt")
        final_episode_reward = rewards_history[-1] if rewards_history else 0.0
        final_average_reward = (final_episode_reward / max(steps_history[-1], 1)) if steps_history else 0.0
        save_checkpoint(final_model_path, mappo, NUM_EPISODES, NUM_EPISODES, total_training_steps, final_episode_reward, final_average_reward)

        end_datetime = datetime.now()
        wall_time = time.time() - training_wall_start
        save_training_summary(txt_path, start_datetime, end_datetime, wall_time, mappo, total_training_steps, rewards_history, steps_history, final_model_path)

        print("\n" + "=" * 80)
        print("TRAINING MAPPO (NO MASK) SELESAI")
        print("=" * 80)
        print(f"Total episode     : {NUM_EPISODES}")
        print(f"Total steps       : {total_training_steps}")
        print(f"Wall-clock time   : {wall_time:.2f} detik")
        if rewards_history:
            print(f"First reward      : {rewards_history[0]:.4f}")
            print(f"Last reward       : {rewards_history[-1]:.4f}")
            print(f"Best reward       : {max(rewards_history):.4f}")
        print(f"\nCSV training      : {csv_path}")
        print(f"TXT training      : {txt_path}")
        print(f"Model final       : {final_model_path}")
        print(f"Model checkpoints : {model_run_dir}")
        print("=" * 80)

    finally:
        if csv_file is not None: csv_file.close()
        env.close()
        print("\nEnvironment ditutup.")


if __name__ == "__main__":
    run_training()