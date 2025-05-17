

import subprocess
import os

CHECKPOINT_DIR = "data"
TAG1 = "mac"
TAG2 = "lg"
MERGED_BUFFER = os.path.join(CHECKPOINT_DIR, "replay_buffer_merged.pt")
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # Merge replay buffers
    buffer1 = os.path.join(CHECKPOINT_DIR, f"replay_buffer_{TAG1}.pt")
    buffer2 = os.path.join(CHECKPOINT_DIR, f"replay_buffer_{TAG2}.pt")
    merge_cmd = f"python scripts/merge_replay_buffers.py {MERGED_BUFFER} {buffer1} {buffer2}"
    run(merge_cmd)

    # Select best model
    ckpt1 = os.path.join(CHECKPOINT_DIR, f"checkpoint_latest_{TAG1}.pt")
    ckpt2 = os.path.join(CHECKPOINT_DIR, f"checkpoint_latest_{TAG2}.pt")
    best_cmd = f"python scripts/select_best_model.py {ckpt1} {ckpt2} {BEST_CHECKPOINT}"
    run(best_cmd)

if __name__ == "__main__":
    main()