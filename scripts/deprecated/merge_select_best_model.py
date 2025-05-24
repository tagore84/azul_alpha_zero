

import os
import sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from azul.env import AzulEnv
from net.azul_net import AzulNet, evaluate_against_previous

def load_model(path, in_channels:int, global_size:int, action_size:int, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model = AzulNet(
        in_channels=in_channels,
        global_size=global_size,
        action_size=action_size
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model

def main():
    #if len(sys.argv) != 4:
    #    print("Usage: python select_best_model.py modelA.pt modelB.pt output_best.pt")
    #    sys.exit(1)

    #model_a_path, model_b_path, output_path = sys.argv[1:]

    model_a_path = 'data/checkpoint_dir/checkpoint_latest_mac.pt'
    model_b_path = 'data/checkpoint_dir/checkpoint_latest_mac_100.pt'
    output_path = 'data/checkpoint_dir/best.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = AzulEnv()
    # Dynamically compute observation sizes from a sample reset
    sample_obs = env.reset()
    obs_flat = env.encode_observation(sample_obs)
    total_obs_size = obs_flat.shape[0]
    # Determine number of spatial channels (must divide by 5*5)
    in_channels = total_obs_size // (5 * 5)
    spatial_size = in_channels * 5 * 5
    global_size = total_obs_size - spatial_size
    print(f"Obs total size: {total_obs_size}, spatial_size: {spatial_size}, global_size: {global_size}, in_channels: {in_channels}")
    action_size = env.action_size
    model_a = load_model(model_a_path, in_channels, global_size, action_size, device)
    model_b = load_model(model_a_path, in_channels, global_size, action_size, device)

    print("Evaluating model A vs model B...")
    env_args = {}
    simulations = 3
    cpuct = 1.0
    wins_a, wins_b = evaluate_against_previous(model_a, model_b, env_args, simulations, cpuct, n_games=1)

    print(f"Model A wins: {wins_a}")
    print(f"Model B wins: {wins_b}")

    best_path = model_a_path if wins_a >= wins_b else model_b_path
    torch.save(torch.load(best_path), output_path)
    print(f"Best model saved to {output_path}")

if __name__ == "__main__":
    main()