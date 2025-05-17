

import os
import sys
import torch
from net.azul_net import AzulNet, evaluate_against_previous
from azul.env import AzulEnv

def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model = AzulNet()
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model

def main():
    if len(sys.argv) != 4:
        print("Usage: python select_best_model.py modelA.pt modelB.pt output_best.pt")
        sys.exit(1)

    model_a_path, model_b_path, output_path = sys.argv[1:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = AzulEnv()
    model_a = load_model(model_a_path, device)
    model_b = load_model(model_b_path, device)

    print("Evaluating model A vs model B...")
    wins_a, wins_b = evaluate_against_previous(env, model_a, model_b, n_games=20)

    print(f"Model A wins: {wins_a}")
    print(f"Model B wins: {wins_b}")

    best_path = model_a_path if wins_a >= wins_b else model_b_path
    torch.save(torch.load(best_path), output_path)
    print(f"Best model saved to {output_path}")

if __name__ == "__main__":
    main()