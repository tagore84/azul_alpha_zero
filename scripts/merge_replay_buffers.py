

import sys
import torch

def load_examples(path):
    data = torch.load(path, weights_only=False)
    print(f"Loading {len(data['examples'])} examples from: {path}")
    return data['examples']

def save_examples(path, examples):
    print(f"Saving {len(examples)} examples to: {path}")
    torch.save({'examples': examples}, path)

def main():
    if len(sys.argv) != 4:
        print("Usage: python merge_replay_buffers.py <output_path> <input1.pt> <input2.pt>")
        sys.exit(1)

    output_path, input1_path, input2_path = sys.argv[1], sys.argv[2], sys.argv[3]

    examples1 = load_examples(input1_path)
    examples2 = load_examples(input2_path)

    merged = examples1 + examples2
    import random
    random.shuffle(merged)
    save_examples(output_path, merged)

if __name__ == "__main__":
    main()