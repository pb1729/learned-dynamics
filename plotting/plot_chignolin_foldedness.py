import argparse
from os import path, listdir

import torch
import matplotlib.pyplot as plt

from managan.config import get_predictor
from managan.predictor import ModelState
from managan.statefiles import read_state_from_file
from managan.utils import must_be


def foldedness(chain):
  """ chain: (..., nres, 3) """
  *_, nres, must_be[3] = chain.shape
  '''M1 = torch.arange(nres, device=chain.device) - nres/2
  M2 = 2*(abs(M1) - nres/4)
  return torch.sqrt(((chain*M2[..., None]).mean(-2)**2).sum(-1)) - 1j*torch.sqrt(((chain*M1[..., None]).mean(-2)**2).sum(-1))'''
  return torch.sqrt(((chain.flip(-2) - chain)**2).sum(-1).mean(-1))

def read_statepytho_files(dir_path):
    """Read all state files from a directory and return a sorted list of states.

    Files are expected to have the format '{rest of filename}_{number}.bin'
    and are sorted by the number.
    """
    if not path.isdir(dir_path):
        raise ValueError(f"{dir_path} is not a directory")

    ans = []
    file_list = []

    # Get all .bin files in the directory
    for file in listdir(dir_path):
        if file.endswith(".bin"):
            try:
                # Extract the number from the filename
                number = int(file.split("_")[-1].split(".")[0])
                file_list.append((number, file))
            except (ValueError, IndexError):
                print(f"Warning: Skipping file {file} as it doesn't match the expected format")

    # Sort files by their number
    file_list.sort()

    # Read states from sorted files
    for _, file in file_list:
        file_path = path.join(dir_path, file)
        try:
            with open(file_path, "rb") as f:
                state = read_state_from_file(f)
                chain = state.x[..., state.metadata.residue_indices, :]
                ans.append(foldedness(chain))
        except Exception as e:
            print(f"Error reading state from {file_path}: {e}")
    return ans

def main():
    """Main function that reads state files and prints the count."""
    parser = argparse.ArgumentParser(description='Read state files from a directory')
    parser.add_argument('directory', help='Directory containing state files')
    args = parser.parse_args()

    fold_values = read_statepytho_files(args.directory)
    print(f"Found {len(fold_values)} state files in {args.directory}")


    plt.figure(figsize=(16, 1 * len(fold_values)))
    for i, fold_value in enumerate(fold_values):
        plt.subplot(len(fold_values), 1, i + 1)
        for idx_batch in range(fold_value.shape[1]):
          plt.plot(fold_value[:, idx_batch].cpu())
        plt.ylim(0., 35.)
        plt.ylabel('Foldedness')
        plt.tight_layout()
    # Hide x-axis for all subplots except the last one
    if i < len(fold_values) - 1:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    else:
        plt.xlabel('Steps')

    # Remove space between subplots
    plt.subplots_adjust(hspace=0)
    plt.show()


if __name__ == "__main__":
    main()
