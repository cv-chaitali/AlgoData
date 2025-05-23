import argparse
import torch
import numpy as np
from datasets import load_dataset
import json
from tqdm import tqdm

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- CRAIG Method ---
def select_subset_craig(embeddings, subset_size):
    """
    Selects a subset of embeddings by maximizing Tr(G_S^T G_S),
    which simplifies to selecting embeddings with the largest squared L2 norms.
    G_S is assumed to be the matrix of selected embeddings.
    """
    print("Selecting subset using CRAIG method (maximizing sum of squared L2 norms)...")
    if subset_size <= 0:
        raise ValueError("Subset size must be positive.")
    if subset_size > embeddings.shape[0]:
        print(f"Warning: Subset size ({subset_size}) is larger than the number of available embeddings ({embeddings.shape[0]}). Selecting all embeddings.")
        subset_size = embeddings.shape[0]
        #return np.arange(embeddings.shape[0])


    norms_sq = torch.sum(embeddings**2, dim=1)
    # Ensure k is not greater than the number of items available for topk
    k_val = min(subset_size, embeddings.shape[0])
    if k_val == 0 and subset_size > 0 : # Handle case where embeddings might be empty but subset_size > 0
        print("Warning: No embeddings available to select from.")
        return np.array([], dtype=int)

    selected_indices = torch.topk(norms_sq, k=k_val, largest=True).indices
    
    print(f"Selected {len(selected_indices)} indices based on max squared L2 norm.")
    return selected_indices.cpu().numpy()

# --- Probabilistic Bilevel Coreset (Placeholder) ---
def select_subset_pbc(embeddings, subset_size):
    """
    Placeholder for Probabilistic Bilevel Coreset selection.
    Actual implementation requires a model, loss functions, and training loop.
    """
    print("\nSelecting subset using Probabilistic Bilevel Coreset (PBC) method...")
    print("--------------------------------------------------------------------------")
    print("WARNING: Probabilistic Bilevel Coreset, as described by the formula:")
    print("  min_{p∈∆k} L_val(θ*(p)) s.t. θ*(p) = arg min_θ Σ p_i ℓ(θ; x_i)")
    print("requires a model (θ), loss functions (ℓ, L_val), and a training/validation setup.")
    print("This cannot be fully implemented using only pre-computed embeddings without further specifications.")
    print("--------------------------------------------------------------------------")
    
    if subset_size <= 0:
        raise ValueError("Subset size must be positive.")
    if subset_size > embeddings.shape[0]:
        print(f"Warning: Subset size ({subset_size}) is larger than the number of available embeddings ({embeddings.shape[0]}) for PBC placeholder. Selecting all embeddings.")
        #return np.arange(embeddings.shape[0])
        subset_size = embeddings.shape[0]

    if embeddings.shape[0] == 0 and subset_size > 0:
        print("Warning: No embeddings available to select from for PBC placeholder.")
        return np.array([], dtype=int)
    if embeddings.shape[0] > 0  and subset_size == 0 :
        return np.array([], dtype=int)
    if embeddings.shape[0] == 0  and subset_size == 0 :
        return np.array([], dtype=int)


    print(f"Returning a random subset of size {subset_size} as a placeholder for PBC.")
    return np.random.choice(embeddings.shape[0], size=subset_size, replace=False)

# --- Stochastic Greedy Method ---
def select_subset_stochastic_greedy(embeddings, subset_size, sample_size_Rt=100, objective_type="facility_location"):
    """
    Selects a subset using a stochastic greedy algorithm.
    S_{t+1} = S_t ∪ arg max_{x∈R_t} [f(S_t ∪ {x}) - f(S_t)]

    Args:
        embeddings (torch.Tensor): The full set of embeddings.
        subset_size (int): The desired size of the subset.
        sample_size_Rt (int): The size of the random sample R_t at each step.
        objective_type (str): "facility_location" or "sum_sq_norms".
    """
    print(f"\nSelecting subset using Stochastic Greedy method...")
    print(f"Parameters: subset_size={subset_size}, sample_size_Rt={sample_size_Rt}, objective_type='{objective_type}'")

    num_embeddings = embeddings.shape[0]

    if subset_size <= 0:
        raise ValueError("Subset size must be positive.")
    if subset_size > num_embeddings:
        print(f"Warning: Subset size ({subset_size}) is larger than the number of available embeddings ({num_embeddings}). Selecting all embeddings.")
        #return np.arange(num_embeddings)
        subset_size = num_embeddings
    
    if num_embeddings == 0 and subset_size > 0 :
        print("Warning: No embeddings available to select from for Stochastic Greedy.")
        return np.array([], dtype=int)
    if num_embeddings > 0  and subset_size == 0 :
        return np.array([], dtype=int)
    if num_embeddings == 0  and subset_size == 0 :
        return np.array([], dtype=int)


    selected_indices_set = set()
    candidate_pool_indices = list(range(num_embeddings))

    if objective_type == "facility_location":
        print("Using Facility Location objective for Stochastic Greedy.")
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        epsilon_norm = 1e-12 # Small epsilon to prevent division by zero for zero vectors
        embeddings_normalized = embeddings / (norms + epsilon_norm)
        # current_max_similarities_to_S stores max_s∈S sim(s,j) for each j ∈ D
        current_max_similarities_to_S = torch.full((num_embeddings,), -float('inf'), device=device, dtype=embeddings.dtype)
    elif objective_type == "sum_sq_norms":
        print("Using sum of squared norms objective for Stochastic Greedy.")
        embedding_norms_sq = torch.sum(embeddings**2, dim=1)
    else:
        raise ValueError(f"Unknown objective_type for Stochastic Greedy: {objective_type}")

    for iter_num in tqdm(range(subset_size), desc="Stochastic Greedy Selection"):
        current_candidate_indices = [i for i in candidate_pool_indices if i not in selected_indices_set]

        if not current_candidate_indices:
            print(f"Stopped early at iteration {iter_num+1} as no more unique candidates are available.")
            break

        # Sample R_t from (D \ S_t)
        if len(current_candidate_indices) <= sample_size_Rt:
            Rt_selected_indices = current_candidate_indices
        else:
            # np.random.choice samples indices from 0 to len-1, so map back to actual embedding indices
            sampled_indices_in_pool = np.random.choice(len(current_candidate_indices), size=sample_size_Rt, replace=False)
            Rt_selected_indices = [current_candidate_indices[i] for i in sampled_indices_in_pool]
        
        if not Rt_selected_indices: # Should not happen if current_candidate_indices is not empty
             print(f"Warning: R_t is empty at iteration {iter_num+1}. Skipping selection step.")
             continue

        best_x_in_Rt = -1
        max_gain_in_Rt = -float('inf')

        for x_idx in Rt_selected_indices:
            gain = 0.0
            if objective_type == "facility_location":
                # Gain for adding x = sum_{j in D} max(0, sim(x,j) - max_{s in S} sim(s,j))
                # sim(x,j) is embeddings_normalized[x_idx] @ embeddings_normalized[j].T
                similarities_of_x_to_all_D = embeddings_normalized[x_idx] @ embeddings_normalized.T
                marginal_gains = torch.maximum(torch.tensor(0.0, device=device, dtype=embeddings.dtype), 
                                               similarities_of_x_to_all_D - current_max_similarities_to_S)
                gain = torch.sum(marginal_gains).item()
            elif objective_type == "sum_sq_norms":
                # Gain for adding x is ||x||^2
                gain = embedding_norms_sq[x_idx].item()
            
            if gain > max_gain_in_Rt:
                max_gain_in_Rt = gain
                best_x_in_Rt = x_idx
        
        if best_x_in_Rt != -1:
            selected_indices_set.add(best_x_in_Rt)
            if objective_type == "facility_location":
                # Update current_max_similarities_to_S with the newly added point
                similarities_of_best_x_to_all_D = embeddings_normalized[best_x_in_Rt] @ embeddings_normalized.T
                current_max_similarities_to_S = torch.maximum(current_max_similarities_to_S, similarities_of_best_x_to_all_D)
        else:
            # This can happen if all gains in R_t are <= 0 or R_t was empty.
            # If R_t was not empty but all gains were <=0, we might pick a random one from R_t,
            # or the one with the least negative gain. For simplicity, if no positive gain,
            # we might not add, or add the one that was 'best' (e.g. closest to 0 gain).
            # Current logic: if no element gives gain > -inf (initial), nothing is added.
            # This implies we only add if there's some positive utility or if it's the first element.
            # For Facility Location, first element's gain = sum(sim(x,j)) for all j, assuming current_max_sim is -inf.
             print(f"Warning: No element from R_t chosen at iteration {iter_num+1} (max_gain: {max_gain_in_Rt}). This might happen if all gains are zero or negative.")
             # To ensure subset_size is met, one could add a random element from Rt_selected_indices if best_x_in_Rt remains -1
             # and Rt_selected_indices is not empty.
             if Rt_selected_indices and len(selected_indices_set) < subset_size : # Fallback if no positive gain
                fallback_idx = np.random.choice(Rt_selected_indices)
                if fallback_idx not in selected_indices_set:
                    selected_indices_set.add(fallback_idx)
                    print(f"Added a fallback element {fallback_idx} to meet subset size.")
                    if objective_type == "facility_location": # Must update if adding
                         similarities_of_fallback_to_all_D = embeddings_normalized[fallback_idx] @ embeddings_normalized.T
                         current_max_similarities_to_S = torch.maximum(current_max_similarities_to_S, similarities_of_fallback_to_all_D)


    final_selected_indices = np.array(list(selected_indices_set), dtype=int)
    
    # If fewer than subset_size selected (e.g. due to candidate exhaustion or persistent non-positive gains)
    if len(final_selected_indices) < subset_size:
        num_still_needed = subset_size - len(final_selected_indices)
        remaining_pool = [i for i in candidate_pool_indices if i not in final_selected_indices]
        if num_still_needed > 0 and remaining_pool:
            print(f"Stochastic Greedy selected {len(final_selected_indices)} items, less than requested {subset_size}. Padding with {min(num_still_needed, len(remaining_pool))} random items from the remainder.")
            padding_count = min(num_still_needed, len(remaining_pool))
            padding_indices = np.random.choice(remaining_pool, size=padding_count, replace=False)
            final_selected_indices = np.concatenate((final_selected_indices, padding_indices))

    print(f"Selected {len(final_selected_indices)} indices using Stochastic Greedy.")
    return final_selected_indices

# --- Main Function ---
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading embeddings from: {args.embedding_checkpoint}")
    try:
        embeddings_np = np.load(args.embedding_checkpoint)
    except FileNotFoundError:
        print(f"Error: Embedding checkpoint file not found at {args.embedding_checkpoint}")
        return
    
    embeddings = torch.tensor(embeddings_np, dtype=torch.float32).to(device)
    print(f"Loaded embeddings with shape: {embeddings.shape}")

    if embeddings.shape[0] == 0:
        print("Error: The loaded embeddings are empty. Cannot proceed with subset selection.")
        # Create an empty JSON file if output is expected
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump([], f)
            print(f"Empty selected data saved to {args.output_json}")
        return

    selected_indices = None
    if args.method == "craig":
        selected_indices = select_subset_craig(embeddings, args.subset_size)
    elif args.method == "pbc":
        selected_indices = select_subset_pbc(embeddings, args.subset_size)
    elif args.method == "stochastic_greedy":
        selected_indices = select_subset_stochastic_greedy(
            embeddings, 
            args.subset_size,
            sample_size_Rt=args.sg_sample_size_Rt,
            objective_type=args.sg_objective_type
        )
    else:
        print(f"Error: Unknown selection method '{args.method}'")
        return

    if selected_indices is None or selected_indices.size == 0:
        print("No indices were selected. Output file will be empty.")
        selected_data = []
    else:
        print(f"\nLoading dataset '{args.dataset_name}' split '{args.dataset_split}' to retrieve selected data points...")
        try:
            U_data = load_dataset(args.dataset_name, split=args.dataset_split, revision=args.dataset_revision)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Cannot save selected data content. Saving only indices if requested elsewhere or exiting.")
            return

        print(f"Original dataset size: {len(U_data)}")
        print(f"Number of selected indices: {len(selected_indices)}")
        
        selected_data = []
        for i in selected_indices:
            idx = int(i) # Ensure it's a Python int for indexing
            if 0 <= idx < len(U_data):
                 # Adapt keys based on your dataset structure if not "instruction", "input", "output"
                try:
                    data_item = U_data[idx]
                    item_to_save = {}
                    # Check for common keys, adapt as needed
                    item_to_save["instruction"] = data_item.get("instruction", data_item.get("instruction", ""))
                    item_to_save["input"] = data_item.get("input", "")
                    item_to_save["output"] = data_item.get("output", data_item.get("completion", ""))
                    # Fallback: save the whole item if keys are very different
                    if not item_to_save["instruction"] and not item_to_save["output"]:
                        item_to_save = dict(data_item)
                    selected_data.append(item_to_save)

                except IndexError:
                    print(f"Warning: Index {idx} is out of bounds for the loaded dataset (size {len(U_data)}). Skipping this index.")
                except Exception as e:
                    print(f"Warning: Could not process data for index {idx}. Error: {e}. Item: {U_data[idx] if 0 <= idx < len(U_data) else 'Index out of bounds'}")
            else:
                print(f"Warning: Selected index {idx} is out of bounds for the loaded dataset (size {len(U_data)}). Skipping this index.")


    if args.output_json:
        print(f"\nSaving {len(selected_data)} selected data points to {args.output_json}...")
        with open(args.output_json, "w") as f:
            json.dump(selected_data, f, indent=4)
        print(f"Selected data saved successfully.")
    else:
        print("No output JSON file specified. Selected data not saved to a file.")

    print("\nScript finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subset selection script using various methods.")
    parser.add_argument("--embedding_checkpoint", type=str, default="/disk/dingding/embeddings_checkpoint_alpaca.npy",
                        help="Path to embeddings checkpoint numpy file (.npy)")
    parser.add_argument("--subset_size", type=int, default=1000, help="Desired subset size") # Reduced default for quicker testing
    parser.add_argument("--method", type=str, default="craig", choices=["craig", "pbc", "stochastic_greedy"],
                        help="Subset selection method to use.")
    
    # Stochastic Greedy specific arguments
    parser.add_argument("--sg_sample_size_Rt", type=int, default=100,
                        help="Sample size for R_t in Stochastic Greedy.")
    parser.add_argument("--sg_objective_type", type=str, default="facility_location", choices=["facility_location", "sum_sq_norms"],
                        help="Objective function for Stochastic Greedy.")

    parser.add_argument("--output_json", type=str, default="selected_data_subset.json",
                        help="Output JSON file to save selected data content.")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned", help="Hugging Face dataset name")
    parser.add_argument("--dataset_revision", type=str, default=None, help="Dataset revision (e.g., a git commit hash)")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use (e.g., 'train', 'validation')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("Current Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 30)

    main(args)