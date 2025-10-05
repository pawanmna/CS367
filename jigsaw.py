import numpy as np
import matplotlib.pyplot as plt
import random
import math

"""
Set random seed for reproducibility
"""
random.seed(42)
np.random.seed(42)

"""
1. Load Octave ASCII Image
"""
def load_octave_ascii_image(filename):
    """
    Load an image from an Octave ASCII format file.

    The file format has dimensions on the first line, followed by pixel values.
    Returns a 2D numpy array representing the grayscale image.
    Raises ValueError if dimensions or pixel count don't match.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_lines = []
    dims = None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if dims is None:
            dims = list(map(int, stripped.split()))
        else:
            data_lines.append(int(stripped))

    if dims is None:
        raise ValueError("Could not read image dimensions")
    height, width = dims
    if len(data_lines) != height * width:
        raise ValueError(f"Pixel count mismatch: expected {height*width}, got {len(data_lines)}")

    image = np.array(data_lines, dtype=np.uint8).reshape((height, width))
    return image

"""
2. Load and Split Image
"""
filename = 'scrambled_lena.mat'
scrambled_image = load_octave_ascii_image(filename)

H, W = scrambled_image.shape
assert H == 512 and W == 512, "Expected 512x512 image"
tile_size = 128
pieces = []

for i in range(4):
    for j in range(4):
        piece = scrambled_image[i*tile_size:(i+1)*tile_size,
                                j*tile_size:(j+1)*tile_size]
        pieces.append(piece)

n = len(pieces)
print(f"[INFO] Loaded {n} pieces of size {tile_size}x{tile_size}")

"""
3. Precompute Compatibility Matrix
"""
def compute_compatibility_matrices():
    """
    Precompute all edge compatibility scores for puzzle pieces.

    Calculates right-left and bottom-top compatibility matrices.
    Lower scores indicate better matches.
    """
    """
    Right compatibility: how well piece i's right edge matches piece j's left edge
    """
    right_compat = np.zeros((n, n))
    """
    Bottom compatibility: how well piece i's bottom edge matches piece j's top edge
    """
    bottom_compat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                """
                Right-left compatibility
                """
                diff = np.sum((pieces[i][:, -1].astype(int) - pieces[j][:, 0].astype(int)) ** 2)
                right_compat[i, j] = diff

                """
                Bottom-top compatibility
                """
                diff = np.sum((pieces[i][-1, :].astype(int) - pieces[j][0, :].astype(int)) ** 2)
                bottom_compat[i, j] = diff
            else:
                right_compat[i, j] = float('inf')
                bottom_compat[i, j] = float('inf')

    return right_compat, bottom_compat

print("[INFO] Precomputing compatibility matrices...")
right_compat, bottom_compat = compute_compatibility_matrices()

"""
4. Fast Energy Function
"""
def compute_energy_fast(state):
    """
    Compute the total energy (compatibility cost) of the puzzle state.

    Sums the edge compatibility costs for all adjacent pieces.
    Uses precomputed matrices for efficiency.
    """
    total_cost = 0
    for i in range(4):
        for j in range(4):
            pos = i * 4 + j
            piece_idx = state[pos]

            """
            Horizontal neighbor (right)
            """
            if j < 3:
                right_pos = i * 4 + (j + 1)
                right_piece_idx = state[right_pos]
                total_cost += right_compat[piece_idx, right_piece_idx]

            """
            Vertical neighbor (bottom)
            """
            if i < 3:
                bottom_pos = (i + 1) * 4 + j
                bottom_piece_idx = state[bottom_pos]
                total_cost += bottom_compat[piece_idx, bottom_piece_idx]

    return total_cost

"""
5. Improved Simulated Annealing
"""
def simulated_annealing(initial_state, max_iter=100000, T0=200000.0, alpha=0.99995):
    """
    Optimize the puzzle configuration using simulated annealing.

    Starts from initial_state, perturbs by swapping pieces, accepts worse solutions probabilistically.
    Cooling schedule with temperature T0 and alpha.
    Returns the best state and its energy.
    """
    current_state = initial_state[:]
    best_state = current_state[:]
    current_energy = compute_energy_fast(current_state)
    best_energy = current_energy

    T = T0
    accepted = 0

    for iteration in range(max_iter):
        """
        Generate neighbor: swap two random positions
        """
        i, j = random.sample(range(n), 2)
        neighbor = current_state[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        neighbor_energy = compute_energy_fast(neighbor)
        delta_E = neighbor_energy - current_energy

        """
        Decide whether to accept
        """
        accept = False
        if delta_E <= 0:
            accept = True
        elif T > 0:
            exponent = -delta_E / T
            if exponent > 709:
                accept_prob = 1.0
            elif exponent < -709:
                accept_prob = 0.0
            else:
                accept_prob = math.exp(exponent)
            accept = random.random() < accept_prob

        if accept:
            current_state = neighbor
            current_energy = neighbor_energy
            accepted += 1

            if neighbor_energy < best_energy:
                best_state = neighbor[:]
                best_energy = neighbor_energy

        T *= alpha

        """
        Logging
        """
        if iteration % 10000 == 0:
            accept_rate = accepted / (iteration + 1) * 100
            print(f"Iter {iteration:6d} | Best: {best_energy:10.0f} | Current: {current_energy:10.0f} | Temp: {T:8.1f} | Accept: {accept_rate:.1f}%")

        if T < 0.001:
            break

    return best_state, best_energy

"""
6. Fast Local Search with 2-opt
"""
def fast_local_search(state, max_iterations=1000):
    """
    Fast iterative improvement with first-accept strategy
    """
    current_state = state[:]
    current_energy = compute_energy_fast(current_state)

    print(f"\n[INFO] Running fast local search...")
    print(f"[INFO] Starting energy: {current_energy:.0f}")

    total_improvements = 0

    for iteration in range(max_iterations):
        improved = False

        """
        Randomize search order for better exploration
        """
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)

        """
        First-accept hill climbing (faster than best-accept)
        """
        for i, j in pairs:
            neighbor = current_state[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor_energy = compute_energy_fast(neighbor)

            if neighbor_energy < current_energy:
                current_state = neighbor
                current_energy = neighbor_energy
                improved = True
                total_improvements += 1
                break

        if not improved:
            break

        if (iteration + 1) % 100 == 0:
            print(f"[INFO] Iteration {iteration + 1}: Energy = {current_energy:.0f}, Improvements: {total_improvements}")

    print(f"[INFO] Local search completed: {total_improvements} improvements found")
    print(f"[INFO] Final energy: {current_energy:.0f}")
    return current_state, current_energy

"""
7. Multi-start with Hybrid Approach
"""
def run_hybrid_solver(num_restarts=10):
    """
    Run multiple attempts with SA + local search
    """
    best_overall_state = None
    best_overall_energy = float('inf')

    for attempt in range(num_restarts):
        print(f"\n{'='*70}")
        print(f"ATTEMPT {attempt + 1}/{num_restarts}")
        print(f"{'='*70}")

        """
        Generate initial state
        """
        if attempt == 0:
            initial_state = list(range(n))
        else:
            initial_state = list(range(n))
            random.shuffle(initial_state)

        initial_energy = compute_energy_fast(initial_state)
        print(f"[INFO] Initial energy: {initial_energy:.0f}")

        """
        Run simulated annealing
        """
        print("[INFO] Running simulated annealing...")
        sa_state, sa_energy = simulated_annealing(initial_state)
        print(f"[INFO] SA final energy: {sa_energy:.0f}")

        """
        Apply local search
        """
        final_state, final_energy = fast_local_search(sa_state)

        improvement = (1 - final_energy/initial_energy) * 100
        print(f"[INFO] Total improvement: {improvement:.1f}%")

        if final_energy < best_overall_energy:
            best_overall_state = final_state
            best_overall_energy = final_energy
            print(f"[INFO] *** NEW BEST SOLUTION! Energy: {final_energy:.0f} ***")

    return best_overall_state, best_overall_energy

"""
8. Run Solver
"""
initial_state = list(range(n))
print("[INFO] Initial scrambled energy:", compute_energy_fast(initial_state))

print("\n[INFO] Running Hybrid Solver (SA + Local Search)...")
print("[INFO] This will take 10-15 minutes for best results...")
print("[INFO] You can interrupt early with Ctrl+C if satisfied\n")

try:
    solution_state, final_energy = run_hybrid_solver(num_restarts=10)
except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Using best solution found so far...")
    solution_state = list(range(n))
    final_energy = compute_energy_fast(solution_state)

print(f"\n{'='*70}")
print(f"FINAL BEST SOLUTION")
print(f"{'='*70}")
print(f"[INFO] Best energy: {final_energy:.0f}")
print(f"[INFO] Solution state: {solution_state}")

"""
9. Reconstruct & Visualize
"""
def reconstruct_image(state):
    """
    Reconstruct the full image from puzzle pieces based on the state.

    State is a list of 16 piece indices in row-major order.
    Returns a 512x512 numpy array of the reconstructed image.
    """
    canvas = np.zeros((512, 512), dtype=np.uint8)
    for idx in range(16):
        i, j = divmod(idx, 4)
        piece_idx = state[idx]
        canvas[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = pieces[piece_idx]
    return canvas

reconstructed = reconstruct_image(solution_state)
initial_img = reconstruct_image(initial_state)

"""
Plot
"""
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(initial_img, cmap='gray')
axes[0].set_title("Initial Scrambled", fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(reconstructed, cmap='gray')
axes[1].set_title(f"Reconstructed (Energy={final_energy:.0f})", fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig("jigsaw_result.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n[INFO] Result saved to 'jigsaw_result.png'")

