import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
os.makedirs('output', exist_ok=True)

# ==========================================
# CLASS: Generic Hopfield Network
# ==========================================
class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train_hebbian(self, patterns):
        """Standard Hebbian Learning for Associative Memory"""
        print(f"Training on {len(patterns)} pattern(s)...")
        for p in patterns:
            p = p.reshape(-1, 1)
            self.weights += np.dot(p, p.T)
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.n_neurons

    def set_weights_manual(self, weight_matrix):
        """For optimization problems"""
        self.weights = weight_matrix
        np.fill_diagonal(self.weights, 0)

    def update_async(self, state, iterations=1000, bias=0):
        """
        Asynchronous update with BIAS.
        Bias is crucial for shifting the activation threshold.
        """
        s = state.copy()
        energy_log = []
        
        for _ in range(iterations):
            idx = np.random.randint(0, self.n_neurons)
            
            # Activation = Weighted Sum + Bias
            # If Activation >= 0, state becomes +1 (Active)
            # If Activation < 0, state becomes -1 (Inactive)
            activation = np.dot(self.weights[idx], s) + bias
            s[idx] = 1 if activation >= 0 else -1
            
            if _ % 100 == 0: 
                # Energy = -0.5 * sWs - bias*sum(s)
                # Simplified energy tracking
                e = -0.5 * np.dot(s.T, np.dot(self.weights, s))
                energy_log.append(e)
                
        return s, energy_log

# ==========================================
# PROBLEM 1: Error Correcting Capability
# ==========================================
def solve_error_correction():
    print("\n--- 1. Error Correcting Capability ---")
    size = 10
    # Create 'X' pattern
    pattern = -1 * np.ones((size, size))
    for i in range(size):
        pattern[i, i] = 1
        pattern[i, size-1-i] = 1
    
    flat_pattern = pattern.flatten()
    hn = HopfieldNetwork(n_neurons=size*size)
    hn.train_hebbian([flat_pattern])
    
    # Corrupt Pattern (20% noise)
    noisy_pattern = flat_pattern.copy()
    np.random.seed(42)
    noise_indices = np.random.choice(len(noisy_pattern), size=20, replace=False)
    noisy_pattern[noise_indices] *= -1
    
    # Recover (Standard Hopfield has Bias=0)
    recovered_state, _ = hn.update_async(noisy_pattern, iterations=2000, bias=0)
    
    # Visualize and SAVE
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(pattern, cmap='Greys'), ax[0].set_title("Original")
    ax[1].imshow(noisy_pattern.reshape(size, size), cmap='Greys'), ax[1].set_title("Corrupted")
    ax[2].imshow(recovered_state.reshape(size, size), cmap='Greys'), ax[2].set_title("Recovered")
    
    filename = "task1_error_correction.png"
    plt.savefig(filename)
    print(f"Observation: The network reduced energy to find the original pattern.")
    print(f"[Saved Image]: {filename}")
    plt.close()

# ==========================================
# PROBLEM 2: Eight-Rook Problem (FIXED)
# ==========================================
def solve_eight_rooks():
    print("\n--- 2. Eight-Rook Problem ---")
    N = 8
    neurons = N * N
    
    # 1. Define Weights (Inhibitory)
    weights = np.zeros((neurons, neurons))
    for i in range(neurons):
        r1, c1 = divmod(i, N)
        for j in range(neurons):
            r2, c2 = divmod(j, N)
            if i != j and (r1 == r2 or c1 == c2):
                weights[i, j] = -2 
    
    hn = HopfieldNetwork(neurons)
    hn.set_weights_manual(weights)
    
    # 2. Configuration for Success
    # BIAS Calculation:
    # Neighbors (Row+Col) = 14.
    # If k neighbors are active (+1), and (14-k) are empty (-1):
    # Input = k*(-2) + (14-k)*(+2) = 28 - 4k.
    # We want ON (+1) if k=0 (Safe). Input=28. Threshold: Bias > -28.
    # We want OFF (-1) if k=1 (Threat). Input=24. Threshold: Bias < -24.
    # Optimal Bias = -26.
    BIAS = -26.0
    
    # Initialization: Start with EMPTY board (-1). 
    # This forces the network to "place" rooks one by one in safe spots.
    init_state = -1 * np.ones(neurons)
    
    # Run Network
    final_state, _ = hn.update_async(init_state, iterations=3000, bias=BIAS)
    
    # Check Validity
    board = final_state.reshape(N, N)
    board01 = np.where(board==1, 1, 0) # Convert to 0/1 for logic check
    
    if np.sum(board01) == 8:
        # Check rows and cols
        rows = np.sum(board01, axis=1)
        cols = np.sum(board01, axis=0)
        if np.all(rows == 1) and np.all(cols == 1):
            print("Success! Found valid 8-Rook solution.")
            
            # Plot
            plt.figure(figsize=(5,5))
            # Map -1 to 0 for cleaner plot
            plot_board = board.copy()
            plot_board[plot_board == -1] = 0
            sns.heatmap(plot_board, linewidths=1, linecolor='black', cmap='Blues', cbar=False)
            plt.title("8-Rook Solution")
            
            filename = "task2_eight_rooks.png"
            plt.savefig(filename)
            print("Reason for weights: -2 inhibits conflicts. Bias -26 promotes placing rooks in empty rows.")
            print(f"[Saved Image]: {filename}")
            plt.close()
            return

    print("Failed to find solution (Unexpected).")

# ==========================================
# PROBLEM 3: TSP (10 Cities) Setup
# ==========================================
def solve_tsp_setup():
    print("\n--- 3. TSP (10 Cities) Setup ---")
    num_cities = 10
    total_neurons = num_cities * 10
    total_weights = total_neurons ** 2
    
    print(f"Total Neurons required: {total_neurons}")
    print(f"Total Weights required: {total_weights}")
    
    np.random.seed(42)
    coords = np.random.rand(num_cities, 2)
    plt.figure(figsize=(6,6))
    plt.scatter(coords[:,0], coords[:,1], c='red', s=100)
    for i, (x,y) in enumerate(coords):
        plt.text(x+0.02, y+0.02, f"City {i}")
    plt.title(f"TSP Setup: {num_cities} Cities")
    plt.xlabel("X Coordinate"); plt.ylabel("Y Coordinate")
    plt.grid(True)
    
    filename = "task3_tsp_setup.png"
    plt.savefig(filename)
    print(f"[Saved Image]: {filename}")
    plt.close()

if __name__ == "__main__":
    solve_error_correction()
    solve_eight_rooks()
    solve_tsp_setup()