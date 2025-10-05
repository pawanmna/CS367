import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

"""
=============================================================================
RAAG BHAIRAV THEORY & CONSTRAINTS
=============================================================================
"""
"""
Raag Bhairav Structure:
- Thaat: Bhairav
- Jati: Audav-Sampurna (5 notes ascending, 7 descending)
- Vadi (King): Dha (6th note)
- Samvadi (Queen): Re komal (flat 2nd)
- Time: Early morning (6-9 AM)
- Mood: Devotional, serious, meditative

Scale (Aroha - Ascending): S r G M P d N S'
Scale (Avaroha - Descending): S' N d P M G r S

Where:
- S = Sa (Tonic)
- r = Re komal (Flat 2nd)
- G = Ga (Major 3rd)
- M = Ma (Perfect 4th)
- P = Pa (Perfect 5th)
- d = Dha komal (Flat 6th)
- N = Ni (Major 7th)
- S' = Upper Sa

Important Phrases (Pakad):
- S r G M P, d P M G r S
- d P M G, r G M P d
- N S' r' S' d P M G
- M G r, S r G M P d
"""

"""
Note mapping dictionary: maps note names to semitones from Sa (tonic).
"""
NOTE_MAP = {
    'S': 0,
    'r': 1,
    'G': 4,
    'M': 5,
    'P': 7,
    'd': 8,
    'N': 11,
    "S'": 12,
    "r'": 13,
    "G'": 16,
}

"""
Allowed notes for ascending (aroha) scale in Raag Bhairav.
"""
AROH_NOTES = ['S', 'r', 'G', 'M', 'P', 'd', 'N', "S'"]
"""
Allowed notes for descending (avaroha) scale in Raag Bhairav.
"""
AVAROHA_NOTES = ["S'", 'N', 'd', 'P', 'M', 'G', 'r', 'S']
"""
All possible notes used in Raag Bhairav, including upper octave extensions.
"""
ALL_NOTES = ['S', 'r', 'G', 'M', 'P', 'd', 'N', "S'", "r'", "G'"]

"""
Important characteristic phrases (pakad) that define Raag Bhairav's identity.
These are melodic patterns commonly used in performances.
"""
PAKAD_PHRASES = [
    ['S', 'r', 'G', 'M', 'P'],
    ['d', 'P', 'M', 'G', 'r', 'S'],
    ['d', 'P', 'M', 'G'],
    ['r', 'G', 'M', 'P', 'd'],
    ['N', "S'", 'd', 'P'],
    ['M', 'G', 'r', 'S'],
    ['P', 'd', 'P', 'M'],
    ["S'", 'd', 'P', 'M', 'G'],
    ['G', 'M', 'P', 'd', 'P'],
    ['r', 'S', 'r', 'G', 'M'],
    ['d', 'N', "S'", 'd', 'P'],
    ['M', 'P', 'd', 'N', "S'"],
]

"""
Weights for note preferences in melody generation.
Higher weights mean the note is more likely to be chosen.
Vadi (d) and Samvadi (r) have the highest weights.
"""
NOTE_WEIGHTS = {
    'S': 10,
    'r': 9,
    'G': 7,
    'M': 7,
    'P': 8,
    'd': 10,
    'N': 6,
    "S'": 8,
}

"""
List of forbidden note transitions that violate Raag Bhairav grammar.
These jumps should be avoided to maintain raga purity.
"""
FORBIDDEN_TRANSITIONS = [
    ('S', 'N'),
    ('r', 'P'),
    ('G', 'd'),
    ('N', 'r'),
]

"""
=============================================================================
GENETIC ALGORITHM COMPONENTS
=============================================================================
"""
# RAAG BHAIRAV THEORY & CONSTRAINTS
# =============================================================================
"""
Raag Bhairav Structure:
- Thaat: Bhairav
- Jati: Audav-Sampurna (5 notes ascending, 7 descending)
- Vadi (King): Dha (6th note)
- Samvadi (Queen): Re komal (flat 2nd)
- Time: Early morning (6-9 AM)
- Mood: Devotional, serious, meditative

Scale (Aroha - Ascending): S r G M P d N S'
Scale (Avaroha - Descending): S' N d P M G r S

Where:
- S = Sa (Tonic)
- r = Re komal (Flat 2nd)
- G = Ga (Major 3rd)
- M = Ma (Perfect 4th)
- P = Pa (Perfect 5th)
- d = Dha komal (Flat 6th)
- N = Ni (Major 7th)
- S' = Upper Sa

Important Phrases (Pakad):
- S r G M P, d P M G r S
- d P M G, r G M P d
- N S' r' S' d P M G
- M G r, S r G M P d
"""

"""
Note mapping dictionary: maps note names to semitones from Sa (tonic).
"""
NOTE_MAP = {
    'S': 0,    # Sa
    'r': 1,    # Re komal (flat 2nd)
    'G': 4,    # Ga (major 3rd)
    'M': 5,    # Ma (perfect 4th)
    'P': 7,    # Pa (perfect 5th)
    'd': 8,    # Dha komal (flat 6th)
    'N': 11,   # Ni (major 7th)
    "S'": 12,  # Upper Sa
    "r'": 13,  # Upper Re komal
    "G'": 16,  # Upper Ga
}

"""
Allowed notes for ascending (aroha) scale in Raag Bhairav.
"""
AROHA_NOTES = ['S', 'r', 'G', 'M', 'P', 'd', 'N', "S'"]
"""
Allowed notes for descending (avaroha) scale in Raag Bhairav.
"""
AVAROHA_NOTES = ["S'", 'N', 'd', 'P', 'M', 'G', 'r', 'S']
"""
All possible notes used in Raag Bhairav, including upper octave extensions.
"""
ALL_NOTES = ['S', 'r', 'G', 'M', 'P', 'd', 'N', "S'", "r'", "G'"]

"""
Important characteristic phrases (pakad) that define Raag Bhairav's identity.
These are melodic patterns commonly used in performances.
"""
PAKAD_PHRASES = [
    ['S', 'r', 'G', 'M', 'P'],           # Basic aroha pakad
    ['d', 'P', 'M', 'G', 'r', 'S'],       # Classic descending pakad
    ['d', 'P', 'M', 'G'],                 # Short descending
    ['r', 'G', 'M', 'P', 'd'],            # Ascending with vadi
    ['N', "S'", 'd', 'P'],                # Upper tetrachord
    ['M', 'G', 'r', 'S'],                 # Resolution phrase
    ['P', 'd', 'P', 'M'],                 # Vadi emphasis
    ["S'", 'd', 'P', 'M', 'G'],           # Upper descent
    ['G', 'M', 'P', 'd', 'P'],            # Vadi approach
    ['r', 'S', 'r', 'G', 'M'],            # Samvadi emphasis
    ['d', 'N', "S'", 'd', 'P'],           # Upper movement
    ['M', 'P', 'd', 'N', "S'"],           # Aroha to upper Sa
]

"""
Weights for note preferences in melody generation.
Higher weights mean the note is more likely to be chosen.
Vadi (d) and Samvadi (r) have the highest weights.
"""
NOTE_WEIGHTS = {
    'S': 10,   # Sa (tonic) - always important
    'r': 9,    # Re komal (Samvadi)
    'G': 7,
    'M': 7,
    'P': 8,    # Pa
    'd': 10,   # Dha komal (Vadi)
    'N': 6,
    "S'": 8,
}

"""
List of forbidden note transitions that violate Raag Bhairav grammar.
These jumps should be avoided to maintain raga purity.
"""
FORBIDDEN_TRANSITIONS = [
    ('S', 'N'),   # Skip too many notes
    ('r', 'P'),   # Skip intermediate notes
    ('G', 'd'),   # Skip intermediate notes
    ('N', 'r'),   # Wrong direction jump
]

# =============================================================================
# GENETIC ALGORITHM COMPONENTS
# =============================================================================

class BhairavMelody:
    def __init__(self, notes, durations=None):
        """
        notes: list of note names
        durations: list of duration values (in beats)
        """
        self.notes = notes
        if durations is None:
            # Default durations: mostly 1 beat, some longer notes
            self.durations = [random.choice([0.5, 1, 1, 1, 2]) for _ in notes]
        else:
            self.durations = durations
        self.fitness = 0.0
    
    def __len__(self):
        """
        Return the number of notes in the melody.
        """
        return len(self.notes)
    
    def copy(self):
        """
        Return a deep copy of the melody.
        """
        return BhairavMelody(self.notes[:], self.durations[:])

def create_random_melody(length=16):
    """
    Generate a random melody following basic Raag Bhairav constraints.

    The melody starts with Sa (tonic) and prefers stepwise motion.
    Notes are chosen with bias towards neighbors for smooth contour.
    The melody is ensured to end on a strong note like Sa or Pa.
    """
    notes = []
    current = 'S'
    notes.append(current)

    for _ in range(length - 1):
        candidates = ALL_NOTES[:]

        current_idx = NOTE_MAP[current]
        weighted_candidates = []
        for note in candidates:
            note_idx = NOTE_MAP[note]
            interval = abs(note_idx - current_idx)
            if interval <= 2:
                weight = 3
            elif interval <= 5:
                weight = 2
            else:
                weight = 1
            weighted_candidates.extend([note] * weight)

        next_note = random.choice(weighted_candidates)
        notes.append(next_note)
        current = next_note

    if notes[-1] not in ['S', 'P', "S'"]:
        notes[-1] = random.choice(['S', 'P', "S'"])
    
    return BhairavMelody(notes)

def fitness_function(melody):
    """
    Evaluate how well the melody follows Raag Bhairav grammar.

    Higher score indicates better adherence to raga rules.
    The score considers pakad phrases, vadi/samvadi emphasis,
    smooth contour, forbidden transitions, note variety, komal usage, etc.
    """
    score = 0
    notes = melody.notes
    
    # 1. Pakad phrase matching (very important) - 50 points
    pakad_score = 0
    pakad_found = set()
    for pakad in PAKAD_PHRASES:
        pakad_len = len(pakad)
        for i in range(len(notes) - pakad_len + 1):
            if notes[i:i+pakad_len] == pakad:
                pakad_id = tuple(pakad)
                if pakad_id not in pakad_found:
                    pakad_score += 25  # Strong bonus for each unique pakad
                    pakad_found.add(pakad_id)
            # Partial match (3 out of 4 notes, etc.)
            elif pakad_len >= 4:
                matches = sum(1 for j in range(pakad_len) if i+j < len(notes) and notes[i+j] == pakad[j])
                if matches >= pakad_len - 1:
                    pakad_score += 8
    score += min(pakad_score, 50)
    
    # 2. Vadi-Samvadi emphasis (25 points)
    note_counts = Counter(notes)
    vadi_presence = note_counts.get('d', 0)
    samvadi_presence = note_counts.get('r', 0)
    # Strong presence of both
    score += min(vadi_presence * 2.5, 12)
    score += min(samvadi_presence * 2.5, 13)
    
    # 3. Start and end on strong notes (15 points)
    if notes[0] == 'S':
        score += 8
    elif notes[0] in ['r', 'G']:
        score += 4
    if notes[-1] in ['S', 'P']:
        score += 7
    elif notes[-1] in ["S'", 'd']:
        score += 4
    
    # 4. Smooth melodic contour with emphasis on phrases (20 points)
    smooth_score = 0
    for i in range(len(notes) - 1):
        interval = abs(NOTE_MAP[notes[i+1]] - NOTE_MAP[notes[i]])
        if interval == 0:  # Same note (nyaas/rest)
            smooth_score += 1
        elif interval <= 2:  # Stepwise motion (best)
            smooth_score += 3
        elif interval <= 5:  # Small leaps
            smooth_score += 1.5
        else:  # Large leaps (slight penalty unless it's a characteristic jump)
            if (notes[i], notes[i+1]) in [('d', 'S'), ('S', 'd'), ('P', "S'"), ("S'", 'P')]:
                smooth_score += 1  # Acceptable large leaps in Bhairav
            else:
                smooth_score -= 0.5
    score += min(smooth_score, 20)
    
    # 5. Avoid forbidden transitions (penalty)
    for i in range(len(notes) - 1):
        transition = (notes[i], notes[i+1])
        if transition in FORBIDDEN_TRANSITIONS:
            score -= 15
    
    # 6. Note variety but not too random (15 points)
    unique_notes = len(set(notes))
    if 6 <= unique_notes <= 8:  # Good variety
        score += 15
    elif 5 <= unique_notes <= 9:
        score += 10
    elif unique_notes >= 4:
        score += 5
    
    # 7. Proper usage of komal notes (r and d) - essential for Bhairav (15 points)
    komal_usage = note_counts.get('r', 0) + note_counts.get('d', 0)
    total_notes = len(notes)
    komal_ratio = komal_usage / total_notes if total_notes > 0 else 0
    if 0.3 <= komal_ratio <= 0.5:  # Ideal range
        score += 15
    elif 0.2 <= komal_ratio <= 0.6:
        score += 10
    else:
        score += 5
    
    # 8. Rhythmic variety (10 points)
    duration_variety = len(set(melody.durations))
    if duration_variety >= 4:
        score += 10
    elif duration_variety >= 3:
        score += 7
    elif duration_variety >= 2:
        score += 4
    
    # 9. Ascending-descending balance (10 points)
    if len(notes) > 1:
        ascending = sum(1 for i in range(len(notes)-1) 
                       if NOTE_MAP[notes[i+1]] > NOTE_MAP[notes[i]])
        descending = sum(1 for i in range(len(notes)-1) 
                        if NOTE_MAP[notes[i+1]] < NOTE_MAP[notes[i]])
        
        if ascending + descending > 0:
            asc_ratio = ascending / (ascending + descending)
            if 0.35 <= asc_ratio <= 0.65:  # Good balance
                score += 10
            elif 0.25 <= asc_ratio <= 0.75:
                score += 6
    
    # 10. Bonus: Multiple different pakad phrases (10 points)
    if len(pakad_found) >= 3:
        score += 10
    elif len(pakad_found) >= 2:
        score += 5
    
    # 11. Sa-Pa-Sa structure (common in Bhairav) (5 points)
    melody_str = ' '.join(notes)
    if 'S P S' in melody_str or "P S' P" in melody_str:
        score += 5
    
    return max(score, 0)

def tournament_selection(population, tournament_size=3):
    """Select parent using tournament selection"""
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda m: m.fitness)

def crossover(parent1, parent2):
    """
    Perform crossover between two parent melodies.

    Creates a child by combining segments from both parents.
    Handles different lengths by using the minimum length.
    """
    if len(parent1) != len(parent2):
        # Handle different lengths
        min_len = min(len(parent1), len(parent2))
        point = random.randint(1, min_len - 1)
    else:
        point = random.randint(1, len(parent1) - 1)
    
    child_notes = parent1.notes[:point] + parent2.notes[point:]
    child_durations = parent1.durations[:point] + parent2.durations[point:]
    
    return BhairavMelody(child_notes, child_durations)

def mutate(melody, mutation_rate=0.2, generation=0, max_generations=300):
    """
    Mutate a melody with adaptive mutation rate based on generation.

    Higher mutation early in evolution, lower later.
    Mutations include changing notes, inserting pakad phrases, swapping notes, emphasizing vadi/samvadi.
    """
    melody = melody.copy()
    
    adaptive_rate = mutation_rate * (1 + (max_generations - generation) / max_generations)

    for i in range(len(melody.notes)):
        if random.random() < adaptive_rate:
            mutation_type = random.random()

            if mutation_type < 0.4:
                current_pitch = NOTE_MAP[melody.notes[i]]
                nearby_notes = [n for n in ALL_NOTES
                              if abs(NOTE_MAP[n] - current_pitch) <= 3]
                if nearby_notes:
                    melody.notes[i] = random.choice(nearby_notes)

            elif mutation_type < 0.8:
                pakad = random.choice(PAKAD_PHRASES)
                if i <= len(melody.notes) - len(pakad):
                    for j, note in enumerate(pakad):
                        melody.notes[i + j] = note

            elif mutation_type < 0.9 and i < len(melody.notes) - 1:
                melody.notes[i], melody.notes[i+1] = melody.notes[i+1], melody.notes[i]

            else:
                melody.notes[i] = random.choice(['d', 'r', 'd', 'r', 'P', 'M'])

    for i in range(len(melody.durations)):
        if random.random() < adaptive_rate * 0.5:
            if random.random() < 0.3:
                melody.durations[i] = random.choice([0.5, 0.5, 1])
            else:
                melody.durations[i] = random.choice([1, 1, 1, 2, 2, 3])
    
    return melody

def genetic_algorithm(pop_size=100, generations=200, melody_length=20):
    """
    Run the genetic algorithm to evolve Bhairav melodies.

    Initializes population, evaluates fitness, evolves over generations using selection, crossover, mutation.
    Returns the best melody, and fitness history.
    """

    print(f"[INFO] Initializing population of {pop_size} melodies...")
    population = [create_random_melody(melody_length) for _ in range(pop_size)]

    # Evaluate initial fitness
    for melody in population:
        melody.fitness = fitness_function(melody)

    best_fitness_history = []
    avg_fitness_history = []

    print(f"[INFO] Running genetic algorithm for {generations} generations...")

    for gen in range(generations):
        # Evaluate fitness
        for melody in population:
            melody.fitness = fitness_function(melody)

        # Track statistics
        fitnesses = [m.fitness for m in population]
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        best_melody = max(population, key=lambda m: m.fitness)

        if gen % 20 == 0:
            print(f"Gen {gen:3d} | Best: {best_fitness:.1f} | Avg: {avg_fitness:.1f} | "
                  f"Notes: {' '.join(best_melody.notes[:10])}...")

        # Create new population
        new_population = []

        # Elitism: keep top 10%
        elite_size = pop_size // 10
        sorted_pop = sorted(population, key=lambda m: m.fitness, reverse=True)
        new_population.extend([m.copy() for m in sorted_pop[:elite_size]])

        # Generate offspring
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            # Sometimes do crossover, sometimes clone best parent
            if random.random() < 0.8:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy() if parent1.fitness > parent2.fitness else parent2.copy()

            child = mutate(child, generation=gen, max_generations=generations)
            new_population.append(child)

        population = new_population

    # Final evaluation of fitness
    for melody in population:
        melody.fitness = fitness_function(melody)

    best_melody = max(population, key=lambda m: m.fitness)

    return best_melody, best_fitness_history, avg_fitness_history

# =============================================================================
# VISUALIZATION & OUTPUT
# =============================================================================

def visualize_melody(melody, title="Raag Bhairav Melody"):
    """Visualize the melody as a musical notation-style plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Pitch contour
    pitches = [NOTE_MAP[note] for note in melody.notes]
    positions = np.cumsum([0] + melody.durations[:-1])
    
    ax1.plot(positions, pitches, 'o-', linewidth=2, markersize=8, color='darkblue')
    ax1.set_ylabel('Pitch (Semitones from Sa)', fontsize=12, fontweight='bold')
    ax1.set_title(title + f' (Fitness: {melody.fitness:.1f})', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 17)
    
    # Add note labels
    for pos, pitch, note in zip(positions, pitches, melody.notes):
        ax1.text(pos, pitch + 0.5, note, ha='center', fontsize=10, fontweight='bold')
    
    # Add horizontal lines for each note
    note_lines = sorted(set(NOTE_MAP.values()))
    for nl in note_lines:
        ax1.axhline(nl, color='gray', linestyle='--', alpha=0.2)
    
    # Plot 2: Note sequence with durations
    colors = matplotlib.colormaps.get_cmap('viridis')(np.linspace(0, 1, len(ALL_NOTES)))
    note_to_color = {note: colors[i] for i, note in enumerate(ALL_NOTES)}
    
    current_pos = 0
    for i, (note, duration) in enumerate(zip(melody.notes, melody.durations)):
        color = note_to_color[note]
        ax2.barh(0, duration, left=current_pos, height=0.8, 
                color=color, edgecolor='black', linewidth=1.5)
        ax2.text(current_pos + duration/2, 0, note, 
                ha='center', va='center', fontsize=11, fontweight='bold')
        current_pos += duration
    
    ax2.set_xlabel('Time (beats)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Notes', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlim(0, sum(melody.durations))
    ax2.set_yticks([])
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_melody_analysis(melody):
    """Print detailed analysis of the generated melody"""
    print("\n" + "="*70)
    print("MELODY ANALYSIS")
    print("="*70)
    
    print(f"\nMelody Notes: {' '.join(melody.notes)}")
    print(f"Durations:    {' '.join([str(d) for d in melody.durations])}")
    print(f"Total Duration: {sum(melody.durations):.1f} beats")
    print(f"Fitness Score: {melody.fitness:.1f}")
    
    # Check for pakad phrases
    print("\n--- Pakad Phrases Detected ---")
    found_pakad = False
    for pakad in PAKAD_PHRASES:
        pakad_str = ' '.join(pakad)
        melody_str = ' '.join(melody.notes)
        if pakad_str in melody_str:
            print(f"✓ Found: {pakad_str}")
            found_pakad = True
    if not found_pakad:
        print("  (No complete pakad phrases found, but may contain partial phrases)")
    
    # Note statistics
    note_counts = Counter(melody.notes)
    print("\n--- Note Frequency ---")
    for note in sorted(note_counts.keys(), key=lambda n: NOTE_MAP[n]):
        count = note_counts[note]
        percentage = (count / len(melody.notes)) * 100
        marker = "★" if note in ['d', 'r'] else " "
        print(f"{marker} {note}: {count:2d} times ({percentage:5.1f}%)")
    
    print("\n--- Vadi-Samvadi Presence ---")
    print(f"Vadi (d):     {note_counts.get('d', 0)} times")
    print(f"Samvadi (r):  {note_counts.get('r', 0)} times")
    
    print("\n" + "="*70)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("RAAG BHAIRAV MELODY GENERATOR")
    print("Using Genetic Algorithm")
    print("="*70)
    
    # Run genetic algorithm
    best_melody, best_history, avg_history = genetic_algorithm(
        pop_size=100,
        generations=200,
        melody_length=28
    )
    
    # Print results
    print_melody_analysis(best_melody)
    
    # Visualize best melody
    fig1 = visualize_melody(best_melody, "Best Generated Melody - Raag Bhairav")
    
    # Plot fitness evolution
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_history, label='Best Fitness', linewidth=2, color='darkgreen')
    ax.plot(avg_history, label='Average Fitness', linewidth=2, color='orange', alpha=0.7)
    ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitness Score', fontsize=12, fontweight='bold')
    ax.set_title('Genetic Algorithm Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()
    
    print("\n[INFO] Melody generation complete!")
    print("[INFO] The generated melody follows Raag Bhairav grammar")
    print("[INFO] Look for characteristic phrases and note patterns")

