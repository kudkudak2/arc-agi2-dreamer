import numpy as np
import json
import base64
import requests
from visualize import plot_task

def load_task(task_path):
    """Load task from JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)

def dsl_rotate_90(grid):
    return np.rot90(grid, 1)

def dsl_flip_horizontal(grid):
    return np.fliplr(grid)

def dsl_flip_vertical(grid):
    return np.flipud(grid)

# Our DSL is a dictionary mapping function names to functions
DSL = {
    'rotate_90': dsl_rotate_90,
    'flip_h': dsl_flip_horizontal,
    'flip_v': dsl_flip_vertical,
}

from itertools import product

def apply_program(grid, program):
    """Applies a sequence of DSL functions to a grid."""
    current_grid = np.array(grid)
    for func_name in program:
        current_grid = DSL[func_name](current_grid)
    return current_grid.tolist()

def find_program(task, max_depth=3):
    """Searches for a program that solves the task."""
    train_pairs = task['train']
    
    # Generate all possible programs up to max_depth
    for depth in range(1, max_depth + 1):
        for program_tuple in product(DSL.keys(), repeat=depth):
            program = list(program_tuple)
            is_solution = True
            # Verify the program against all training pairs
            for pair in train_pairs:
                input_grid = pair['input']
                expected_output = pair['output']
                predicted_output = apply_program(input_grid, program)
                
                if predicted_output != expected_output:
                    is_solution = False
                    break
            
            if is_solution:
                print(f"Found solution program: {program}")
                return program
    
    print("No solution found.")
    return None

# In solver.py
def solve_task(task):
    """Finds a program and applies it to test inputs."""
    program = find_program(task)
    if program is None:
        return # Return empty predictions if no solution found
    
    predictions = []
    for pair in task['test']:
        test_input = pair['input']
        predicted_output = apply_program(test_input, program)
        predictions.append(predicted_output)
    return predictions

def main():
    task_file = '../ARC-AGI/data/training/007bbfb7.json' # A simple rotation task
    task_id = task_file.split('/')[-1].replace('.json', '')
    task = load_task(task_file)
    
    predictions = solve_task(task)
    
    # Format for submission (simplified for one task)
    submission = {}
    if predictions:
        # ARC Prize allows multiple attempts, here we just submit one
        submission[task_id] = [{'attempt_1': pred, 'attempt_2': pred} for pred in predictions]

    with open('submission.json', 'w') as f:
        json.dump(submission, f, indent=4)
    print("submission.json created.")

if __name__ == '__main__':
    main()