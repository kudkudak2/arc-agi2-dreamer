#!/usr/bin/env python3
"""
ARC Solver Evaluation Script

This script evaluates the ARC solver on random puzzles from ARC-AGI datasets.
It tests both 'trivial' and 'real_world' prompt versions and displays results
in a real-time updating table.

Usage:
    python eval.py arc-agi-1    # Evaluate on ARC-AGI dataset
    python eval.py arc-agi-2    # Evaluate on ARC-AGI-2 dataset
    
    # With custom parameters:
    python eval.py arc-agi-1 --num_puzzles=5 --seed=123 --ntries=5
    
    # Get help:
    python eval.py --help

The script will:
1. Select 10 random puzzles (seeded for reproducibility)
2. Solve each puzzle with both 'trivial' and 'real_world' prompts
3. Check for existing solutions to avoid regeneration
4. Display a real-time updating table with accuracy results
5. Show final summary with overall accuracy
"""

import json
import os
import random
import glob
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import fire

# Import solver functions
from solver import main as solve_task, load_task

console = Console()

def get_evaluation_files(dataset: str) -> List[str]:
    """Get all JSON files from the specified dataset evaluation directory."""
    if dataset == "arc-agi-1":
        eval_dir = "/var/shared-space/kudkudak/my_research/ARC-AGI/data/evaluation"
    elif dataset == "arc-agi-2":
        eval_dir = "/var/shared-space/kudkudak/my_research/ARC-AGI-2/data/evaluation"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'arc-agi-1' or 'arc-agi-2'")
    
    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")
    
    json_files = glob.glob(os.path.join(eval_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {eval_dir}")
    
    return sorted(json_files)

def select_random_puzzles(json_files: List[str], num_puzzles: int = 10, seed: int = 42) -> List[str]:
    """Select random puzzles with seeding for reproducibility."""
    random.seed(seed)
    selected = random.sample(json_files, min(num_puzzles, len(json_files)))
    return selected

def check_existing_solution(task_file: str, prompt_version: str) -> Optional[Dict]:
    """Check if a solution already exists and return the result."""
    task_id = os.path.splitext(os.path.basename(task_file))[0]
    prompt_tag = 'real' if prompt_version in ('real_world', 'real') else 'trivial'
    prompt_tag = "real_v2" if prompt_version == "real_world_v2" else prompt_tag
    output_json_path = f"{task_id}.{prompt_tag}.output.json"
    
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r') as f:
                result = json.load(f)
            return result
        except (json.JSONDecodeError, KeyError):
            return None
    return None

def solve_puzzle(task_file: str, prompt_version: str, ntries: int = 3) -> Dict:
    """Solve a single puzzle and return the result."""
    # Check for existing solution first
    existing = check_existing_solution(task_file, prompt_version)
    if existing is not None:
        return existing
    
    # Solve the puzzle
    try:
        # Import and call the solver's main function
        from solver import main as solve_task
        solve_task(task_file, recalculate_cache=False, ntries=ntries, prompt_version=prompt_version)
        
        # Load the generated result
        task_id = os.path.splitext(os.path.basename(task_file))[0]
        prompt_tag = 'real' if prompt_version in ('real_world', 'real') else 'trivial'
        prompt_tag = "real_v2" if prompt_version == "real_world_v2" else prompt_tag
        output_json_path = f"{task_id}.{prompt_tag}.output.json"
        
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r') as f:
                result = json.load(f)
            return result
        else:
            error_msg = "Output file not generated"
            console.print(f"[bold red]Error: {error_msg}[/bold red]")
            console.print(f"Task file: {task_file}")
            console.print(f"Prompt version: {prompt_version}")
            console.print(f"Expected output: {output_json_path}")
            raise RuntimeError(error_msg)
    except Exception as e:
        console.print(f"[bold red]Error solving puzzle: {e}[/bold red]")
        console.print(f"Task file: {task_file}")
        console.print(f"Prompt version: {prompt_version}")
        console.print(f"Error type: {type(e).__name__}")
        console.print(f"Error details: {str(e)}")
        raise

def create_results_table(results: List[Dict], dataset: str) -> Table:
    """Create a rich table showing evaluation results."""
    table = Table(title=f"ARC Solver Evaluation - {dataset.upper()}")
    table.add_column("Puzzle", style="cyan", no_wrap=True)
    table.add_column("Trivial", justify="center", style="green")
    table.add_column("Real World", justify="center", style="blue")
    table.add_column("Both Correct", justify="center", style="bold green")
    
    trivial_correct = 0
    real_world_correct = 0
    both_correct = 0
    total_puzzles = len(results) // 2  # Each puzzle has 2 results (trivial + real_world)
    
    # Group results by puzzle
    puzzle_results = {}
    for result in results:
        task_file = result.get("task_file", "")
        puzzle_id = os.path.splitext(os.path.basename(task_file))[0]
        prompt_version = result.get("prompt_version", "")
        
        if puzzle_id not in puzzle_results:
            puzzle_results[puzzle_id] = {}
        puzzle_results[puzzle_id][prompt_version] = result
    
    # Add rows to table
    for puzzle_id, versions in puzzle_results.items():
        trivial_result = versions.get("trivial", {})
        real_world_result = versions.get("real_world", {})
        
        trivial_correct_flag = trivial_result.get("correct", False)
        real_world_correct_flag = real_world_result.get("correct", False)
        both_correct_flag = trivial_correct_flag and real_world_correct_flag
        
        if trivial_correct_flag:
            trivial_correct += 1
        if real_world_correct_flag:
            real_world_correct += 1
        if both_correct_flag:
            both_correct += 1
        
        table.add_row(
            puzzle_id,
            "✓" if trivial_correct_flag else "✗",
            "✓" if real_world_correct_flag else "✗",
            "✓" if both_correct_flag else "✗"
        )
    
    # Add summary row
    table.add_section()
    table.add_row(
        "[bold]SUMMARY[/bold]",
        f"{trivial_correct}/{total_puzzles}",
        f"{real_world_correct}/{total_puzzles}",
        f"{both_correct}/{total_puzzles}"
    )
    
    return table

def main(dataset: str, num_puzzles: int = 10, seed: int = 42, ntries: int = 3):
    """
    Evaluate the ARC solver on random puzzles from the specified dataset.
    
    Args:
        dataset: Either 'arc-agi-1' or 'arc-agi-2'
        num_puzzles: Number of random puzzles to evaluate (default: 10)
        seed: Random seed for reproducible selection (default: 42)
        ntries: Number of retry attempts for code generation (default: 3)
    """
    console.print(f"[bold blue]Starting ARC Solver Evaluation[/bold blue]")
    console.print(f"Dataset: {dataset}")
    console.print(f"Puzzles: {num_puzzles}")
    console.print(f"Seed: {seed}")
    console.print(f"Retries: {ntries}")
    console.print()
    
    try:
        # Get evaluation files
        json_files = get_evaluation_files(dataset)
        console.print(f"Found {len(json_files)} puzzles in evaluation directory")
        
        if len(json_files) == 0:
            console.print("[bold red]No JSON files found in evaluation directory![/bold red]")
            return
        
        # Select random puzzles
        selected_puzzles = select_random_puzzles(json_files, num_puzzles, seed)
        console.print(f"Selected {len(selected_puzzles)} random puzzles")
        console.print()
        
        # Initialize results
        results = []
        
        # Create initial table
        table = create_results_table(results, dataset)
        
        with Live(table, refresh_per_second=4) as live:
            # Solve each puzzle with both prompt versions
            for i, puzzle_file in enumerate(selected_puzzles):
                puzzle_id = os.path.splitext(os.path.basename(puzzle_file))[0]
                
                try:
                    # Solve with trivial prompt
                    console.print(f"[yellow]Solving {puzzle_id} with trivial prompt...[/yellow]")
                    trivial_result = solve_puzzle(puzzle_file, "trivial", ntries)
                    results.append(trivial_result)
                    
                    # Update table
                    table = create_results_table(results, dataset)
                    live.update(table)
                    
                    # Solve with real_world prompt
                    console.print(f"[yellow]Solving {puzzle_id} with real_world prompt...[/yellow]")
                    real_world_result = solve_puzzle(puzzle_file, "real_world", ntries)
                    results.append(real_world_result)
                    
                    # Update table
                    table = create_results_table(results, dataset)
                    live.update(table)
                    
                    console.print(f"[green]Completed puzzle {i+1}/{len(selected_puzzles)}[/green]")
                    console.print()
                    
                except Exception as e:
                    console.print(f"[bold red]Evaluation stopped due to error in puzzle {puzzle_id}[/bold red]")
                    console.print(f"Error: {e}")
                    console.print(f"Puzzle file: {puzzle_file}")
                    console.print()
                    console.print("[bold red]Evaluation terminated early due to solver error.[/bold red]")
                    return
        
        # Final results
        console.print(Panel.fit(
            f"[bold green]Evaluation Complete![/bold green]\n"
            f"Dataset: {dataset.upper()}\n"
            f"Puzzles evaluated: {len(selected_puzzles)}\n"
            f"Total attempts: {len(results)}",
            title="Final Results"
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error during evaluation: {e}[/bold red]")
        raise

if __name__ == "__main__":
    fire.Fire(main)
