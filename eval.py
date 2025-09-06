#!/usr/bin/env python3
"""
ARC Solver Evaluation Script

This script evaluates the ARC solver on random puzzles from ARC-AGI datasets.
It tests both 'trivial' and 'real_world' prompt versions and displays results
in a real-time updating table.

Usage:
    python eval.py arc-agi-1    # Evaluate on ARC-AGI dataset (all methods)
    python eval.py arc-agi-2    # Evaluate on ARC-AGI-2 dataset (all methods)
    
    # Evaluate specific method only:
    python eval.py arc-agi-1 --method=trivial
    python eval.py arc-agi-1 --method=real_world
    python eval.py arc-agi-1 --method=baseline
    
    # With custom parameters:
    python eval.py arc-agi-1 --num_puzzles=5 --seed=123 --ntries=5 --method=trivial
    
    # Get help:
    python eval.py --help

The script will:
1. Select 10 random puzzles (seeded for reproducibility)
2. Solve each puzzle with specified method(s) ('all', 'trivial', 'real_world', or 'baseline')
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
    prompt_tag = "baseline" if prompt_version == "baseline" else prompt_tag
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
    # existing = check_existing_solution(task_file, prompt_version)
    # if existing is not None:
    #     return existing
    
    # Solve the puzzle
    try:
        # Import and call the solver's main function
        from solver import main as solve_task
        solve_task(task_file, recalculate_cache=False, ntries=ntries, prompt_version=prompt_version)
        
        # Load the generated result
        task_id = os.path.splitext(os.path.basename(task_file))[0]
        prompt_tag = 'real' if prompt_version in ('real_world', 'real') else 'trivial'
        prompt_tag = "real_v2" if prompt_version == "real_world_v2" else prompt_tag
        prompt_tag = "baseline" if prompt_version == "baseline" else prompt_tag
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

def create_results_table(results: List[Dict], dataset: str, method: str = "all") -> Table:
    """Create a rich table showing evaluation results.
    
    Symbols used:
    - OK = Task solved successfully
    - FAIL = Task attempted but failed
    - ? = Task not yet attempted
    """
    # Determine which columns to show based on method
    if method == "all":
        table = Table(title=f"ARC Solver Evaluation - {dataset.upper()}")
        table.add_column("Puzzle", style="cyan", no_wrap=True)
        table.add_column("Trivial", justify="center", style="green", header_style="bold")
        table.add_column("Real World", justify="center", style="blue", header_style="bold")
        table.add_column("Baseline", justify="center", style="magenta", header_style="bold")
        table.add_column("All Correct", justify="center", style="bold green", header_style="bold")
        table.add_column("Completed", style="yellow", no_wrap=True)
    else:
        method_display = method.replace("_", " ").title()
        table = Table(title=f"ARC Solver Evaluation - {dataset.upper()} ({method_display})")
        table.add_column("Puzzle", style="cyan", no_wrap=True)
        table.add_column(method_display, justify="center", style="green", header_style="bold")
        table.add_column("Completed", style="yellow", no_wrap=True)
    
    trivial_correct = 0
    real_world_correct = 0
    baseline_correct = 0
    all_correct = 0
    completed_puzzles = []
    
    # Group results by puzzle
    puzzle_results = {}
    for result in results:
        task_file = result.get("task_file", "")
        puzzle_id = os.path.splitext(os.path.basename(task_file))[0]
        prompt_version = result.get("prompt_version", "")
        
        if puzzle_id not in puzzle_results:
            puzzle_results[puzzle_id] = {}
        puzzle_results[puzzle_id][prompt_version] = result
    
    # Calculate total puzzles from the grouped results
    total_puzzles = len(puzzle_results)
    
    # Add rows to table
    for puzzle_id, versions in puzzle_results.items():
        if method == "all":
            trivial_result = versions.get("trivial", {})
            real_world_result = versions.get("real_world", {})
            baseline_result = versions.get("baseline", {})
            
            trivial_correct_flag = trivial_result.get("correct", False)
            real_world_correct_flag = real_world_result.get("correct", False)
            baseline_correct_flag = baseline_result.get("correct", False)
            all_correct_flag = trivial_correct_flag and real_world_correct_flag and baseline_correct_flag
            
            # Check if puzzle is completed (all three versions attempted)
            is_completed = len(versions) == 3
            if is_completed and puzzle_id not in completed_puzzles:
                completed_puzzles.append(puzzle_id)
            
            if trivial_correct_flag:
                trivial_correct += 1
            if real_world_correct_flag:
                real_world_correct += 1
            if baseline_correct_flag:
                baseline_correct += 1
            if all_correct_flag:
                all_correct += 1
            
            # Show "OK" for correct, "FAIL" for attempted but failed, "?" for not attempted
            trivial_display = "OK" if trivial_correct_flag else ("?" if "trivial" not in versions else "FAIL")
            real_world_display = "OK" if real_world_correct_flag else ("?" if "real_world" not in versions else "FAIL")
            baseline_display = "OK" if baseline_correct_flag else ("?" if "baseline" not in versions else "FAIL")
            all_display = "OK" if all_correct_flag else ("?" if len(versions) < 3 else "FAIL")
            
            # Create completed puzzles list for this row
            completed_list = ", ".join(completed_puzzles) if completed_puzzles else "None"
            
            table.add_row(
                puzzle_id,
                trivial_display,
                real_world_display,
                baseline_display,
                all_display,
                completed_list
            )
        else:
            # Single method evaluation
            method_result = versions.get(method, {})
            method_correct_flag = method_result.get("correct", False)
            
            # Check if puzzle is completed (this method attempted)
            is_completed = method in versions
            if is_completed and puzzle_id not in completed_puzzles:
                completed_puzzles.append(puzzle_id)
            
            if method_correct_flag:
                if method == "trivial":
                    trivial_correct += 1
                elif method == "real_world":
                    real_world_correct += 1
                elif method == "baseline":
                    baseline_correct += 1
            
            # Show "OK" for correct, "FAIL" for attempted but failed, "?" for not attempted
            method_display = "OK" if method_correct_flag else ("?" if method not in versions else "FAIL")
            
            # Create completed puzzles list for this row
            completed_list = ", ".join(completed_puzzles) if completed_puzzles else "None"
            
            table.add_row(
                puzzle_id,
                method_display,
                completed_list
            )
    
    # Add legend row
    table.add_section()
    if method == "all":
        table.add_row(
            "[bold]LEGEND[/bold]",
            "[green]OK = Solved[/green]",
            "[red]FAIL = Failed[/red]", 
            "[yellow]? = Not Attempted[/yellow]",
            "",
            ""
        )
    else:
        table.add_row(
            "[bold]LEGEND[/bold]",
            "[green]OK = Solved[/green]",
            "[red]FAIL = Failed[/red]", 
            "[yellow]? = Not Attempted[/yellow]"
        )
        
    # Add summary row
    table.add_section()
    completed_list = ", ".join(completed_puzzles) if completed_puzzles else "None"
    
    if method == "all":
        table.add_row(
            "[bold]SUMMARY[/bold]",
            f"{trivial_correct}/{total_puzzles}",
            f"{real_world_correct}/{total_puzzles}",
            f"{baseline_correct}/{total_puzzles}",
            f"{all_correct}/{total_puzzles}",
            completed_list
        )
    else:
        method_correct = 0
        if method == "trivial":
            method_correct = trivial_correct
        elif method == "real_world":
            method_correct = real_world_correct
        elif method == "baseline":
            method_correct = baseline_correct
        
        table.add_row(
            "[bold]SUMMARY[/bold]",
            f"{method_correct}/{total_puzzles}",
            completed_list
        )
        
        # Add summary row
        table.add_section()
        completed_list = ", ".join(completed_puzzles) if completed_puzzles else "None"
        method_correct = trivial_correct if method == "trivial" else (real_world_correct if method == "real_world" else baseline_correct)
        table.add_row(
            "[bold]SUMMARY[/bold]",
            f"{method_correct}/{total_puzzles}",
            completed_list
        )
    
    return table

def main(dataset: str, num_puzzles: int = 10, seed: int = 42, ntries: int = 3, method: str = "all"):
    """
    Evaluate the ARC solver on random puzzles from the specified dataset.
    
    Args:
        dataset: Either 'arc-agi-1' or 'arc-agi-2'
        num_puzzles: Number of random puzzles to evaluate (default: 10)
        seed: Random seed for reproducible selection (default: 42)
        ntries: Number of retry attempts for code generation (default: 3)
        method: Which method to evaluate:
                - 'all': Evaluate all three methods (trivial, real_world, baseline)
                - 'trivial': Evaluate only the trivial prompt method
                - 'real_world': Evaluate only the real_world prompt method
                - 'baseline': Evaluate only the baseline prompt method
                (default: 'all')
    """
    # Validate method argument
    valid_methods = ["all", "trivial", "real_world", "baseline"]
    if method not in valid_methods:
        console.print(f"[bold red]Error: Invalid method '{method}'. Must be one of: {', '.join(valid_methods)}[/bold red]")
        return
    
    console.print(f"[bold blue]Starting ARC Solver Evaluation[/bold blue]")
    console.print(f"Dataset: {dataset}")
    console.print(f"Puzzles: {num_puzzles}")
    console.print(f"Seed: {seed}")
    console.print(f"Retries: {ntries}")
    console.print(f"Method: {method}")
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
        table = create_results_table(results, dataset, method)
        
        with Live(table, refresh_per_second=4) as live:
            # Determine which methods to evaluate
            if method == "all":
                methods_to_eval = ["trivial", "real_world", "baseline"]
            elif method in ["trivial", "real_world", "baseline"]:
                methods_to_eval = [method]
            else:
                raise ValueError(f"Invalid method: {method}. Use 'all', 'trivial', 'real_world', or 'baseline'")
            
            console.print(f"Evaluating methods: {', '.join(methods_to_eval)}")
            console.print()
            
            # Solve each puzzle with the specified method(s)
            for i, puzzle_file in enumerate(selected_puzzles):
                puzzle_id = os.path.splitext(os.path.basename(puzzle_file))[0]
                
                try:
                    for method_name in methods_to_eval:
                        console.print(f"[yellow]Solving {puzzle_id} with {method_name} prompt...[/yellow]")
                        result = solve_puzzle(puzzle_file, method_name, ntries)
                        results.append(result)
                        
                        # Update table
                        table = create_results_table(results, dataset, method)
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
