import json
import base64
import requests
import os
import hashlib
import copy
import traceback
import time
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
from rich.console import Console
from diskcache import Cache
from visualize import plot_task

CACHE_DIR = os.environ.get("ARC_CACHE_DIR", os.path.join(os.path.dirname(__file__), ".arc_cache"))
CACHE = Cache(CACHE_DIR)
console = Console()

def load_task(task_path):
    """Load task from JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_output_text(response_json: dict) -> str:
    # Cookbook-style: walk response.output and collect .content[].text
    parts = []
    for item in response_json.get("output", []):
        for c in item.get("content", []):
            txt = c.get("text")
            if txt:
                parts.append(txt)
    # Some server versions also return a convenience "output_text"
    if not parts and isinstance(response_json.get("output_text"), str):
        return response_json["output_text"].strip()
    return "\n".join(parts).strip()

def _compute_prompt_cache_key_text(prompt: str, model: str | None) -> str:
    model_id = model or "gpt-5-2025-08-07"
    # Use raw prompt string as the cache key (no hashing)
    return f"openai:{model_id}:prompt={prompt}"

def _compute_prompt_cache_key_image(image_path: str, prompt: str, model: str | None) -> str:
    # Key by raw prompt (not hash) as requested. Include image path for disambiguation.
    model_id = model or "gpt-5-2025-08-07"
    return f"openai:{model_id}:image={image_path}:prompt={prompt}"

def submit_to_openai_api(image_path: str, prompt: str, api_key: str = None, model: str = None, recalculate_cache=False):
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or pass api_key.")
    # Use the exact model you saw in your logs for reproducibility:
    # e.g. "gpt-5-2025-08-07"; otherwise "gpt-5"
    model = model or "gpt-5-2025-08-07"

    cache_key = _compute_prompt_cache_key_image(image_path, prompt, model)
    if cache_key in CACHE and not recalculate_cache:
        console.log(f"[dim]Cache hit (image+text): model={model}, image={image_path}[/dim]")
        return CACHE[cache_key]

    base64_image = encode_image_to_base64(image_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"},
                ],
            }
        ],
        # Keep hidden reasoning short so the model actually emits text
        "reasoning": {"effort": "high"},
        # Optional but supported in GPT-5 examples:
        "text": {"verbosity": "low"},
        # You can omit this; included here to bound the answer size
        "max_output_tokens": 100000, # 10k can be too short sometimes, 100k is a lot 
    }

    # Set timeout to 15 minutes (900 seconds) to avoid ReadTimeoutError
    start_ts = time.time()
    console.log(f"OpenAI vision+text request → model={model}, image={os.path.basename(image_path)}")
    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload,
        timeout=1800  # 30 minutes
    )
    # Show API error details verbatim on failure (helps pinpoint 400 root cause)
    if not resp.ok:
        # No try/except: let requests' .json() raise if response is not JSON
        detail = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw_text": resp.text}
        raise RuntimeError(f"OpenAI API {resp.status_code}: {json.dumps(detail, ensure_ascii=False)}")

    data = resp.json()
    output_text = extract_output_text(data)
    console.log(f"[green]OpenAI vision+text response ✓[/green] in {time.time()-start_ts:.2f}s, text_len={len(output_text)}")
    CACHE.set(cache_key, (output_text, data))
    return output_text, data

def submit_openai_text(prompt: str, api_key: str = None, model: str = None):
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or pass api_key.")
    model = model or "gpt-5-2025-08-07"

    cache_key = _compute_prompt_cache_key_text(prompt, model)
    if cache_key in CACHE:
        console.log("[dim]Cache hit (text-only)[/dim]")
        return CACHE[cache_key]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        "reasoning": {"effort": "high"},
        "text": {"verbosity": "low"},
        "max_output_tokens": 100000, # its a lot actually.
    }

    start_ts = time.time()
    console.log(f"OpenAI text-only request → model={model}")
    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload,
        timeout=900
    )
    if not resp.ok:
        detail = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw_text": resp.text}
        raise RuntimeError(f"OpenAI API {resp.status_code}: {json.dumps(detail, ensure_ascii=False)}")

    data = resp.json()
    output_text = extract_output_text(data)
    console.log(f"[green]OpenAI text-only response ✓[/green] in {time.time()-start_ts:.2f}s, code_len={len(output_text)}")
    CACHE.set(cache_key, (output_text, data))
    return output_text, data

def build_codegen_prompt(initial_reasoning: str, task: Dict[str, Any]) -> str:
    train_pairs = task["train"]
    test_inputs = [p["input"] for p in task.get("test", [])]
    return (
        "You are given an ARC task. Based on the reasoning below, write Python 3 code that implements the transformation from input grid to output grid.\n"
        "Constraints:\n"
        "- Provide a function named predict_output(input_grid) -> list[list[int]].\n"
        "- Do not read files or network; be deterministic.\n"
        "- Only output valid Python code, no markdown fences, no explanation.\n\n"
        f"Reasoning:\n{initial_reasoning}\n\n"
        f"Training pairs (JSON):\n{json.dumps(train_pairs)}\n\n"
        f"Test inputs (JSON):\n{json.dumps(test_inputs)}\n\n"
        "Implement predict_output so that it reproduces all training outputs when applied to the training inputs."
    )

def build_first_prompt(task_without_test_output: Dict[str, Any], version: str) -> str:
    if version == "trivial":
        return (
            "You are given ARC training input/output pairs and a test input. "
            "Predict the output for the test input. Be deterministic; infer a rule mapping input to output.\n\n"
            f"Task (JSON without test outputs): {json.dumps(task_without_test_output)}"
        )

    if version == "baseline":
        return (
            "You are given ARC training input/output pairs and test inputs. "
            "For each test input, predict the corresponding output. "
            "Be deterministic; infer a rule mapping input to output from the training examples.\n\n"
            f"Task (JSON without test outputs): {json.dumps(task_without_test_output)}\n\n"
            "Return your predictions as a JSON array where each element corresponds to the test input at the same index. "
            "Each prediction should be a 2D array (list of lists) representing the output grid.\n\n"
            "Example format:\n"
            "If you have 2 test inputs, return:\n"
            "[\n"
            "  [[0, 1, 2], [3, 4, 5]],\n"
            "  [[1, 0, 1], [0, 1, 0]]\n"
            "]\n\n"
            "Where the first array is the prediction for the first test input, and the second array is the prediction for the second test input."
        )

    if version == 'real_world_v2':
        return (
"""
Your goal is to build a story (a scene with objects and how they will change in time) that explains all changes in the training input/output images and apply it to the test inputs.

Your job: from training input/output images, infer a single simple world-story that explains all changes, then apply that same story to the test inputs. 

How to think (human style): 

1. See a scene with a story, not pixels. Name the setting (sky, water, floor, canvas), the objects (clouds, islands, ladders, bricks, stains, windows), and their roles (background, actor, marker, wall, paint). Something will happen soon!
2. Collect invariants & cues. Colors-as-roles (background vs ink/marker), borders as walls/frames, gravity = vertical, contact = sticking/painting, symmetry, repetition, counting, enclosure, ordering. 
3. Tell the story that fits every train pair. Examples: clouds drift right until a wall; paint spreads through connected water; the tallest stack wins and repaints the row it touches; each window shows a rotated copy of its symbol. 
4. State the law crisply. Turn the story into one clear rule with triggers and outcomes: “If an actor touches a marker, recolor the actor with the marker’s color; otherwise leave it.” Prefer one sentence; allow short tie-breakers (e.g., choose largest, leftmost, first in reading order).
5. Mentally simulate. Apply the law to each test input step by step (move, paint, merge, mirror, count…). 
6. Keep the same story; do not invent new rules for tests. 
7. Self-check. Verify the story explains all train pairs exactly; if not, revise. It might be you are missing some contextualization

Key things about rules: 
A) rules are often contextual
B) rules are often compositional
C) symbols will have often semantic meaning. 

Note that A-C is just like in the real world where what happens next depends on the context! 
"""
        )

    # default: real_world
    return (
        "The image shows examples of input (top) output (bottom) pairs for 3 train examples. "
        "Do not just predict the output for the fourth test example. Give a high-level explanation instead of how to map input situations into output.\n\n"
        "Think about the images as encoding a real world situation in an abstract way. "
        "The top is at some time and bottom is after some time.\n\n"
        "Think about a simple story or explanation for these. Remember it has to deterministically allow to predict output from input image\n\n"
        f"Here is also representation of the task as simple JSON: {json.dumps(task_without_test_output)}\n"
        "Remember to think about it as abstract depiction of a real world situation."
    )

def validate_predictions_against_test_outputs(predictions: List[List[List[int]]], task: Dict[str, Any]) -> Tuple[bool, str | None]:
    """Validate predictions against expected test outputs."""
    test_cases = task.get("test", [])
    if len(predictions) != len(test_cases):
        return False, f"Expected {len(test_cases)} predictions, got {len(predictions)}"
    
    for i, (prediction, test_case) in enumerate(zip(predictions, test_cases)):
        expected_output = test_case.get("output")

        if expected_output is None:
            raise ValueError(f"Expected output is None for test case {i}")
        

        if prediction != expected_output:
            return False, json.dumps({
                "type": "test_mismatch",
                "test_index": i,
                "test_input": test_case["input"],
                "expected_output": expected_output,
                "predicted_output": prediction,
            })
    
    return True, None

def execute_and_validate_generated_code(code_str: str, task: Dict[str, Any]) -> Tuple[bool, str | None, List[List[List[int]]] | None]:
    namespace: Dict[str, Any] = {}
    try:
        console.log(f"Executing generated code (chars={len(code_str)}) …")
        exec(code_str, namespace, namespace)
        predict_fn = namespace.get("predict_output")
        if not callable(predict_fn):
            console.log("[yellow]predict_output not found in generated code[/yellow]")
            return False, "predict_output function not found", None

        for pair in task["train"]:
            predicted = predict_fn(pair["input"])
            if predicted != pair["output"]:
                console.log("[yellow]Mismatch on a training pair; will refine code[/yellow]")
                return False, json.dumps({
                    "type": "mismatch",
                    "failing_input": pair["input"],
                    "expected_output": pair["output"],
                    "predicted_output": predicted,
                }), None

        predictions: List[List[List[int]]] = []
        for pair in task.get("test", []):
            predictions.append(predict_fn(pair["input"]))
        console.log("[green]Generated code passed all training pairs ✓[/green]")
        
        # Validate against expected test outputs
        # test_valid, test_error = validate_predictions_against_test_outputs(predictions, task)
        # if not test_valid:
        #     console.log(f"[yellow]Generated code failed test validation: {test_error}[/yellow]")
        #     return False, test_error, predictions
        
        console.log("[green]Generated code passed all test validations ✓[/green]")
        return True, None, predictions
    except Exception as e:
        tb = traceback.format_exc()
        console.log("[bold red]Runtime error while executing generated code[/bold red]")
        console.log(tb)
        return False, "traceback\n" + tb, None

def refine_code_prompt(previous_prompt: str, code_str: str, error_text: str, task: Dict[str, Any]) -> str:
    return (
        "Your previous attempt did not pass. Fix it.\n"
        "It might have not passed due to the story being incorrect e.g. too general. Please consider revising the story! \n"
        "Only output valid Python code (no fences, no commentary).\n\n"
        f"Previous prompt:\n{previous_prompt}\n\n"
        f"Previous code:\n{code_str}\n\n"
        f"Error / mismatch details:\n{error_text}\n\n"
        f"Training pairs (for reference):\n{json.dumps(task['train'])}"
    )

def extract_text(response_json):
    # Responses API returns an "output" list with structured content
    out = []
    for item in response_json.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out.append(c.get("text", ""))
    return "\n".join(out).strip()

def parse_baseline_predictions(response_text: str, task: Dict[str, Any]) -> Tuple[bool, str | None, List[List[List[int]]] | None]:
    """Parse baseline predictions from LLM response and validate them."""
    try:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if not json_match:
            return False, "No JSON array found in response", None
        
        predictions_json = json_match.group(0)
        predictions = json.loads(predictions_json)
        
        # Validate predictions format
        if not isinstance(predictions, list):
            return False, "Predictions must be a JSON array", None
        
        test_inputs = [p["input"] for p in task.get("test", [])]
        if len(predictions) != len(test_inputs):
            return False, f"Expected {len(test_inputs)} predictions, got {len(predictions)}", None
        
        # Validate each prediction
        for i, pred in enumerate(predictions):
            if not isinstance(pred, list):
                return False, f"Prediction {i} must be a list", None
            for j, row in enumerate(pred):
                if not isinstance(row, list):
                    return False, f"Prediction {i}, row {j} must be a list", None
                for k, cell in enumerate(row):
                    if not isinstance(cell, int):
                        return False, f"Prediction {i}, row {j}, cell {k} must be an integer", None
        
        # Check against training pairs
        for pair in task["train"]:
            # Find matching test input (if any)
            for i, test_input in enumerate(test_inputs):
                if test_input == pair["input"]:
                    if predictions[i] != pair["output"]:
                        return False, json.dumps({
                            "type": "training_mismatch",
                            "failing_input": pair["input"],
                            "expected_output": pair["output"],
                            "predicted_output": predictions[i],
                        }), None
                    break
        
        # Validate against expected test outputs
        test_valid, test_error = validate_predictions_against_test_outputs(predictions, task, no_test=True)
        if not test_valid:
            return False, test_error, None
        
        return True, None, predictions
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in predictions: {e}", None
    except Exception as e:
        return False, f"Error parsing predictions: {e}", None

def main(task_file: str = '../ARC-AGI/data/training/d631b094.json', recalculate_cache=False, ntries: int = 3, prompt_version: str = 'real_world'):
    # Load task from JSON file
    task_without_test_output = load_task(task_file)
    for ex in task_without_test_output['test']:
        assert ex['output'] is not None, f"Expected output is None for test case {ex}"
        del ex['output']
    
    console.log(f"Starting ARC solve pipeline (task_file={task_file}, ntries={ntries}, prompt_version={prompt_version})")
    task_id = os.path.splitext(os.path.basename(task_file))[0]
    prompt_tag = 'real' if prompt_version in ('real_world', 'real') else 'trivial'
    prompt_tag = "real_v2" if prompt_version == "real_world_v2" else prompt_tag
    prompt_tag = "baseline" if prompt_version == "baseline" else prompt_tag
    
    # Visualize inputs (without test outputs) for context only
    plot_task(task_without_test_output, show_test_output=False)
    task_image_path = f"{task_id}.{prompt_tag}.task.png"
    plt.savefig(task_image_path)
    plt.close()
    console.log(f"Saved task visualization → {task_image_path}")

    # First LLM call (vision + text) to get reasoning/description
    prompt1 = build_first_prompt(task_without_test_output, prompt_version)
    first_text, first_raw = submit_to_openai_api(task_image_path, prompt1, recalculate_cache=recalculate_cache)
    console.log(f"First call (vision+text) produced description_len={len(first_text)}")
    console.log("First call output_text:")
    console.log(first_text)

    # Handle baseline prompt version (no code generation)
    if prompt_version == "baseline":
        console.log("Using baseline mode - parsing predictions directly from LLM response")
        success, error_msg, predictions = parse_baseline_predictions(first_text, task_without_test_output)
        code_attempts = [{
            "attempt": 1,
            "prompt": prompt1,
            "code": None,
            "raw_response": first_raw,
            "error": error_msg,
            "passed_train": success,
        }]
        if success:
            console.log("[green]Baseline predictions validated successfully ✓[/green]")
        else:
            console.log(f"[yellow]Baseline predictions failed validation: {error_msg}[/yellow]")
    else:
        # Second LLM call: code generation based on first_text and task
        code_attempts: List[Dict[str, Any]] = []
        code_prompt = build_codegen_prompt(first_text, task_without_test_output)
        attempt = 0
        success = False
        predictions: List[List[List[int]]] | None = None
        while attempt < max(1, int(ntries)) and not success:
            attempt += 1
            console.log(f"Codegen attempt {attempt}/{ntries}")
            code_text, code_raw = submit_openai_text(code_prompt)
            ok, err, preds = execute_and_validate_generated_code(code_text, task_without_test_output)
            code_attempts.append({
                "attempt": attempt,
                "prompt": code_prompt,
                "code": code_text,
                "raw_response": code_raw,
                "error": err,
                "passed_train": ok,
            })
            if ok:
                success = True
                predictions = preds
                console.log("[green]Codegen succeeded on training pairs ✓[/green]")
                break
            # refine
            code_prompt = refine_code_prompt(code_prompt, code_text, err or "unknown error", task_without_test_output)
            console.log(f"Refined prompt for next attempt (len={len(code_prompt)})")

    # Prepare visualization with predictions (if any)
    viz_task = json.loads(json.dumps(task_without_test_output))
    if predictions is not None:
        for i, pred in enumerate(predictions):
            if i < len(viz_task.get("test", [])):
                viz_task["test"][i]["output"] = pred
    plot_task(viz_task, show_test_output=True)
    predicted_image_path = f"{task_id}.{prompt_tag}.output.png"
    plt.savefig(predicted_image_path)
    plt.close()
    console.log(f"Saved predicted visualization → {predicted_image_path}")

    # Determine if predictions are correct by validating against test outputs
    task = load_task(task_file) # Load for the first time with outputs
    correct = False
    if predictions is not None:
        test_valid, _ = validate_predictions_against_test_outputs(predictions, task)
        correct = test_valid
    
    # Save detailed JSON
    out_json = {
        "first_call": {
            "prompt": prompt1,
            "output_text": first_text,
            "raw_response": first_raw,
        },
        "code_attempts": code_attempts,
        "predictions": predictions,
        "ntries": ntries,
        "success": success,
        "correct": correct,
    }
    out_json["task_file"] = task_file
    out_json["task_id"] = task_id
    out_json["prompt_version"] = prompt_version
    out_json["prompt_tag"] = prompt_tag
    out_json_path = f"{task_id}.{prompt_tag}.output.json"
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    console.log(f"Saved outputs → {out_json_path}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
