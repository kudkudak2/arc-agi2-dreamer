import json
import base64
import requests
import os
import hashlib
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

def submit_to_openai_api(image_path: str, prompt: str, api_key: str = None, model: str = None):
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or pass api_key.")
    # Use the exact model you saw in your logs for reproducibility:
    # e.g. "gpt-5-2025-08-07"; otherwise "gpt-5"
    model = model or "gpt-5-2025-08-07"

    cache_key = _compute_prompt_cache_key_image(image_path, prompt, model)
    if cache_key in CACHE:
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
        "max_output_tokens": 10000,
    }

    # Set timeout to 15 minutes (900 seconds) to avoid ReadTimeoutError
    start_ts = time.time()
    console.log(f"OpenAI vision+text request → model={model}, image={os.path.basename(image_path)}")
    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload,
        timeout=900  # 15 minutes
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
        "max_output_tokens": 10000,
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
        "- Do not use external libraries (no numpy); use pure Python lists.\n"
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
        return True, None, predictions
    except Exception as e:
        tb = traceback.format_exc()
        console.log("[bold red]Runtime error while executing generated code[/bold red]")
        return False, "traceback\n" + tb, None

def refine_code_prompt(previous_prompt: str, code_str: str, error_text: str, task: Dict[str, Any]) -> str:
    return (
        "Your previous code did not pass. Fix it.\n"
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

def main(task_file: str = '../ARC-AGI/data/training/d631b094.json', ntries: int = 3, prompt_version: str = 'real_world'):
    # Load task from JSON file
    task = load_task(task_file)
    task_without_test_output = task.copy()
    for ex in task_without_test_output['test']:
        del ex['output']
    
    console.log(f"Starting ARC solve pipeline (task_file={task_file}, ntries={ntries}, prompt_version={prompt_version})")
    task_id = os.path.splitext(os.path.basename(task_file))[0]
    # Visualize inputs (without test outputs) for context only
    plot_task(task, show_test_output=False)
    task_image_path = f"{task_id}.task.png"
    plt.savefig(task_image_path)
    plt.close()
    console.log(f"Saved task visualization → {task_image_path}")

    # First LLM call (vision + text) to get reasoning/description
    prompt1 = build_first_prompt(task_without_test_output, prompt_version)
    first_text, first_raw = submit_to_openai_api(task_image_path, prompt1)
    console.log(f"First call (vision+text) produced description_len={len(first_text)}")
    console.log("First call output_text:")
    console.log(first_text)

    # Second LLM call: code generation based on first_text and task
    code_attempts: List[Dict[str, Any]] = []
    code_prompt = build_codegen_prompt(first_text, task)
    attempt = 0
    success = False
    predictions: List[List[List[int]]] | None = None
    while attempt < max(1, int(ntries)) and not success:
        attempt += 1
        console.log(f"Codegen attempt {attempt}/{ntries}")
        code_text, code_raw = submit_openai_text(code_prompt)
        ok, err, preds = execute_and_validate_generated_code(code_text, task)
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
        code_prompt = refine_code_prompt(code_prompt, code_text, err or "unknown error", task)
        console.log(f"Refined prompt for next attempt (len={len(code_prompt)})")

    # Prepare visualization with predictions (if any)
    viz_task = json.loads(json.dumps(task))
    if predictions is not None:
        for i, pred in enumerate(predictions):
            if i < len(viz_task.get("test", [])):
                viz_task["test"][i]["output"] = pred
    plot_task(viz_task, show_test_output=True)
    predicted_image_path = f"{task_id}.output.png"
    plt.savefig(predicted_image_path)
    plt.close()
    console.log(f"Saved predicted visualization → {predicted_image_path}")

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
    }
    out_json["task_file"] = task_file
    out_json["task_id"] = task_id
    out_json_path = f"{task_id}.output.json"
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    console.log(f"Saved outputs → {out_json_path}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
