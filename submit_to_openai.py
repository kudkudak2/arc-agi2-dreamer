#!/usr/bin/env python3
"""Submit ARC task to OpenAI API for analysis."""

import os
import sys
import base64
import requests
from solver import load_task
from visualize import plot_task

def encode_image_to_base64(image_path):
    """Encode image to base64 string for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def submit_to_openai_api(api_key, image_path, prompt):
    """Submit request to OpenAI API with image."""
    base64_image = encode_image_to_base64(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

def main():
    # First, generate the visualization
    print("Generating task visualization...")
    task_file = '../ARC-AGI/data/training/007bbfb7.json'
    
    try:
        task = load_task(task_file)
        print(f"✓ Loaded task with {len(task['train'])} training examples")
        
        # Generate and save visualization
        plot_task(task)
        print("✓ Visualization saved as 'task.png'")
        
    except FileNotFoundError:
        print(f"✗ Task file not found: {task_file}")
        print("Please ensure the ARC-AGI repository is available at ../ARC-AGI/")
        return
    except Exception as e:
        print(f"✗ Error generating visualization: {e}")
        return
    
    # Check if image was created
    if not os.path.exists('task.png'):
        print("✗ Visualization file 'task.png' was not created")
        return
    
    print(f"\n✓ Image file size: {os.path.getsize('task.png')} bytes")
    
    # OpenAI API submission
    print("\n" + "="*50)
    print("OPENAI API SUBMISSION")
    print("="*50)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Define the prompt
    prompt = """Please analyze this Abstract and Reasoning Corpus (ARC) task.

The task shows 5 training examples where each input is a 3x3 grid and each output is a 9x9 grid. The colored cells (non-black) in the input should be transformed to the output following a specific rule.

Please:
1. Identify the transformation pattern from input to output
2. Explain the rule that maps 3x3 inputs to 9x9 outputs
3. Predict what the output should be for the test case
4. Provide a step-by-step explanation of how you arrived at your solution

Focus on understanding how the spatial relationships and patterns are preserved and transformed. This is a pattern recognition task that requires abstract reasoning."""
    
    print("📝 Submitting to OpenAI API...")
    print(f"Model: gpt-4-vision-preview")
    print(f"Max tokens: 1500")
    
    # Submit to API
    result = submit_to_openai_api(api_key, 'task.png', prompt)
    
    if result:
        print("\n✅ API Response received!")
        print("\n" + "="*50)
        print("OPENAI ANALYSIS")
        print("="*50)
        
        # Extract and display the response
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            print(content)
        else:
            print("Unexpected API response format:")
            print(result)
    else:
        print("❌ Failed to get response from OpenAI API")

if __name__ == "__main__":
    main()
