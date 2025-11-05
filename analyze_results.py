#!/usr/bin/env python3
"""
Result Analysis Script for Set-Consistency Verification
Processes JSON result files and formats them as Table 2a from the paper.

Author: Kyungjin Oh
Date: 2025
"""

import json
import os
from pathlib import Path

# Define the 8 experiment result files
RESULT_FILES = {
    'lconvqa': {
        'gpt-4o': {
            '5shot': 'results/vqa/lconvqa/llm/gpt-4o/5shot/arbitrary_pairs.json',
            '0shot': 'results/vqa/lconvqa/llm/gpt-4o/0shot/arbitrary_pairs.json',
        },
        'gpt-o3-mini': {
            '5shot': 'results/vqa/lconvqa/llm/gpt-o3-mini/5shot/arbitrary_pairs.json',
            '0shot': 'results/vqa/lconvqa/llm/gpt-o3-mini/0shot/arbitrary_pairs.json',
        }
    },
    'set_nli': {
        'gpt-4o': {
            '5shot': 'results/nli/set_nli/llm/gpt-4o/5shot/arbitrary_pairs.json',
            '0shot': 'results/nli/set_nli/llm/gpt-4o/0shot/arbitrary_pairs.json',
        },
        'gpt-o3-mini': {
            '5shot': 'results/nli/set_nli/llm/gpt-o3-mini/5shot/arbitrary_pairs.json',
            '0shot': 'results/nli/set_nli/llm/gpt-o3-mini/0shot/arbitrary_pairs.json',
        }
    }
}

def load_results(filepath):
    """Load JSON result file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_overall_accuracy(data):
    """Calculate overall accuracy from accuracy_by_sizes"""
    all_in_one = data['prediction']['all_in_one']
    accuracy_by_sizes = all_in_one['accuracy_by_sizes']
    
    # Simple average across all set sizes
    accuracies = [float(acc) for acc in accuracy_by_sizes.values()]
    return sum(accuracies) / len(accuracies) if accuracies else 0.0

def main():
    print("=" * 80)
    print("Set-Consistency Verification - Baseline Evaluation Results")
    print("=" * 80)
    print()
    
    # Store results in a structured format
    results = {}
    
    # Process each experiment
    for dataset in ['lconvqa', 'set_nli']:
        results[dataset] = {}
        for model in ['gpt-4o', 'gpt-o3-mini']:
            results[dataset][model] = {}
            for shot in ['5shot', '0shot']:
                filepath = RESULT_FILES[dataset][model][shot]
                full_path = os.path.join(os.getcwd(), filepath)
                
                print(f"Processing: {filepath}")
                
                if os.path.exists(full_path):
                    data = load_results(full_path)
                    accuracy = calculate_overall_accuracy(data)
                    results[dataset][model][shot] = accuracy * 100  # Convert to percentage
                    print(f"  → Overall Accuracy: {accuracy*100:.2f}%")
                else:
                    print(f"  → File not found!")
                    results[dataset][model][shot] = None
                print()
    
    # Format as Table 2a
    print("\n" + "=" * 80)
    print("Table 2a: Baseline Results (LLM Models)")
    print("=" * 80)
    print()
    
    print("| Model            | Shot | Set-LConVQA | Set-SNLI |")
    print("|------------------|------|-------------|----------|")
    
    for model in ['gpt-4o', 'gpt-o3-mini']:
        model_name = model.upper().replace('-', ' ')
        for shot in ['5shot', '0shot']:
            shot_label = shot.replace('shot', '-shot')
            lconvqa_acc = results['lconvqa'][model][shot]
            setnli_acc = results['set_nli'][model][shot]
            
            lconvqa_str = f"{lconvqa_acc:.2f}" if lconvqa_acc is not None else "N/A"
            setnli_str = f"{setnli_acc:.2f}" if setnli_acc is not None else "N/A"
            
            print(f"| {model_name:16} | {shot_label:4} | {lconvqa_str:11} | {setnli_str:8} |")
    
    print()
    print("Note: Values are overall accuracy (%) averaged across all set sizes.")
    print()
    
    # Save detailed results to file
    output_file = 'results/baseline_results_summary.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {output_file}")
    
    # Document all JSON files used
    print("\n" + "=" * 80)
    print("JSON Files Referenced:")
    print("=" * 80)
    for i, (dataset, models) in enumerate(RESULT_FILES.items(), 1):
        for model, shots in models.items():
            for shot, filepath in shots.items():
                full_path = os.path.join(os.getcwd(), filepath)
                exists = "✓" if os.path.exists(full_path) else "✗"
                print(f"{exists} {filepath}")
    print()

if __name__ == '__main__':
    main()
