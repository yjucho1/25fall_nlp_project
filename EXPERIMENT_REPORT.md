# Baseline Evaluation Experiment Report

**Date**: November 5, 2025  
**Branch**: replication-kyungjin  
**Researcher**: Kyungjin Oh

## Experiment Status

### API Quota Issue
The OpenAI API key ran out of credits during the experiments. Only **1 out of 8** planned experiments completed successfully with valid results.

### Completed Experiments ✅

| Dataset | Model | Shot | Status | Result File |
|---------|-------|------|--------|-------------|
| lconvqa | GPT-4o | 5-shot | ✅ **COMPLETED** | `results/vqa/lconvqa/llm/gpt-4o/5shot/arbitrary_pairs.json` (17KB) |

**Result**: Overall accuracy = **96.35%** across all set sizes

### Failed Experiments ❌ (API Quota Exceeded)

| # | Dataset | Model | Shot | Status | 
|---|---------|-------|------|--------|
| 2 | lconvqa | GPT-4o | 0-shot | ❌ API quota exceeded |
| 3 | lconvqa | o3-mini | 5-shot | ❌ API quota exceeded |
| 4 | lconvqa | o3-mini | 0-shot | ❌ API quota exceeded |
| 5 | set_nli | GPT-4o | 5-shot | ❌ API quota exceeded |
| 6 | set_nli | GPT-4o | 0-shot | ❌ API quota exceeded |
| 7 | set_nli | o3-mini | 5-shot | ❌ API quota exceeded |
| 8 | set_nli | o3-mini | 0-shot | ❌ API quota exceeded |

## Partial Results Table (Table 2a Format)

| Model | Shot | Set-LConVQA | Set-SNLI |
|-------|------|-------------|----------|
| GPT-4o | 5-shot | **96.35** | N/A (quota) |
| GPT-4o | 0-shot | N/A (quota) | N/A (quota) |
| o3-mini | 5-shot | N/A (quota) | N/A (quota) |
| o3-mini | 0-shot | N/A (quota) | N/A (quota) |

## Next Steps

1. **Refill API Credits**: Add more credits to OpenAI API account
2. **Resume Experiments**: Re-run the remaining 7 experiments:
   ```bash
   # lconvqa experiments
   python evaluate_baseline.py --task vqa --dataset lconvqa --type llm --model gpt-4o --shot_num 0
   python evaluate_baseline.py --task vqa --dataset lconvqa --type llm --model gpt-o3-mini --shot_num 5
   python evaluate_baseline.py --task vqa --dataset lconvqa --type llm --model gpt-o3-mini --shot_num 0
   
   # set_nli experiments  
   python evaluate_baseline.py --task nli --dataset set_nli --type llm --model gpt-4o --shot_num 5
   python evaluate_baseline.py --task nli --dataset set_nli --type llm --model gpt-4o --shot_num 0
   python evaluate_baseline.py --task nli --dataset set_nli --type llm --model gpt-o3-mini --shot_num 5
   python evaluate_baseline.py --task nli --dataset set_nli --type llm --model gpt-o3-mini --shot_num 0
   ```
3. **Complete Analysis**: Run `python analyze_results.py` after all experiments finish
4. **Integration**: Merge teammate's Qwen results when available

## Log Files

- `results/exp1_lconvqa_gpt4o_5shot.log` - Completed successfully
- `results/exp2_lconvqa_gpt4o_0shot.log` - API quota error
- `results/exp3_lconvqa_o3mini_5shot.log` - Completed but quota error at 0-shot
- `results/exp4_lconvqa_o3mini_0shot.log` - API quota error
- `results/exp5_setnli_gpt4o_5shot.log` - API quota error
- `results/exp6_setnli_gpt4o_0shot.log` - API quota error
- `results/exp7_setnli_o3mini_5shot.log` - API quota error
- `results/exp8_setnli_o3mini_0shot.log` - API quota error

## Files Created

- `analyze_results.py` - Result analysis and Table 2a formatting script
- `EXPERIMENT_REPORT.md` - This report
- `results/baseline_results_summary.json` - JSON summary of results

## Notes

- All experimental setup and environment configuration were successful
- The code runs correctly when API credits are available
- Only 1 experiment completed due to API quota limitations
- Dataset folder structure was fixed (set_lconvqa → lconvqa)
- Conda environment created successfully without CUDA (Mac compatibility)
- Branch `replication-kyungjin` is ready for the remaining experiments

