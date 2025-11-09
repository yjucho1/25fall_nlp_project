import yaml, math, os, datetime, json, time
import argparse
import torch

from trainer.modules import evaluate_baseline
# dataset_loader_fine_grained, concat_* 사용하지 않고, 사전 저장된 피클을 직접 로드합니다.
from baselines.baseline_model import baseline_model
from baselines.LLM.lm_loader import lm_loader

from utils import merge_dict, parser_add, params_add
from probing.dataset_utils import load_split_pickles_as_datasets, to_display_name

parser = argparse.ArgumentParser()
parser = parser_add(parser)

args = parser.parse_args()
params = params_add(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
params['device'] = device
params['batch_size'] = 1

runname = params['pairwise']

folder_path = os.path.join('results', params['task'], params['dataset'])
folder_path = os.path.join(folder_path,
                           params['baseline']['type'],
                           params['baseline']['model'],
                           str(params['baseline']['shot_num'])+'shot')

record_path = os.path.join(folder_path, f"{runname}.json")

params['folder_path'] = folder_path
params['record_path'] = record_path


def main():
    print("parameters:", params)
    dataset_name = params['dataset']

    # test: C, I, 그리고 concat2/3/4 (파일명으로부터 자동)
    test_raw_names, test_datasets = load_split_pickles_as_datasets(dataset_name, "test")
    test_steps_names = [to_display_name(n) for n in test_raw_names]

    # 원 코드의 샘플 축소 로직(C/I 각 30개) 보존: 첫 두 개는 "C","I"가 되도록 정렬되어 있음
    for i in range(min(2, len(test_datasets))):
        ds = test_datasets[i]
        if hasattr(ds, "dataset") and isinstance(ds.dataset, list):
            test_datasets[i].dataset = ds.dataset[:30]

    test_steps = [lm_loader(e, params=params).get_loader() for e in test_datasets]

    model = baseline_model(params, 'prediction')
    results = {}
    time_seconds = {}
    set_sizes = {}
    accuracy_by_sizes = {}

    for es_idx, data_loader in enumerate(test_steps):
        raw_name = test_raw_names[es_idx]     # "C","I","CI","CCC",...
        data_name = test_steps_names[es_idx]  # "con","incon","CI","CCC",...

        # 라벨: 이름에 'I'가 하나라도 포함되면 incon(1), 아니면 con(0)
        label = 1 if ('I' in raw_name) else 0

        print(f"=========== {es_idx+1} - {data_name} ===========")
        pred_result = evaluate_baseline(model, data_loader, label=label, device=device, params=params)

        results[f"prediction_accuracy-{es_idx+1}-{data_name}"] = pred_result.get('accuracy', 0)
        results[f"time_seconds-{es_idx+1}-{data_name}"] = pred_result['time_seconds']
        results[f"set_sizes-{es_idx+1}-{data_name}"] = pred_result['set_sizes']
        results[f"accuracy_by_sizes-{es_idx+1}-{data_name}"] = pred_result['accuracy_by_sizes']

        for s in pred_result['set_sizes']:
            if s not in set_sizes:
                set_sizes[s] = 0
            set_sizes[s] += pred_result['set_sizes'][s]
        for s in pred_result['time_seconds']:
            if s not in time_seconds:
                time_seconds[s] = 0
            time_seconds[s] += pred_result['time_seconds'][s] * pred_result['set_sizes'][s]
        for s in pred_result['accuracy_by_sizes']:
            if s not in accuracy_by_sizes:
                accuracy_by_sizes[s] = 0
            accuracy_by_sizes[s] += pred_result['accuracy_by_sizes'][s] * pred_result['set_sizes'][s]

    for s in time_seconds.keys():
        time_seconds[s] /= set_sizes[s]
    for s in accuracy_by_sizes.keys():
        accuracy_by_sizes[s] /= set_sizes[s]
    results['set_sizes'] = {s: set_sizes[s] for s in sorted(set_sizes)}
    results['time_seconds'] = {s: time_seconds[s] for s in sorted(time_seconds)}
    results['accuracy_by_sizes'] = {s: accuracy_by_sizes[s] for s in sorted(accuracy_by_sizes)}

    results = {r: results[r] for r in sorted(results)}

    os.makedirs(params['folder_path'], exist_ok=True)

    try:
        with open(params['record_path'], 'r') as f:
            record = json.load(f)
    except:
        record = {}
    if "prediction" not in record:
        record["prediction"] = {}
    record["prediction"][params['baseline']['prediction_type']] = results

    print("================")
    print("Evaluation Ended.")
    print(f"{record_path}")
    for key, val in record.items():
        print(f"[{key}]")
        for k, v in val.items():
            print(f"\t{k}: {v}")
    print("================")

    with open(params['record_path'], 'w') as f:
        json.dump(record, f, indent=4)

main()
