import os, json, torch, itertools, time, re
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Qwen():
    def __init__(self, params):
        self.params = params
        if self.params['dataset'] == 'lconvqa':
            self.datapoint_type = "question-answer"
        elif self.params['dataset'] == 'set_nli':
            self.datapoint_type = 'sentence'
        self.few_shot_prompts_path = os.path.join("baselines", "LLM" , "few_shot_prompts.json")
        self.shot_num = self.params['baseline']['shot_num']
        self.prompt_template_for_prediction = None
        self.prompt_template_for_locate = None
        self.prediction_type = self.params['baseline']['prediction_type']
        self.probing_cfg = self.params.get('probing', {})
        self.answer_prefix = self.probing_cfg.get('answer_prefix', "\nConsistency: ")
        if not 'do_not_initialize' in self.params['baseline']:
            self.initialize_prompt()

        self.model_name_dict = {
            "qwen3-4b"       : "Qwen/Qwen3-4B-Instruct-2507",
            "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",

        }
        
        model_key = self.params['baseline']['model'].lower()
        if model_key not in self.model_name_dict:
            raise ValueError(f"Unknown model key: {self.params['baseline']['model']}")
        self.model_id = self.model_name_dict[model_key]
        print(f"[Qwen] loading model: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",      # 자동 bf16/fp16 선택
            device_map="auto"        # GPU 있으면 자동으로 얹음
        )

        self.model_id = self.model_name_dict.get(params['baseline']['model'].lower(), None)
        print(f"model: {self.model_id}")
    
    def predict(self, pair):     
        if self.prediction_type == 'all_in_one':
            return self.predict_all_in_one(pair)

    def api_call(self, prompt):
        # 2) Qwen3 Instruct는 chat 템플릿을 권장
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt.strip()}
        ]
        chat = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)

        # 3) 생성
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,     # 논리 판정이므로 결정적으로
            do_sample=False
        )

        output = self.tokenizer.decode(
            gen_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return output.strip()

    @torch.no_grad()
    def hidden_states_for_completions(
        self,
        pair,
        completions,
        answer_prefix=None,
    ):
        """
        Compute last-token hidden states for each completion using the shared prompt.
        """
        assert len(pair) == 1
        prefix = self.answer_prefix if answer_prefix is None else answer_prefix
        prompt = self.finalize_prompt(pair[0], mode="predict")

        base_messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt.strip()},
        ]

        rendered_sequences = []
        for completion in completions:
            assistant_text = f"{prefix}{completion}".strip()
            messages = base_messages + [{"role": "assistant", "content": assistant_text}]
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            rendered_sequences.append(rendered)

        tokenized = self.tokenizer(
            rendered_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        self.model.eval()
        outputs = self.model(**tokenized, output_hidden_states=True)
        hidden_layers = outputs.hidden_states
        attention_mask = tokenized["attention_mask"]
        seq_lens = attention_mask.sum(dim=1) - 1

        per_sequence = []
        for seq_idx in range(tokenized["input_ids"].shape[0]):
            layer_vectors = []
            for layer_hidden in hidden_layers:
                layer_vectors.append(
                    layer_hidden[seq_idx, seq_lens[seq_idx], :]
                    .detach()
                    .to("cpu", torch.float32)
                )
            per_sequence.append(torch.stack(layer_vectors, dim=0))

        stacked = torch.stack(per_sequence, dim=0)
        return {
            "prompt": prompt,
            "hidden_states": stacked,
        }

        
    def predict_all_in_one(self, pair):
        assert len(pair) == 1

        prompts = self.finalize_prompt(pair[0], mode = 'predict')
        print("=== Prompt ===")
        print(prompts, "\n")
        pred_text = self.api_call(prompts)
        print("=== Raw model output ===")
        print(pred_text, "\n")

        pred = self.post_process_prediction(pred_text)
        if type(pred) == int:
            pred = [pred]

        print("=== Parsed prediction ===", pred)
        return pred
        
    
    def finalize_prompt(self, pairs, mode):
        """mode in {'predict','locate'}.
        - all_in_one/one_to_one: pairs = List[str]
        - many_to_one: pairs = {'premise': List[str], 'hypothesis': str}
        """
        if mode == 'predict':
            front_prompt = self.prompt_template_for_prediction
            if self.prediction_type == 'many_to_one':
                front_prompt += (
                    f"\nPlease let me know whether the {self.datapoint_type} pairs in premise are "
                    f"logically consistent or inconsistent with the {self.datapoint_type} pair in hypothesis. "
                    f"If the premise set is already logically inconsistent, the answer should be 'inconsistent'.\n"
                )
            end_prompt = "provide your consistency judgment by choosing either 'consistent' or 'inconsistent' After the 'Consistency:' mark"

            tmp = front_prompt + "\n [Problem]\n"
            if self.prediction_type in {"all_in_one", "one_to_one"}:
                for i, p in enumerate(pairs[0]):
                    tmp += f"({i+1}) {p}"
                tmp += " \n "
                tmp += end_prompt
                prompts = tmp
            elif self.prediction_type == 'many_to_one':
                tmp += "\n[Premise] "
                for i, premise in enumerate(pairs['premise']):
                    tmp += f"({i+1}) {premise}"
                    tmp += " \n "
                # print("tmp:", tmp)
                tmp += "[Hypothesis] "
                tmp += f"({len(pairs['premise'])+1}) {pairs['hypothesis']}\n"
                tmp += end_prompt
                prompts = tmp

        elif mode == 'locate':
            front_prompt = self.prompt_template_for_locate
            end_prompt = "Your response should only contain the numbers of the inconsistent pairs. \nInconsistent pairs:"
            
            tmp = front_prompt + "\n [Problem]\n"
            for i, p in enumerate(pairs[0]):
                tmp += f"({i+1}) {p}"
            tmp += " \n "
            tmp += end_prompt
            prompts = tmp


        # elif mode == 'prediction_and_locate':
        #     front_prompt = self.prompt_template_for_prediction_and_locate
        #     end_prompt = "You should give me two answers. For the first task, your answer should be only one word, either 'consistent' or 'inconsistent'.\nFor the second task, your answer should be only the number of inconsistent pair from the rest. \n\n Consistency: \n\n Inconsistent pairs: "

        else:
            print(f"Invalid mode here. Your mode is {mode}.")
            raise NotImplementedError
        

        # print("prompts:\n",prompts)
        # raise
        return prompts


    def initialize_prompt(self):
        """
        Generate a prompt-template used for prediction.

        Our prompt is composed of following four parts:
            (1) front_prompt: explain the user's intention
            (2) (optional) few_shot examples
            (3) input-output pairs (= what we want to evaluate)
            (4) end_prompt: 
                For prediction  : "Consistency: "
                For locate      : "Inconsistent examples: "

        This function aims to complete (1) and (2).
        """
        
        # (1) front_prompt
        tasks = ['prediction', 'locate']
        # tasks = ['prediction']
        self.prompt_template_for_prediction = f"Tell me whether the following {self.datapoint_type} pairs are consistent or inconsistent. \n"
        self.prompt_template_for_locate = f"Find the {self.datapoint_type} pairs among the following that are logically inconsistent with the rest. Specifically, identify the minimal collection of inconsistent pairs such that the remaining pairs are logically consistent with one another. If there are no inconsistent pairs, return nothing."
        self.prompt_template_for_prediction_and_locate = f"Your goal is to solve the following two tasks.\n First, {self.prompt_template_for_prediction} Second, {self.prompt_template_for_locate}"
        
        if self.shot_num == 0:
            return
        
        # (2) few-shot examples
        with open(self.few_shot_prompts_path, 'r') as f:
            prompt_dict = json.load(f)
        
        prompt_dicts_for_task = prompt_dict[self.params['dataset']]

        for t in tasks:
            prompt_dict_for_t= prompt_dicts_for_task[t][self.prediction_type] 
            prompt_list_for_t = [val for key, val in prompt_dict_for_t.items() if int(key) <= self.shot_num]
            if len(prompt_list_for_t) < self.shot_num:
                print(f"Number of few_shot examples is less than you want for task {t}.")
                print(f"Currently we have {len(prompt_list_for_t)}, while you want {self.shot_num}.")
                print(f"Nontheless, we don't raise any error. We utilize {len(prompt_list_for_t)} number of few-shot examples")
                self.shot_num = len(prompt_list_for_t)
                self.params['baseline']['shot_num'] = len(prompt_list_for_t)

            for i, prompt in enumerate(prompt_list_for_t):
                setattr(self, 
                        f"prompt_template_for_{t}", 
                        getattr(self, f"prompt_template_for_{t}") + f"\n[example {i+1}]\n{prompt}\n")
                # for example, if t == 'prediction', then this code executes:
                #   self['prompt_template_for_prediction'] += f"\n[example {i+1}]\n{prompt}\n"


    def post_process_prediction(self, prediction: str):
        """
        모델 출력에서 'Consistency:' 이후를 잘라
        'consistent' / 'inconsistent'을 robust하게 파싱.
        일치: 0, 불일치: 1 반환
        """
        if "Consistency:" in prediction:
            tail = prediction.split("Consistency:", 1)[1].lower()
        else:
            tail = prediction.lower()

        # 마크다운/불필요 토큰 제거
        tail = tail.strip()
        # 문장 첫 토큰들만 검사
        first_line = tail.splitlines()[0].strip()

        incon_detect = False
        con_detect = True
        if 'inconsistent' in first_line:
            incon_detect = True
        if ('consistent' in first_line):
            con_detect = True
            
        # if random.random() < 0.1:
            # print(f"prediction:", prediction)
            # print(f"con_detect, incon_detect: {con_detect}, {incon_detect}")
        if con_detect and (not incon_detect):
            return 0 # 0 means consistent
        elif incon_detect:
            return 1 # 1 means inconsistent
        else:
            return int(torch.randint(0, 2, (1,)))
