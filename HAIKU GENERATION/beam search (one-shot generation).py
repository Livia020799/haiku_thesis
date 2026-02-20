#prova 1 - around 6 minutes

import os
import re
import argparse
import math
import torch
import pandas as pd
import pyopenjtalk
from transformers import AutoTokenizer, AutoModelForCausalLM

# Monkey‐patch & disable Dynamo to avoid Python 3.12 compile errors
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def is_ascii_free(s: str) -> bool:
    # True if there are NO ASCII letters in the entire string
    return not re.search(r"[A-Za-z]", s)

def clean_haiku(raw: str) -> str:
    """
    Extract up to the first 3 “real” lines:
      - ignore empty lines
      - stop completely if you see:
         * a Markdown heading (#…)
         * bold markers (***… or **…)
         * our <end_of_turn> token
      - skip any line containing ASCII letters
    """
    haiku_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        # stop on markdown or end token
        if re.match(r'^(#|\*)', s) or s.startswith("<end_of_turn>"):
            break
        # skip lines with English letters
        if re.search(r"[A-Za-z]", s):
            continue
        haiku_lines.append(s)
        if len(haiku_lines) == 3:
            break
    # pad to exactly 3 lines
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def parse_args():
    p = argparse.ArgumentParser(description="Few‐shot 俳句生成の評価スクリプト")
    p.add_argument("--model_dir",     required=True, help="モデルのディレクトリ")
    p.add_argument("--tokenizer_dir", required=True, help="トークナイザのディレクトリ")
    p.add_argument("--train_jsonl",   required=True, help="学習用JSONLファイル")
    p.add_argument("--kigo_jsonl",    required=True, help="季語リストJSONLファイル")
    p.add_argument("--output_dir",    default="./fewshot_results", help="結果出力先ディレクトリ")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load & merge
    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl,  lines=True)
    df   = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
    df.rename(columns={
        "season_haiku":       "season",
        "haiku":               "ref_haiku",
        "5_mora_segment_1":    "m5_1",
        "7_mora_segment":      "m7",
        "5_mora_segment_2":    "m5_2",
    }, inplace=True)

    # Model & Tokenizer
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, use_fast=False, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model.eval()

    # Few‐shot prompt template
    few_shot_k = 4
    template = (
        "季語: {word}\n"
        "季節: {season}\n"
        "構造: {haiku_structure}\n"
        "五音節１: {m5_1}\n"
        "七音節: {m7}\n"
        "五音節２: {m5_2}\n"
        "俳句: {ref_haiku}\n"
    )

    def retrieve_examples(target):
        pool = df[df.season == target.season]
        same = pool[pool.kigo_id == target.kigo_id]
        if len(same) >= few_shot_k:
            pool = same
        return pool.sample(min(few_shot_k, len(pool)))

    def build_prompt(target, examples):
        blocks = [template.format(**ex) for _, ex in examples.iterrows()]
        blocks.append(
            f"次の条件で俳句を書いてください:\n"
            f"季語: {target.word}\n"
            f"季節: {target.season}\n"
            f"構造: {target.haiku_structure}\n"
            f"五音節１: {target.m5_1}\n"
            f"七音節: {target.m7}\n"
            f"五音節２: {target.m5_2}\n"
            "俳句:"
        )
        return "\n".join(blocks)

    stats = []
    for _, tgt in df.sample(10, random_state=42).iterrows():
        exs    = retrieve_examples(tgt)
        prompt = build_prompt(tgt, exs)

        outputs, ppls = [], []
        for _ in range(100):
            inp = tok(prompt, return_tensors="pt").to(device)
            input_len = inp.input_ids.shape[-1]
            with torch.cuda.amp.autocast(), torch.no_grad():
                gen = model.generate(
                    **inp,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7
                )
            gen_ids = gen[0][input_len:]
            raw_txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
            haiku   = clean_haiku(raw_txt)

            outputs.append(haiku)
            ppls.append(compute_perplexity(model, tok, haiku, device))

        # Metrics
        mora_rate = sum(is_575(h) for h in outputs) / len(outputs)
        kigo_rate = sum(tgt.word in h for h in outputs) / len(outputs)
        avg_ppl   = sum(ppls) / len(ppls)

        # Representative selection (no 5-7-5 filtering)
        # 1) non-empty
        valid = [(i, h, p) for i, (h, p) in enumerate(zip(outputs, ppls)) if h.strip()]
        if not valid:
            valid = list(enumerate(zip(outputs, ppls)))
        # 2) prefer ascii-free
        ascii_free = [(i, h, p) for (i, h, p) in valid if is_ascii_free(h)]
        candidates = ascii_free if ascii_free else valid
        # 3) pick lowest perplexity
        best_idx = min(candidates, key=lambda tup: tup[2])[0]
        repr_haiku = outputs[best_idx]

        stats.append({
            "kigo":            tgt.word,
            "season":          tgt.season,
            "haiku_structure": tgt.haiku_structure,
            "m5_1":            tgt.m5_1,
            "m7":              tgt.m7,
            "m5_2":            tgt.m5_2,
            "ref_haiku":       tgt.ref_haiku,
            "repr_haiku":      repr_haiku,
            "mora_rate":       mora_rate,
            "kigo_rate":       kigo_rate,
            "ppl":             avg_ppl
        })

    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(
        os.path.join(args.output_dir, "fewshot_eval.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Done → few‑shot eval saved to {args.output_dir}/fewshot_eval.csv")

if __name__ == "__main__":
    main()

#prova 2 - beam search
'''
fewshot_eval_1
around 6 min with:
    # Settings
    few_shot_k  = 4
    num_targets = 10
    num_samples = 100
'''

'''
fewshot_eval_2
around 1h with:
    # Settings
    few_shot_k  = 8
    num_targets = 30
    num_samples = 300
'''

import os
import re
import string
import argparse
import math
import torch
import pandas as pd
import pyopenjtalk
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# Monkey‑patch & disable Dynamo to avoid Python 3.12 compile errors
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def is_ascii_free(s: str) -> bool:
    return not re.search(r"[A-Za-z]", s)

def clean_haiku(raw: str) -> str:
    """
    Build up to the first 3 “real” lines:
      - ignore empty, markdown (#…), hint lines (*… or containing 'ヒント'), <end_of_turn>
      - strip ASCII letters, digits, punctuation
      - pad to 3 lines
    """
    haiku_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("*") or s.startswith("<end_of_turn>") or "ヒント" in s:
            continue
        cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", s).strip()
        if not cleaned:
            continue
        haiku_lines.append(cleaned)
        if len(haiku_lines) == 3:
            break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",     required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--train_jsonl",   required=True)
    p.add_argument("--kigo_jsonl",    required=True)
    p.add_argument("--output_dir",    default="./fewshot_results")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # merge train and kigo datasets
    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl,  lines=True)
    df   = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
    df.rename(columns={
        "season_haiku":    "season",
        "haiku":           "ref_haiku",
        "5_mora_segment_1":"m5_1",
        "7_mora_segment":  "m7",
        "5_mora_segment_2":"m5_2",
    }, inplace=True)

    # model & tokenizer
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, use_fast=False, local_files_only=True
    )
    # assicuriamoci che <end_of_turn> sia special token
    if "<end_of_turn>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
    end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tok))
    model.eval()

    # stopping criteria
    stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

    # Settings
    few_shot_k  = 4
    num_targets = 10
    num_samples = 100

    # Prompt template
    template = (
        "季語: {word}\n"
        "季節: {season}\n"
        "構造: {haiku_structure}\n"
        "五音節１: {m5_1}\n"
        "七音節: {m7}\n"
        "五音節２: {m5_2}\n"
        "俳句: {ref_haiku}\n"
    )

    def retrieve_examples(target):
        pool = df[df.season == target.season]
        same = pool[pool.kigo_id == target.kigo_id]
        if len(same) >= few_shot_k:
            pool = same
        return pool.sample(min(few_shot_k, len(pool)), random_state=target.name)

    def build_prompt(target, examples):
        blocks = [template.format(**ex) for _, ex in examples.iterrows()]
        blocks.append(
            f"次の条件で俳句を書いてください:\n"
            f"季語: {target.word}\n"
            f"季節: {target.season}\n"
            f"構造: {target.haiku_structure}\n"
            f"五音節１: {target.m5_1}\n"
            f"七音節: {target.m7}\n"
            f"五音節２: {target.m5_2}\n"
            "俳句:"
        )
        return "\n".join(blocks)

    stats = []
    for _, tgt in df.sample(num_targets, random_state=42).iterrows():
        exs    = retrieve_examples(tgt)
        prompt = build_prompt(tgt, exs)

        # 1) sampling per stats
        outputs, ppls = [], []
        for _ in range(num_samples):
            inp = tok(prompt, return_tensors="pt").to(device)
            with torch.cuda.amp.autocast(), torch.no_grad():
                gen = model.generate(
                    **inp,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    stopping_criteria=stop_criteria
                )
            gen_ids = gen[0][inp.input_ids.shape[-1]:]
            raw_txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
            haiku   = clean_haiku(raw_txt)
            outputs.append(haiku)
            if not haiku:
                ppls.append(float('inf'))
            else:
                ppls.append(compute_perplexity(model, tok, haiku, device))

        finite_ppls = [p for p in ppls if math.isfinite(p)]
        mora_rate   = sum(is_575(h) for h in outputs) / len(outputs)
        kigo_rate   = sum(1 for h in outputs if tgt.word in (h or "")) / len(outputs)
        avg_ppl     = sum(finite_ppls) / len(finite_ppls) if finite_ppls else float('inf')

        # 2) beam search representative
        inp = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            beam = model.generate(
                **inp,
                max_new_tokens=50,
                num_beams=5,
                early_stopping=True,
                stopping_criteria=stop_criteria
            )
        beam_txt    = tok.decode(beam[0][inp.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        repr_haiku  = clean_haiku(beam_txt)

        stats.append({
            "kigo":            tgt.word,
            "season":          tgt.season,
            "haiku_structure": tgt.haiku_structure,
            "m5_1":            tgt.m5_1,
            "m7":              tgt.m7,
            "m5_2":            tgt.m5_2,
            "ref_haiku":       tgt.ref_haiku,
            "repr_haiku":      repr_haiku,
            "mora_rate":       mora_rate,
            "kigo_rate":       kigo_rate,
            "avg_ppl":         avg_ppl
        })

    pd.DataFrame(stats).to_csv(
        os.path.join(args.output_dir, "fewshot_eval.csv"),
        index=False, encoding="utf-8-sig"
    )
    print(f"Done → few‑shot eval salvato in {args.output_dir}/fewshot_eval.csv")

if __name__=="__main__":
    main()


#prova 3 - beam search + optimized prompt (with help from ChatGPT) + making sure the model knows to stop at <end of turn>

import os
import re
import string
import argparse
import math
import torch
import pandas as pd
import pyopenjtalk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Monkey‑patch & disable Dynamo per Python 3.12
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("*") or s.startswith("<end_of_turn>") or "ヒント" in s:
            continue
        cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", s).strip()
        if not cleaned:
            continue
        haiku_lines.append(cleaned)
        if len(haiku_lines) == 3:
            break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir")
    p.add_argument("--tokenizer_dir")
    p.add_argument("--train_jsonl")
    p.add_argument("--kigo_jsonl")
    p.add_argument("--output_dir")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl, lines=True)
    df = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku", "_kigo"))
    df.rename(columns={
        "season_haiku": "season",
        "haiku": "ref_haiku",
        "5_mora_segment_1": "m5_1",
        "7_mora_segment": "m7",
        "5_mora_segment_2": "m5_2",
    }, inplace=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    if "<end_of_turn>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
    end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tok))
    model.eval()

    stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

    few_shot_k = 4
    num_targets = 10
    num_beams = 100
    num_return_sequences = 100
    num_beam_groups = 10
    diversity_penalty = 1.0

    template = (
        "以下は俳句の例です。\n"
        "季語: {word}\n"
        "季節: {season}\n"
        "構造: {haiku_structure}\n"
        "五音節１: {m5_1}\n"
        "七音節: {m7}\n"
        "五音節２: {m5_2}\n"
        "完成俳句:\n{ref_haiku}\n"
    )

    def retrieve_examples(target):
        pool = df[df.season == target.season]
        same = pool[pool.kigo_id == target.kigo_id]
        if len(same) >= few_shot_k:
            pool = same
        return pool.sample(min(few_shot_k, len(pool)), random_state=target.name)

    def build_prompt(target, examples):
        blocks = [template.format(**ex) for _, ex in examples.iterrows()]
        blocks.append(
            "次の条件に基づいて俳句を一つ作成してください。\n"
            f"季語: {target.word}\n"
            f"季節: {target.season}\n"
            f"構造: {target.haiku_structure}\n"
            f"五音節１: {target.m5_1}\n"
            f"七音節: {target.m7}\n"
            f"五音節２: {target.m5_2}\n"
            "完成俳句:\n"
            "<end_of_turn>\n"
        )
        return "\n".join(blocks)

    stats = []
    for _, tgt in df.sample(num_targets, random_state=42).iterrows():
        exs = retrieve_examples(tgt)
        prompt = build_prompt(tgt, exs)

        inp = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inp,
                max_new_tokens=50,
                num_beams=100,
                num_return_sequences=100,
                num_beam_groups=10,
                diversity_penalty=1.0,
                early_stopping=True,
                stopping_criteria=stop_criteria
            )


        outputs, ppls = [], []
        for g in gen:
            raw = tok.decode(g[inp.input_ids.shape[-1]:], skip_special_tokens=False).strip()
            h = clean_haiku(raw)
            outputs.append(h)
            ppls.append(compute_perplexity(model, tok, h, device) if h else float("inf"))

        valid_haiku = [(h, p) for h, p in zip(outputs, ppls) if h and math.isfinite(p)]
        if valid_haiku:
            repr_haiku, best_ppl = min(valid_haiku, key=lambda x: x[1])
        else:
            repr_haiku, best_ppl = "", float("inf")

        mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
        kigo_rate = sum(1 for h in outputs if tgt.word in (h or "")) / len(outputs)
        avg_ppl = sum(p for p in ppls if math.isfinite(p)) / len(ppls)

        stats.append({
            "kigo": tgt.word,
            "season": tgt.season,
            "haiku_structure": tgt.haiku_structure,
            "m5_1": tgt.m5_1,
            "m7": tgt.m7,
            "m5_2": tgt.m5_2,
            "ref_haiku": tgt.ref_haiku,
            "repr_haiku": repr_haiku,
            "mora_rate": mora_rate,
            "kigo_rate": kigo_rate,
            "avg_ppl": avg_ppl,
        })

    pd.DataFrame(stats).to_csv(
        os.path.join(args.output_dir, "fewshot_eval.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    print(f"Done → few‑shot eval salvato in {args.output_dir}/fewshot_eval.csv")

if __name__ == "__main__":
    main()

#prova 4
#beam search + optimized prompt (with help from ChatGPT) + making sure the model knows to stop at <end of turn> + nucleus sampling
# + retrieve similar samples with semantic k-NN

#!/usr/bin/env python

import os
import re
import string
import argparse
import math
import torch
import pandas as pd
import pyopenjtalk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from sentence_transformers import SentenceTransformer, util

# Monkey‑patch & disable Dynamo per Python 3.12
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    """
    Build up to the first 3 “real” lines:
      - ignore empty, markdown (#…), hint lines (*… or containing 'ヒント'), <end_of_turn>
      - strip ASCII letters, digits, punctuation
      - pad to 3 lines
    """
    haiku_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("*") or s.startswith("<end_of_turn>") or "ヒント" in s:
            continue
        cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", s).strip()
        if not cleaned:
            continue
        haiku_lines.append(cleaned)
        if len(haiku_lines) == 3:
            break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def parse_args():
    p = argparse.ArgumentParser(description="Few‑shot haiku eval con retrieval semantico e nucleus sampling")
    p.add_argument("--model_dir",     required=True, help="Directory del modello")
    p.add_argument("--tokenizer_dir", required=True, help="Directory del tokenizer")
    p.add_argument("--train_jsonl",   required=True, help="File JSONL di training")
    p.add_argument("--kigo_jsonl",    required=True, help="File JSONL dei kigo")
    p.add_argument("--output_dir",    default="./fewshot_results", help="Cartella output")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carica e unisci dataset
    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl,  lines=True)
    df   = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
    df.rename(columns={
        "season_haiku":    "season",
        "haiku":           "ref_haiku",
        "5_mora_segment_1":"m5_1",
        "7_mora_segment":  "m7",
        "5_mora_segment_2":"m5_2",
    }, inplace=True)

    # 1) inizializza l’embedder e calcola gli embedding di riferimento
    embedder = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    df["_emb"] = embedder.encode(df["ref_haiku"].tolist(), convert_to_tensor=True)

    # Tokenizer & modello
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    if "<end_of_turn>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
    end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tok))
    model.eval()

    stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

    few_shot_k  = 8
    num_targets = 30
    num_samples = 300

    template = (
        "以下は俳句の例です。\n"
        "季語: {word}\n"
        "季節: {season}\n"
        "構造: {haiku_structure}\n"
        "五音節１: {m5_1}\n"
        "七音節: {m7}\n"
        "五音節２: {m5_2}\n"
        "完成俳句:\n{ref_haiku}\n"
    )

    # 2) retrieval semantico via k‑NN su embedding
    def retrieve_examples(target):
        tgt_emb = embedder.encode(target.ref_haiku, convert_to_tensor=True)
        sims = util.cos_sim(tgt_emb, torch.stack(df["_emb"].tolist()))[0]
        topk = sims.topk(few_shot_k + 10)
        inds = topk.indices.cpu().tolist()
        exs  = df.iloc[inds]
        if len(exs) < few_shot_k:
            exs = df[df.season == target.season].sample(few_shot_k, random_state=target.name)
        return exs.head(few_shot_k)

    def build_prompt(target, examples):
        blocks = [template.format(**ex) for _, ex in examples.iterrows()]
        blocks.append(
            "次の条件に基づいて俳句を一つ作成してください。\n"
            f"季語: {target.word}\n"
            f"季節: {target.season}\n"
            f"構造: {target.haiku_structure}\n"
            f"五音節１: {target.m5_1}\n"
            f"七音節: {target.m7}\n"
            f"五音節２: {target.m5_2}\n"
            "完成俳句:\n"
            "<end_of_turn>\n"
        )
        return "\n".join(blocks)

    stats = []
    for _, tgt in df.sample(num_targets, random_state=42).iterrows():
        exs    = retrieve_examples(tgt)
        prompt = build_prompt(tgt, exs)

        # 3) sampling con nucleus sampling per metriche
        outputs, ppls = [], []
        for _ in range(num_samples):
            inp = tok(prompt, return_tensors="pt").to(device)
            with torch.cuda.amp.autocast(), torch.no_grad():
                gen = model.generate(
                    **inp,
                    max_new_tokens=50,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    stopping_criteria=stop_criteria
                )
            raw = tok.decode(gen[0][inp.input_ids.shape[-1]:], skip_special_tokens=False).strip()
            h   = clean_haiku(raw)
            outputs.append(h)
            ppls.append(compute_perplexity(model, tok, h, device) if h else float("inf"))

        finite_ppls = [p for p in ppls if math.isfinite(p)]
        mora_rate   = sum(is_575(h) for h in outputs) / len(outputs)
        kigo_rate   = sum(1 for h in outputs if tgt.word in (h or "")) / len(outputs)
        avg_ppl     = sum(finite_ppls) / len(finite_ppls) if finite_ppls else float("inf")

        # beam search per haiku rappresentativo
        inp = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            beam = model.generate(
                **inp,
                max_new_tokens=50,
                num_beams=5,
                early_stopping=True,
                stopping_criteria=stop_criteria
            )
        beam_txt   = tok.decode(beam[0][inp.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        repr_haiku = clean_haiku(beam_txt) or next((h for h in outputs if h), "")

        stats.append({
            "kigo":            tgt.word,
            "season":          tgt.season,
            "haiku_structure": tgt.haiku_structure,
            "m5_1":            tgt.m5_1,
            "m7":              tgt.m7,
            "m5_2":            tgt.m5_2,
            "ref_haiku":       tgt.ref_haiku,
            "repr_haiku":      repr_haiku,
            "mora_rate":       mora_rate,
            "kigo_rate":       kigo_rate,
            "avg_ppl":         avg_ppl,
        })

    pd.DataFrame(stats).to_csv(
        os.path.join(args.output_dir, "fewshot_eval.csv"),
        index=False, encoding="utf-8-sig"
    )
    print(f"Done → few‑shot eval salvato in {args.output_dir}/fewshot_eval.csv")

if __name__ == "__main__":
    main()



# prova 5 (DA PROVARE)
#CORRECTED BEAM SEARCH for HAIKU GENERATION — with no overlap between target and few-shot

import os
import re
import string
import argparse
import math
import time
import torch
import pandas as pd
import pyopenjtalk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Monkey‑patch & disable Dynamo per Python 3.12
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def _init_(self, end_token_id: int):
        super()._init_()
        self.end_token_id = end_token_id
    def _call_(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("*") or s.startswith("<end_of_turn>") or "ヒント" in s:
            continue
        cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", s).strip()
        if not cleaned:
            continue
        haiku_lines.append(cleaned)
        if len(haiku_lines) == 3:
            break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir")
    p.add_argument("--tokenizer_dir")
    p.add_argument("--train_jsonl")
    p.add_argument("--kigo_jsonl")
    p.add_argument("--output_dir")
    return p.parse_args()

def main():
    start_time = time.time()

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl, lines=True)
    df = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku", "_kigo"))
    df.rename(columns={
        "season_haiku": "season",
        "haiku": "ref_haiku",
        "5_mora_segment_1": "m5_1",
        "7_mora_segment": "m7",
        "5_mora_segment_2": "m5_2",
    }, inplace=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    if "<end_of_turn>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
    end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tok))
    model.eval()

    stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

    # Optimized parameters
    few_shot_k = 4
    num_targets = 30
    num_beams = 50
    num_return_sequences = 50
    num_beam_groups = 5
    diversity_penalty = 0.7

    template = (
        "以下は俳句の例です。\n"
        "季語: {word}\n"
        "季節: {season}\n"
        "構造: {haiku_structure}\n"
        "五音節１: {m5_1}\n"
        "七音節: {m7}\n"
        "五音節２: {m5_2}\n"
        "完成俳句:\n{ref_haiku}\n"
    )

    def retrieve_examples(target):
        pool = df[df.season == target.season].copy()
        pool = pool[pool.haiku_id != target.haiku_id]  # Exclude target haiku itself
        same = pool[pool.kigo_id == target.kigo_id]
        if len(same) >= few_shot_k:
            pool = same
        return pool.sample(min(few_shot_k, len(pool)), random_state=target.name)

    def build_prompt(target, examples):
        blocks = [template.format(**ex) for _, ex in examples.iterrows()]
        blocks.append(
            "次の条件に基づいて俳句を一つ作成してください。\n"
            f"季語: {target.word}\n"
            f"季節: {target.season}\n"
            f"構造: {target.haiku_structure}\n"
            f"五音節１: {target.m5_1}\n"
            f"七音節: {target.m7}\n"
            f"五音節２: {target.m5_2}\n"
            "完成俳句:\n"
            "<end_of_turn>\n"
        )
        return "\n".join(blocks)

    stats = []
    for _, tgt in df.sample(num_targets, random_state=42).iterrows():
        exs = retrieve_examples(tgt)
        prompt = build_prompt(tgt, exs)

        inp = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inp,
                max_new_tokens=30,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                early_stopping=True,
                stopping_criteria=stop_criteria
            )

        outputs, ppls = [], []
        for g in gen:
            raw = tok.decode(g[inp.input_ids.shape[-1]:], skip_special_tokens=False).strip()
            h = clean_haiku(raw)
            outputs.append(h)
            ppls.append(compute_perplexity(model, tok, h, device) if h else float("inf"))

        valid_haiku = [(h, p) for h, p in zip(outputs, ppls) if h and math.isfinite(p)]
        if valid_haiku:
            repr_haiku, best_ppl = min(valid_haiku, key=lambda x: x[1])
        else:
            repr_haiku, best_ppl = "", float("inf")

        repr_ppl = best_ppl
        mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
        word = tgt.word or ""
        kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)
        avg_ppl = sum(p for p in ppls if math.isfinite(p)) / len(ppls)

        stats.append({
            "kigo": tgt.word,
            "season": tgt.season,
            "haiku_structure": tgt.haiku_structure,
            "m5_1": tgt.m5_1,
            "m7": tgt.m7,
            "m5_2": tgt.m5_2,
            "ref_haiku": tgt.ref_haiku,
            "repr_haiku": repr_haiku,
            "repr_ppl": repr_ppl,
            "mora_rate": mora_rate,
            "kigo_rate": kigo_rate,
            "avg_ppl": avg_ppl,
        })

    pd.DataFrame(stats).to_csv(
        os.path.join(args.output_dir, "fewshot_eval.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    print(f"Done → few‑shot eval salvato in {args.output_dir}/fewshot_eval.csv")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo di esecuzione totale: {elapsed_time:.2f} secondi ({elapsed_time/60:.2f} minuti)")

if __name__ == "_main_":
    main()



#prova 5 (fewshot_eval_good)

# MODIFIED BEAM SEARCH — simplified: only avg perplexity, mora & kigo rates

import os
import re
import string
import argparse
import math
import time
import torch
import pandas as pd
import pyopenjtalk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("完成俳句:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir")
    p.add_argument("--tokenizer_dir")
    p.add_argument("--train_jsonl")
    p.add_argument("--kigo_jsonl")
    p.add_argument("--output_dir")
    return p.parse_args()

def main():
    start_time = time.time()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl, lines=True)
    df = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku", "_kigo"))
    df.rename(columns={
        "season_haiku": "season",
        "haiku": "ref_haiku",
        "5_mora_segment_1": "m5_1",
        "7_mora_segment": "m7",
        "5_mora_segment_2": "m5_2",
    }, inplace=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    if "<end_of_turn>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
    end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tok))
    model.eval()

    stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

    few_shot_k = 4
    num_targets = 30
    num_beams = 50
    num_return_sequences = 50
    num_beam_groups = 5
    diversity_penalty = 0.7

    def build_example_block(ex):
        if is_575(ex["ref_haiku"]):
            return (
                f"季語: {ex['word']}\n"
                f"季節: {ex['season']}\n"
                f"構造: {ex['haiku_structure']}\n"
                f"五音節１: {ex['m5_1']}\n"
                f"七音節: {ex['m7']}\n"
                f"五音節２: {ex['m5_2']}\n"
                f"完成俳句:\n{ex['ref_haiku']}\n<end_of_turn>\n"
            )
        else:
            return (
                f"季語: {ex['word']}\n"
                f"季節: {ex['season']}\n"
                f"構造: {ex['haiku_structure']}\n"
                f"完成俳句:\n{ex['ref_haiku']}\n<end_of_turn>\n"
            )

    def retrieve_examples(target):
        pool = df[df.season == target.season].copy()
        pool = pool[pool.haiku_id != target.haiku_id]
        same = pool[pool.kigo_id == target.kigo_id]
        if len(same) >= few_shot_k:
            pool = same
        return pool.sample(min(few_shot_k, len(pool)), random_state=target.name)

    def build_prompt(target, examples):
        blocks = [build_example_block(ex) for _, ex in examples.iterrows()]
        blocks.append(
            "以下の条件に従って、俳句を一つだけ作ってください（五七五でなくても構いませんが、三行で書いてください）。\n"
            f"季語: {target.word}\n"
            f"季節: {target.season}\n"
            f"構造: {target.haiku_structure}\n"
            "俳句:\n"
        )
        return "\n".join(blocks)

    stats = []
    raw_outputs = []

    for _, tgt in df.sample(num_targets, random_state=42).iterrows():
        exs = retrieve_examples(tgt)
        prompt = build_prompt(tgt, exs)
        inp = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen = model.generate(
                **inp,
                max_new_tokens=30,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                early_stopping=True,
                stopping_criteria=stop_criteria
            )

        outputs, ppls = [], []
        for g in gen:
            raw = tok.decode(g, skip_special_tokens=False).strip()
            h = clean_haiku(raw)
            outputs.append(h)
            ppls.append(compute_perplexity(model, tok, h, device) if h else float("inf"))
            raw_outputs.append({
                "prompt_kigo": tgt.word,
                "raw_output": raw,
                "cleaned_haiku": h
            })

        avg_ppl = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
        mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
        word = tgt.word or ""
        kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)

        stats.append({
            "kigo": tgt.word,
            "season": tgt.season,
            "haiku_structure": tgt.haiku_structure,
            "m5_1": tgt.m5_1,
            "m7": tgt.m7,
            "m5_2": tgt.m5_2,
            "ref_haiku": tgt.ref_haiku,
            "repr_haiku": outputs[0],
            "avg_ppl": avg_ppl,
            "mora_rate": mora_rate,
            "kigo_rate": kigo_rate,
        })

    pd.DataFrame(stats).to_csv(
        os.path.join(args.output_dir, "fewshot_eval.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame(raw_outputs).to_csv(
        os.path.join(args.output_dir, "fewshot_raw_outputs.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    elapsed = time.time() - start_time
    print(f"Done → few-shot eval salvato in {args.output_dir}/fewshot_eval.csv")
    print(f"Output grezzi salvati in {args.output_dir}/fewshot_raw_outputs.csv")
    print(f"Tempo di esecuzione totale: {elapsed:.2f} secondi ({elapsed/60:.2f} minuti)")

if __name__ == "__main__":
    main()


# prova 6
#MODIFIED BEAM SEARCH for Haiku generation — reranking by perplexity, BLEU evaluation

import os
import re
import string
import math
import time
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Set output directory on Google Drive
drive_output_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/fewshot_results"
os.makedirs(drive_output_dir, exist_ok=True)

# Patch torch.compile if using Python 3.12
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("完成俳句:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]  # character-level BLEU
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

# Manual args setup for Colab
model_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/tokenizer_model"          # change if different
tokenizer_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/tokenizer_model"  # change if different
train_jsonl = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/train_df_ready.jsonl"  # your uploaded file
kigo_jsonl = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/kigo_df_ready.jsonl"    # your uploaded file

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, local_files_only=True)
if "<end_of_turn>" not in tok.get_vocab():
    tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True
)
model.resize_token_embeddings(len(tok))
model.eval()

stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

# Load and prepare data
df_h = pd.read_json(train_jsonl, lines=True)
df_k = pd.read_json(kigo_jsonl, lines=True)
df = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku", "_kigo"))
df.rename(columns={
    "season_haiku": "season",
    "haiku": "ref_haiku",
    "5_mora_segment_1": "m5_1",
    "7_mora_segment": "m7",
    "5_mora_segment_2": "m5_2",
}, inplace=True)

# Beam Search Parameters
few_shot_k = 8
num_targets = 30
num_beams = 5
num_return_sequences = 5
num_beam_groups = 1
diversity_penalty = 0.0
max_new_tokens = 20

def build_example_block(ex):
    if is_575(ex["ref_haiku"]):
        return (
            f"季語: {ex['word']}\n"
            f"季節: {ex['season']}\n"
            f"構造: {ex['haiku_structure']}\n"
            f"五音節１: {ex['m5_1']}\n"
            f"七音節: {ex['m7']}\n"
            f"五音節２: {ex['m5_2']}\n"
            f"完成俳句:\n{ex['ref_haiku']}\n<end_of_turn>\n"
        )
    else:
        return (
            f"季語: {ex['word']}\n"
            f"季節: {ex['season']}\n"
            f"構造: {ex['haiku_structure']}\n"
            f"完成俳句:\n{ex['ref_haiku']}\n<end_of_turn>\n"
        )

def retrieve_examples(target):
    pool = df[df.season == target.season].copy()
    pool = pool[pool.haiku_id != target.haiku_id]
    same = pool[pool.kigo_id == target.kigo_id]
    if len(same) >= few_shot_k:
        pool = same
    return pool.sample(min(few_shot_k, len(pool)), random_state=target.name)

def build_prompt(target, examples):
    blocks = [build_example_block(ex) for _, ex in examples.iterrows()]
    blocks.append(
        "あなたは俳人です。次の情報をもとに、意味のある美しい俳句を三行で一つだけ生成してください（五七五でなくても構いません）。\n"
        f"季語: {target.word}\n"
        f"季節: {target.season}\n"
        f"構造: {target.haiku_structure}\n"
        "俳句:\n"
    )
    return "\n".join(blocks)

# Evaluation loop
stats = []
raw_outputs = []

for _, tgt in df.sample(num_targets, random_state=42).iterrows():
    exs = retrieve_examples(tgt)
    prompt = build_prompt(tgt, exs)
    inp = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            early_stopping=True,
            stopping_criteria=stop_criteria
        )

    outputs, ppls = [], []
    for g in gen:
        raw = tok.decode(g, skip_special_tokens=False).strip()
        h = clean_haiku(raw)
        outputs.append(h)
        ppls.append(compute_perplexity(model, tok, h, device) if h else float("inf"))
        raw_outputs.append({
            "prompt_kigo": tgt.word,
            "raw_output": raw,
            "cleaned_haiku": h
        })

    best_index = ppls.index(min(ppls))
    repr_haiku = outputs[best_index]
    bleu = compute_bleu(tgt.ref_haiku, repr_haiku)

    avg_ppl = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
    mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
    word = tgt.word or ""
    kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)

    stats.append({
        "kigo": tgt.word,
        "season": tgt.season,
        "haiku_structure": tgt.haiku_structure,
        "m5_1": tgt.m5_1,
        "m7": tgt.m7,
        "m5_2": tgt.m5_2,
        "ref_haiku": tgt.ref_haiku,
        "repr_haiku": repr_haiku,
        "avg_ppl": avg_ppl,
        "mora_rate": mora_rate,
        "kigo_rate": kigo_rate,
        "bleu": bleu
    })

# Save results to Drive
pd.DataFrame(stats).to_csv(
    os.path.join(drive_output_dir, "fewshot_eval.csv"), index=False, encoding="utf-8-sig"
)
pd.DataFrame(raw_outputs).to_csv(
    os.path.join(drive_output_dir, "fewshot_raw_outputs.csv"), index=False, encoding="utf-8-sig"
)

print("\n✅ Done. Results saved in Google Drive folder: haiku_results")


# prova 7
# MODIFIED BEAM SEARCH for Haiku generation — reranking by perplexity, BLEU evaluation

import os
import re
import string
import math
import time
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Set output directory on Google Drive
drive_output_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/fewshot_results"
os.makedirs(drive_output_dir, exist_ok=True)

# Patch torch.compile if using Python 3.12
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("俳句の例:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]  # character-level BLEU
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

# Manual args setup for Colab
model_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/tokenizer_model"
tokenizer_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/tokenizer_model"
train_jsonl = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/train_df_ready.jsonl"
kigo_jsonl = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/kigo_df_ready.jsonl"

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, local_files_only=True)
if "<end_of_turn>" not in tok.get_vocab():
    tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True
)
model.resize_token_embeddings(len(tok))
model.eval()

stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

# Load and prepare data
df_h = pd.read_json(train_jsonl, lines=True)
df_k = pd.read_json(kigo_jsonl, lines=True)
df = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku", "_kigo"))

df.rename(columns={
    "season_haiku": "season",
    "haiku": "ref_haiku",
    "5_mora_segment_1": "m5_1",
    "7_mora_segment": "m7",
    "5_mora_segment_2": "m5_2",
}, inplace=True)

# ⬇️ Filter only regular haiku
df = df[df["haiku_structure"] == "Regular"].copy()

# ⬇️ Format ref_haiku as 3 lines for 5-7-5
def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""
df["ref_haiku"] = df[["m5_1", "m7", "m5_2"]].apply(make_575, axis=1)

# Beam Search Parameters
few_shot_k = 5
num_targets = 20
max_new_tokens = 40
num_beams = 6
num_return_sequences = 6
num_beam_groups = 2
diversity_penalty = 0.5


def build_example_block(ex):
    return (
        f"季語: {ex['word']}\n"
        f"季節: {ex['season']}\n"
        f"構造: {ex['haiku_structure']}\n"
        f"五モーラ１: {ex['m5_1']}\n"
        f"七モーラ: {ex['m7']}\n"
        f"五モーラ２: {ex['m5_2']}\n"
        f"俳句の例:\n{ex['ref_haiku']}\n<end_of_turn>\n"
    )

def retrieve_examples(target):
    pool = df[df.season == target.season].copy()
    pool = pool[pool.haiku_id != target.haiku_id]
    same = pool[pool.kigo_id == target.kigo_id]
    if len(same) >= few_shot_k:
        pool = same
    return pool.sample(min(few_shot_k, len(pool)), random_state=target.name)

def build_prompt(target, examples):
    blocks = [build_example_block(ex) for _, ex in examples.iterrows()]
    blocks.append(
        f"季節　{target.season} から季語「{target.word}」を使って、{target.haiku_structure} の形式で3行の俳句を作成してください。\n"
        "俳句:\n"
        "<end_of_turn>\n"
    )
    return "\n".join(blocks)

# Evaluation loop
stats = []
raw_outputs = []

for _, tgt in df.sample(num_targets, random_state=42).iterrows():
    exs = retrieve_examples(tgt)
    prompt = build_prompt(tgt, exs)
    inp = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            early_stopping=True,
            stopping_criteria=stop_criteria
        )

    outputs, ppls, bleus, final_scores = [], [], [], []
    for g in gen:
        raw = tok.decode(g, skip_special_tokens=False).strip()
        h = clean_haiku(raw)

        #print(f"\n🔹 Raw output:\n{raw}\n")
        #print(f"🔸 Cleaned haiku:\n{h}\n")

        outputs.append(h)
        ppl = compute_perplexity(model, tok, h, device) if h else float("inf")
        bleu = compute_bleu(tgt.ref_haiku, h) if h else 0.0

        ppls.append(ppl)
        bleus.append(bleu)
        final_scores.append(bleu - 0.2 * math.log(ppl) if math.isfinite(ppl) and ppl > 0 else -float("inf"))

        raw_outputs.append({
            "prompt_kigo": tgt.word,
            "raw_output": raw,
            "cleaned_haiku": h
        })

    best_index = final_scores.index(max(final_scores))
    repr_haiku = outputs[best_index]
    bleu = bleus[best_index]

    avg_ppl = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
    mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
    word = tgt.word or ""
    kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)

    stats.append({
        "kigo": tgt.word,
        "season": tgt.season,
        "haiku_structure": tgt.haiku_structure,
        "m5_1": tgt.m5_1,
        "m7": tgt.m7,
        "m5_2": tgt.m5_2,
        "ref_haiku": tgt.ref_haiku,
        "repr_haiku": repr_haiku,
        "avg_ppl": avg_ppl,
        "mora_rate": mora_rate,
        "kigo_rate": kigo_rate,
        "bleu": bleu
    })

# Save results to Drive
pd.DataFrame(stats).to_csv(
    os.path.join(drive_output_dir, "fewshot_eval.csv"), index=False, encoding="utf-8-sig"
)
pd.DataFrame(raw_outputs).to_csv(
    os.path.join(drive_output_dir, "fewshot_raw_outputs.csv"), index=False, encoding="utf-8-sig"
)

print("\n✅ Done. Results saved in Google Drive folder: haiku_results")


# prova 8
# MODIFIED BEAM SEARCH for Haiku generation — reranking by perplexity, BLEU evaluation

import os
import re
import string
import math
import time
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Set output directory on Google Drive
drive_output_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/fewshot_results"
os.makedirs(drive_output_dir, exist_ok=True)

# Patch torch.compile if using Python 3.12
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("俳句の例:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

# Manual args setup for Colab
model_dir     = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/tokenizer_model"
tokenizer_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/tokenizer_model"
train_jsonl   = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/train_df_ready.jsonl"
kigo_jsonl    = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/kigo_df_ready.jsonl"

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, local_files_only=True)
if "<end_of_turn>" not in tok.get_vocab():
    tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True
)
model.resize_token_embeddings(len(tok))
model.eval()

stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

# Load and prepare data
df_h = pd.read_json(train_jsonl, lines=True)
df_k = pd.read_json(kigo_jsonl,  lines=True)
df   = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
df.rename(columns={
    "season_haiku":    "season",
    "haiku":           "ref_haiku",
    "5_mora_segment_1":"m5_1",
    "7_mora_segment":  "m7",
    "5_mora_segment_2":"m5_2",
}, inplace=True)

# Filter only Regular haiku
df = df[df["haiku_structure"] == "Regular"].copy()

# Format ref_haiku as 3 lines
def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""
df["ref_haiku"] = df[["m5_1","m7","m5_2"]].apply(make_575, axis=1)

# Gemma‑style few‑shot prompt builders
def build_example_block_gemma(ex):
    return (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n\n"
        f"季語: {ex['word']}\n"
        f"季節: {ex['season']}\n"
        f"構造: {ex['haiku_structure']}\n"
        f"俳句:\n{ex['ref_haiku']}\n"
        "<end_of_turn>\n"
    )

def build_prompt_gemma(target, examples):
    prompt_blocks = [build_example_block_gemma(ex) for _, ex in examples.iterrows()]
    target_block = (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n\n"
        f"季語: {target.word}\n"
        f"季節: {target.season}\n"
        f"構造: {target.haiku_structure}\n"
    )
    # add hint for regular
    if target.haiku_structure == "Regular" and all(pd.notnull([target.m5_1, target.m7, target.m5_2])):
        target_block += f"ヒント: {target.m5_1} / {target.m7} / {target.m5_2}\n"
    target_block += "俳句:\n<end_of_turn>\n"
    return "\n".join(prompt_blocks + [target_block])

# Beam search & evaluation settings
few_shot_k         = 5
num_targets        = 20
max_new_tokens     = 40
num_beams          = 6
num_return_sequences = 6
num_beam_groups    = 2
diversity_penalty  = 0.5

def retrieve_examples(target):
    pool = df[df.season == target.season].copy()
    pool = pool[pool.haiku_id != target.haiku_id]
    same = pool[pool.kigo_id == target.kigo_id]
    if len(same) >= few_shot_k:
        pool = same
    return pool.sample(min(few_shot_k, len(pool)), random_state=target.name)

# Run evaluation
stats = []
raw_outputs = []

for _, tgt in df.sample(num_targets, random_state=42).iterrows():
    exs    = retrieve_examples(tgt)
    prompt = build_prompt_gemma(tgt, exs)
    inp    = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            early_stopping=True,
            stopping_criteria=stop_criteria
        )

    outputs, ppls, bleus, final_scores = [], [], [], []
    for g in gen:
        raw = tok.decode(g, skip_special_tokens=False).strip()
        h   = clean_haiku(raw)

        outputs.append(h)
        ppl  = compute_perplexity(model, tok, h, device) if h else float("inf")
        bleu = compute_bleu(tgt.ref_haiku, h) if h else 0.0

        ppls.append(ppl)
        bleus.append(bleu)
        final_scores.append(
            bleu - 0.2 * math.log(ppl)
            if math.isfinite(ppl) and ppl>0 else -float("inf")
        )

        raw_outputs.append({
            "prompt_kigo":   tgt.word,
            "raw_output":    raw,
            "cleaned_haiku": h
        })

    best_idx   = final_scores.index(max(final_scores))
    repr_haiku = outputs[best_idx]
    repr_bleu  = bleus[best_idx]

    avg_ppl   = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
    mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
    word      = tgt.word or ""
    kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)

    stats.append({
        "kigo":           tgt.word,
        "season":         tgt.season,
        "haiku_structure":tgt.haiku_structure,
        "m5_1":           tgt.m5_1,
        "m7":             tgt.m7,
        "m5_2":           tgt.m5_2,
        "ref_haiku":      tgt.ref_haiku,
        "repr_haiku":     repr_haiku,
        "avg_ppl":        avg_ppl,
        "mora_rate":      mora_rate,
        "kigo_rate":      kigo_rate,
        "bleu":           repr_bleu
    })

# Save results
pd.DataFrame(stats).to_csv(
    os.path.join(drive_output_dir, "fewshot_eval.csv"),
    index=False, encoding="utf-8-sig"
)
pd.DataFrame(raw_outputs).to_csv(
    os.path.join(drive_output_dir, "fewshot_raw_outputs.csv"),
    index=False, encoding="utf-8-sig"
)

print("\n✅ Done. Results saved in fewshot_results on your Drive.")


# prova 9
# MODIFIED BEAM SEARCH for Haiku generation — reranking by perplexity, BLEU evaluation, GPT-3.5 scoring, structure and kigo filtering

import os
import re
import string
import math
import time
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Set output directory on Google Drive
drive_output_dir = "/content/drive/MyDrive/THESIS/haiku_project/fewshot_results"
os.makedirs(drive_output_dir, exist_ok=True)

# Patch torch.compile if using Python 3.12
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("俳句の例:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

model_dir     = "/content/drive/MyDrive/THESIS/haiku_project/tokenizer_model"
tokenizer_dir = "/content/drive/MyDrive/THESIS/haiku_project/tokenizer_model"
train_jsonl   = "/content/drive/MyDrive/THESIS/haiku_project/train_df_ready.jsonl"
kigo_jsonl    = "/content/drive/MyDrive/THESIS/haiku_project/kigo_df_ready.jsonl"

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, local_files_only=True)
if "<end_of_turn>" not in tok.get_vocab():
    tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True
)
model.resize_token_embeddings(len(tok))
model.eval()

stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

# Load and prepare data
df_h = pd.read_json(train_jsonl, lines=True)
df_k = pd.read_json(kigo_jsonl,  lines=True)
df   = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
df.rename(columns={
    "season_haiku":    "season",
    "haiku":           "ref_haiku",
    "5_mora_segment_1":"m5_1",
    "7_mora_segment":  "m7",
    "5_mora_segment_2":"m5_2",
}, inplace=True)

# Filter only Regular haiku
df = df[df["haiku_structure"] == "Regular"].copy()

def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""
df["ref_haiku"] = df[["m5_1","m7","m5_2"]].apply(make_575, axis=1)

def build_example_block_gemma(ex):
    return (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n"
        "5音の行 / 7音の行 / 5音の行 の三行で構成してください。\n"
        "余韻を大切にしてください。\n"
        f"季語: {ex['word']}\n"
        f"季節: {ex['season']}\n"
        f"構造: {ex['haiku_structure']}\n"
        f"俳句:\n{ex['ref_haiku']}\n"
        "<end_of_turn>\n"
    )

def build_prompt_gemma(target, examples):
    prompt_blocks = [build_example_block_gemma(ex) for _, ex in examples.iterrows()]
    target_block = (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n"
        "5音の行 / 7音の行 / 5音の行 の三行で構成してください。\n"
        "余韻を大切にしてください。\n"
        f"季語: {target.word}\n"
        f"季節: {target.season}\n"
        f"構造: {target.haiku_structure}\n"
    )
    if target.haiku_structure == "Regular" and all(pd.notnull([target.m5_1, target.m7, target.m5_2])):
        target_block += f"ヒント: {target.m5_1} / {target.m7} / {target.m5_2}\n"
    target_block += "俳句:\n<end_of_turn>\n"
    return "\n".join(prompt_blocks + [target_block])

few_shot_k         = 5
num_targets        = 20
max_new_tokens     = 40
num_beams          = 6
num_return_sequences = 6
num_beam_groups    = 2
diversity_penalty  = 0.5

def retrieve_examples(target):
    pool = df[df.season == target.season].copy()
    pool = pool[pool.haiku_id != target.haiku_id]
    same = pool[pool.kigo_id == target.kigo_id]
    if len(same) >= few_shot_k:
        pool = same
    return pool.sample(min(few_shot_k, len(pool)), random_state=target.name)

stats = []
raw_outputs = []

for _, tgt in df.sample(num_targets, random_state=42).iterrows():
    exs    = retrieve_examples(tgt)
    prompt = build_prompt_gemma(tgt, exs)
    inp    = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            early_stopping=True,
            stopping_criteria=stop_criteria
        )

    outputs, ppls, bleus, final_scores = [], [], [], []
    for g in gen:
        raw = tok.decode(g, skip_special_tokens=False).strip()
        h   = clean_haiku(raw)

        if h:
            mora_counts = [count_mora(line) for line in h.splitlines()]
            print("\n📄 Generated Haiku:\n", h)
            print(f"🧮 Mora counts per line: {mora_counts}")
            print(f"✅ is_575: {is_575(h)} | Kigo: {'✅' if tgt.word in h else '❌'}")
            print(f"📌 Season: {tgt.season} | Kigo: {tgt.word}")

        outputs.append(h)
        ppl  = compute_perplexity(model, tok, h, device)
        bleu = compute_bleu(tgt.ref_haiku, h)

        ppls.append(ppl)
        bleus.append(bleu)
        final_scores.append(
            bleu - 0.2 * math.log(ppl)
            if math.isfinite(ppl) and ppl > 0 else -float("inf")
        )

        raw_outputs.append({
            "prompt_kigo":   tgt.word,
            "raw_output":    raw,
            "cleaned_haiku": h
        })

    if not outputs:
        continue

    best_idx   = final_scores.index(max(final_scores))
    repr_haiku = outputs[best_idx]
    repr_bleu  = bleus[best_idx]

    avg_ppl   = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
    mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
    word      = tgt.word or ""
    kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)

    stats.append({
        "kigo":           tgt.word,
        "season":         tgt.season,
        "haiku_structure":tgt.haiku_structure,
        "m5_1":           tgt.m5_1,
        "m7":             tgt.m7,
        "m5_2":           tgt.m5_2,
        "ref_haiku":      tgt.ref_haiku,
        "repr_haiku":     repr_haiku,
        "avg_ppl":        avg_ppl,
        "mora_rate":      mora_rate,
        "kigo_rate":      kigo_rate,
        "bleu":           repr_bleu
    })

pd.DataFrame(stats).to_csv(
    os.path.join(drive_output_dir, "fewshot_eval.csv"),
    index=False, encoding="utf-8-sig"
)
pd.DataFrame(raw_outputs).to_csv(
    os.path.join(drive_output_dir, "fewshot_raw_outputs.csv"),
    index=False, encoding="utf-8-sig"
)

print("\n✅ Done. Results saved in fewshot_results on your Drive.")


# prova 10 — GPT-3.5 scoring + author diversity + debug mora count
import os
import re
import string
import math
import time
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    StoppingCriteria, StoppingCriteriaList,
)

try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
except ImportError:
    openai = None

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id): self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs): return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    text = japanese_text.replace(" ", "")  # 🚨 Strip all spaces
    phonemes = pyopenjtalk.g2p(text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = [line.replace(" ", "") for line in haiku.strip().split("\n")]  # 🚨 Strip per line
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw):
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line: break
        if not line or line.startswith("#") or "ヒント" in line: continue
        if line.startswith("俳句:"): haiku_started = True; continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + "]+", "", line).strip()
            if cleaned: haiku_lines.append(cleaned)
            if len(haiku_lines) == 3: break
    while len(haiku_lines) < 3: haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=SmoothingFunction().method1, weights=(1.0,))

def gpt35_score(prompt, haiku):
    if not openai: return None
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a haiku critic. Rate from 1 to 10 based on beauty, fluency, seasonal relevance."},
                {"role": "user", "content": f"Prompt:\n{prompt}\n\nHaiku:\n{haiku}"}
            ],
            temperature=0.7
        )
        text = response['choices'][0]['message']['content']
        return int(re.search(r"\d+", text).group())
    except Exception as e:
        print(f"GPT-3.5 scoring error: {e}")
        return None

# Paths
model_dir = "/content/drive/MyDrive/THESIS/haiku_project/tokenizer_model"
tokenizer_dir = model_dir
train_jsonl = "/content/drive/MyDrive/THESIS/haiku_project/train_df_ready.jsonl"
kigo_jsonl = "/content/drive/MyDrive/THESIS/haiku_project/kigo_df_ready.jsonl"
out_dir = "/content/drive/MyDrive/THESIS/haiku_project/fewshot_results"
os.makedirs(out_dir, exist_ok=True)

# Load model/tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, local_files_only=True)
if "<end_of_turn>" not in tok.get_vocab():
    tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, local_files_only=True)  #device_map="auto"
model.resize_token_embeddings(len(tok))
model.eval()

stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

# Load and prepare data
df_h = pd.read_json(train_jsonl, lines=True)
df_k = pd.read_json(kigo_jsonl,  lines=True)
df   = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
df.rename(columns={
    "season_haiku":    "season",
    "haiku":           "ref_haiku",
    "5_mora_segment_1":"m5_1",
    "7_mora_segment":  "m7",
    "5_mora_segment_2":"m5_2",
}, inplace=True)

# Filter only Regular haiku
df = df[df["haiku_structure"] == "Regular"].copy()


def build_example_block(ex):
    return (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n"
        "5音の行 / 7音の行 / 5音の行 の三行で構成してください。\n"
        "余韻を大切にしてください。\n"
        f"季語: {ex['word']}\n季節: {ex['season']}\n構造: {ex['haiku_structure']}\n俳句:\n{ex['ref_haiku']}\n<end_of_turn>\n"
    )

def build_prompt(target, examples):
    blocks = [build_example_block(ex) for _, ex in examples.iterrows()]
    tgt = f"以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n"
    tgt += f"季語: {target.word}\n季節: {target.season}\n構造: {target.haiku_structure}\n"
    if all(pd.notnull([target["m5_1"], target["m7"], target["m5_2"]])):
        tgt += f"ヒント: {target['m5_1']} / {target['m7']} / {target['m5_2']}\n"
    return "\n".join(blocks + [tgt + "俳句:\n<end_of_turn>\n"])

def retrieve_examples(target, k=5):
    pool = df[(df.season == target.season) & (df.haiku_id != target.haiku_id)].copy()
    if "author" in df.columns:
        used_authors = {target.get("author")}
        diverse_pool = pool[~pool.author.isin(used_authors)]
        if len(diverse_pool) >= k:
            return diverse_pool.sample(k, random_state=target.name)
    return pool.sample(min(k, len(pool)), random_state=target.name)

# Generation loop
stats, raw_outputs = [], []
for _, tgt in df.sample(20, random_state=42).iterrows():
    examples = retrieve_examples(tgt)
    prompt = build_prompt(tgt, examples)
    inp = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = model.generate(
            **inp,
            max_new_tokens=40,
            num_beams=6,
            num_beam_groups=2,
            num_return_sequences=6,
            diversity_penalty=0.5,
            stopping_criteria=StoppingCriteriaList([StopOnEndOfTurn(end_token_id)]),
            early_stopping=True
        )

    haikus, scores = [], []
    for g in gen:
        raw = tok.decode(g, skip_special_tokens=False).strip()
        h = clean_haiku(raw)
        if h:
            mora_counts = [count_mora(line) for line in h.splitlines()]
            print("\n📄", h, "\n🧮 Mora:", mora_counts, "→ is_575:", is_575(h))
            bleu = compute_bleu(tgt.ref_haiku, h)
            ppl = compute_perplexity(model, tok, h, device)
            gpt_score = gpt35_score(prompt, h)
            final = bleu - 0.2 * math.log(ppl) if ppl > 0 and math.isfinite(ppl) else -float("inf")
            haikus.append((h, bleu, ppl, final, gpt_score))
        raw_outputs.append({"prompt_kigo": tgt.word, "raw_output": raw, "cleaned_haiku": h})

    if not haikus: continue
    best = max(haikus, key=lambda x: x[3])
    stats.append({
        "kigo": tgt.word,
        "season": tgt.season,
        "ref_haiku": tgt.ref_haiku,
        "repr_haiku": best[0],
        "avg_ppl": sum(h[2] for h in haikus) / len(haikus),
        "bleu": best[1],
        "mora_rate": sum(is_575(h[0]) for h in haikus) / len(haikus),
        "kigo_rate": sum(1 for h in haikus if tgt.word in h[0]) / len(haikus),
        "gpt35_score": best[4]
    })

# Save
pd.DataFrame(stats).to_csv(f"{out_dir}/fewshot_eval.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(raw_outputs).to_csv(f"{out_dir}/fewshot_raw_outputs.csv", index=False, encoding="utf-8-sig")
print("✅ Done.")


# prova 11
# MODIFIED BEAM SEARCH for Haiku generation — reranking by perplexity, BLEU evaluation

import os
import re
import string
import math
import time
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Set output directory on Google Drive
drive_output_dir = "/content/drive/MyDrive/THESIS/haiku_project/fewshot_results"
os.makedirs(drive_output_dir, exist_ok=True)

# Patch torch.compile if using Python 3.12
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("俳句の例:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            cleaned = cleaned.replace(" ", "") #removing spaces
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

# Manual args setup for Colab
model_dir     = "/content/drive/MyDrive/THESIS/haiku_project/tokenizer_model"
tokenizer_dir = "/content/drive/MyDrive/THESIS/haiku_project/tokenizer_model"
train_jsonl   = "/content/drive/MyDrive/THESIS/haiku_project/train_df_ready.jsonl"
kigo_jsonl    = "/content/drive/MyDrive/THESIS/haiku_project/kigo_df_ready.jsonl"

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, local_files_only=True)
if "<end_of_turn>" not in tok.get_vocab():
    tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True
)
model.resize_token_embeddings(len(tok))
model.eval()

stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

# Load and prepare data
df_h = pd.read_json(train_jsonl, lines=True)
df_k = pd.read_json(kigo_jsonl,  lines=True)
df   = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
df.rename(columns={
    "season_haiku":    "season",
    "haiku":           "ref_haiku",
    "5_mora_segment_1":"m5_1",
    "7_mora_segment":  "m7",
    "5_mora_segment_2":"m5_2",
}, inplace=True)

# Filter only Regular haiku
df = df[df["haiku_structure"] == "Regular"].copy()

# Format ref_haiku as 3 lines
def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""
df["ref_haiku"] = df[["m5_1","m7","m5_2"]].apply(make_575, axis=1)

# Gemma‑style few‑shot prompt builders
def build_example_block_gemma(ex):
    return (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n\n"
        f"季語: {ex['word']}\n"
        f"季節: {ex['season']}\n"
        f"構造: {ex['haiku_structure']}\n"
        f"俳句:\n{ex['ref_haiku']}\n"
        "<end_of_turn>\n"
    )

def build_prompt_gemma(target, examples):
    prompt_blocks = [build_example_block_gemma(ex) for _, ex in examples.iterrows()]
    target_block = (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n\n"
        f"季語: {target.word}\n"
        f"季節: {target.season}\n"
        f"構造: {target.haiku_structure}\n"
    )
    # add hint for regular
    if target.haiku_structure == "Regular" and all(pd.notnull([target.m5_1, target.m7, target.m5_2])):
        target_block += f"ヒント: {target.m5_1} / {target.m7} / {target.m5_2}\n"
    target_block += "俳句:\n<end_of_turn>\n"
    return "\n".join(prompt_blocks + [target_block])

# Beam search & evaluation settings
few_shot_k         = 6
num_targets        = 20
max_new_tokens     = 32
num_beams          = 6
num_return_sequences = 5
num_beam_groups    = 3
diversity_penalty  = 0.7

def retrieve_examples(target):
    pool = df[df.season == target.season].copy()
    pool = pool[pool.haiku_id != target.haiku_id]

    # If many same-kigo haiku exist, use them (as before)
    same = pool[pool.kigo_id == target.kigo_id]
    if len(same) >= few_shot_k:
        pool = same

    # Prioritize diverse authors
    unique_authors = pool['author'].dropna().unique()
    selected = []

    rng = pd.Series(unique_authors).sample(frac=1, random_state=target.name)  # shuffle

    for author in rng:
        samples = pool[pool.author == author]
        if not samples.empty:
            selected.append(samples.sample(1, random_state=target.name))
        if len(selected) == few_shot_k:
            break

    # Fallback if not enough diverse authors
    while len(selected) < few_shot_k:
        selected.append(pool.sample(1, random_state=target.name + len(selected)))

    return pd.concat(selected).reset_index(drop=True)


# Run evaluation
stats = []
raw_outputs = []

for _, tgt in df.sample(num_targets, random_state=42).iterrows():
    exs    = retrieve_examples(tgt)
    prompt = build_prompt_gemma(tgt, exs)
    inp    = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            early_stopping=True,
            stopping_criteria=stop_criteria
        )

    outputs, ppls, bleus, final_scores = [], [], [], []
    for g in gen:
        raw = tok.decode(g, skip_special_tokens=False).strip()
        h   = clean_haiku(raw)

        outputs.append(h)
        ppl  = compute_perplexity(model, tok, h, device) if h else float("inf")
        bleu = compute_bleu(tgt.ref_haiku, h) if h else 0.0

        ppls.append(ppl)
        bleus.append(bleu)
        final_scores.append(
            bleu - 0.2 * math.log(ppl)
            if math.isfinite(ppl) and ppl>0 else -float("inf")
        )

        raw_outputs.append({
            "prompt_kigo":   tgt.word,
            "raw_output":    raw,
            "cleaned_haiku": h
        })

    best_idx   = final_scores.index(max(final_scores))
    repr_haiku = outputs[best_idx]
    repr_bleu  = bleus[best_idx]

    avg_ppl   = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
    mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
    word      = tgt.word or ""
    kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)

    stats.append({
        "kigo":           tgt.word,
        "season":         tgt.season,
        "haiku_structure":tgt.haiku_structure,
        "m5_1":           tgt.m5_1,
        "m7":             tgt.m7,
        "m5_2":           tgt.m5_2,
        "ref_haiku":      tgt.ref_haiku,
        "repr_haiku":     repr_haiku,
        "avg_ppl":        avg_ppl,
        "mora_rate":      mora_rate,
        "kigo_rate":      kigo_rate,
        "bleu":           repr_bleu
    })

# Save results
pd.DataFrame(stats).to_csv(
    os.path.join(drive_output_dir, "fewshot_eval.csv"),
    index=False, encoding="utf-8-sig"
)
pd.DataFrame(raw_outputs).to_csv(
    os.path.join(drive_output_dir, "fewshot_raw_outputs.csv"),
    index=False, encoding="utf-8-sig"
)

print("\n✅ Done. Results saved in fewshot_results on your Drive.")


# prova 12
# MODIFIED BEAM SEARCH for Haiku generation — reranking by perplexity, BLEU evaluation

import os
import re
import string
import math
import time
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Set output directory on Google Drive
drive_output_dir = "/content/drive/MyDrive/THESIS/haiku_project/fewshot_results"
os.makedirs(drive_output_dir, exist_ok=True)

# Patch torch.compile if using Python 3.12
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("俳句の例:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            cleaned = cleaned.replace(" ", "") #removing spaces
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

# Manual args setup for Colab
model_dir     = "/content/drive/MyDrive/THESIS/haiku_project/tokenizer_model"
tokenizer_dir = "/content/drive/MyDrive/THESIS/haiku_project/tokenizer_model"
train_jsonl   = "/content/drive/MyDrive/THESIS/haiku_project/train_df_ready.jsonl"
kigo_jsonl    = "/content/drive/MyDrive/THESIS/haiku_project/kigo_df_ready.jsonl"

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, local_files_only=True)
if "<end_of_turn>" not in tok.get_vocab():
    tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True
)
model.resize_token_embeddings(len(tok))
model.eval()

stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

# Load and prepare data
df_h = pd.read_json(train_jsonl, lines=True)
df_k = pd.read_json(kigo_jsonl,  lines=True)
df   = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
df.rename(columns={
    "season_haiku":    "season",
    "haiku":           "ref_haiku",
    "5_mora_segment_1":"m5_1",
    "7_mora_segment":  "m7",
    "5_mora_segment_2":"m5_2",
}, inplace=True)

# Filter only Regular haiku
df = df[df["haiku_structure"] == "Regular"].copy()

# Format ref_haiku as 3 lines
def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""
df["ref_haiku"] = df[["m5_1","m7","m5_2"]].apply(make_575, axis=1)

# Gemma‑style few‑shot prompt builders
def build_example_block_gemma(ex):
    return (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n\n"
        f"季語: {ex['word']}\n"
        f"季節: {ex['season']}\n"
        f"構造: {ex['haiku_structure']}\n"
        f"俳句:\n{ex['ref_haiku']}\n"
        "<end_of_turn>\n"
    )

def build_prompt_gemma(target, examples):
    prompt_blocks = [build_example_block_gemma(ex) for _, ex in examples.iterrows()]
    target_block = (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n\n"
        f"季語: {target.word}\n"
        f"季節: {target.season}\n"
        f"構造: {target.haiku_structure}\n"
    )
    # add hint for regular
    if target.haiku_structure == "Regular" and all(pd.notnull([target.m5_1, target.m7, target.m5_2])):
        target_block += f"ヒント: {target.m5_1} / {target.m7} / {target.m5_2}\n"
    target_block += "俳句:\n<end_of_turn>\n"
    return "\n".join(prompt_blocks + [target_block])

# Beam search & evaluation settings
few_shot_k         = 4
num_targets        = 40
max_new_tokens     = 20
num_beams          = 4
num_return_sequences = 4
num_beam_groups    = 2
diversity_penalty  = 0.4

def retrieve_examples(target):
    pool = df[df.season == target.season].copy()
    pool = pool[pool.haiku_id != target.haiku_id]

    # If many same-kigo haiku exist, use them (as before)
    same = pool[pool.kigo_id == target.kigo_id]
    if len(same) >= few_shot_k:
        pool = same

    # Prioritize diverse authors
    unique_authors = pool['author'].dropna().unique()
    selected = []

    rng = pd.Series(unique_authors).sample(frac=1, random_state=target.name)  # shuffle

    for author in rng:
        samples = pool[pool.author == author]
        if not samples.empty:
            selected.append(samples.sample(1, random_state=target.name))
        if len(selected) == few_shot_k:
            break

    # Fallback if not enough diverse authors
    while len(selected) < few_shot_k:
        selected.append(pool.sample(1, random_state=target.name + len(selected)))

    return pd.concat(selected).reset_index(drop=True)


# Run evaluation
stats = []
raw_outputs = []

for _, tgt in df.sample(num_targets, random_state=42).iterrows():
    exs    = retrieve_examples(tgt)
    prompt = build_prompt_gemma(tgt, exs)
    inp    = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            early_stopping=True,
            stopping_criteria=stop_criteria
        )

    outputs, ppls, bleus, final_scores = [], [], [], []
    for g in gen:
        raw = tok.decode(g, skip_special_tokens=False).strip()
        h   = clean_haiku(raw)

        outputs.append(h)
        ppl  = compute_perplexity(model, tok, h, device) if h else float("inf")
        bleu = compute_bleu(tgt.ref_haiku, h) if h else 0.0

        ppls.append(ppl)
        bleus.append(bleu)
        final_scores.append(
            bleu - 0.2 * math.log(ppl)
            if math.isfinite(ppl) and ppl>0 else -float("inf")
        )

        raw_outputs.append({
            "prompt_kigo":   tgt.word,
            "raw_output":    raw,
            "cleaned_haiku": h
        })

    best_idx   = final_scores.index(max(final_scores))
    repr_haiku = outputs[best_idx]
    repr_bleu  = bleus[best_idx]

    avg_ppl   = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
    mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
    word      = tgt.word or ""
    kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)

    stats.append({
        "kigo":           tgt.word,
        "season":         tgt.season,
        "haiku_structure":tgt.haiku_structure,
        "m5_1":           tgt.m5_1,
        "m7":             tgt.m7,
        "m5_2":           tgt.m5_2,
        "ref_haiku":      tgt.ref_haiku,
        "repr_haiku":     repr_haiku,
        "avg_ppl":        avg_ppl,
        "mora_rate":      mora_rate,
        "kigo_rate":      kigo_rate,
        "bleu":           repr_bleu
    })

# Save results
pd.DataFrame(stats).to_csv(
    os.path.join(drive_output_dir, "fewshot_eval.csv"),
    index=False, encoding="utf-8-sig"
)
pd.DataFrame(raw_outputs).to_csv(
    os.path.join(drive_output_dir, "fewshot_raw_outputs.csv"),
    index=False, encoding="utf-8-sig"
)

print("\n✅ Done. Results saved in fewshot_results on your Drive.")



# haiku_beamsearch_elyza.py - prova 13
# Haiku generation with ELYZA (LLaMA) — Beam Search + PPL+BLEU reranking

import os
import re
import string
import math
import time
import argparse
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--kigo_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("俳句の例:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            cleaned = cleaned.replace(" ", "")
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""

def build_example_block_gemma(ex):
    return (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n\n"
        f"季語: {ex['word']}\n"
        f"季節: {ex['season']}\n"
        f"構造: {ex['haiku_structure']}\n"
        f"俳句:\n{ex['ref_haiku']}\n"
        "<end_of_turn>\n"
    )

def build_prompt_gemma(target, examples):
    prompt_blocks = [build_example_block_gemma(ex) for _, ex in examples.iterrows()]
    target_block = (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n\n"
        f"季語: {target.word}\n"
        f"季節: {target.season}\n"
        f"構造: {target.haiku_structure}\n"
    )
    if target.haiku_structure == "Regular" and all(pd.notnull([target.m5_1, target.m7, target.m5_2])):
        target_block += f"ヒント: {target.m5_1} / {target.m7} / {target.m5_2}\n"
    target_block += "俳句:\n<end_of_turn>\n"
    return "\n".join(prompt_blocks + [target_block])

def retrieve_examples(target, pool, few_shot_k=6):
    pool = pool[pool.season == target.season].copy()
    pool = pool[pool.haiku_id != target.haiku_id]
    same = pool[pool.kigo_id == target.kigo_id]
    if len(same) >= few_shot_k:
        pool = same
    unique_authors = pool['author'].dropna().unique()
    selected = []
    rng = pd.Series(unique_authors).sample(frac=1, random_state=target.name)
    for author in rng:
        samples = pool[pool.author == author]
        if not samples.empty:
            selected.append(samples.sample(1, random_state=target.name))
        if len(selected) == few_shot_k:
            break
    while len(selected) < few_shot_k:
        selected.append(pool.sample(1, random_state=target.name + len(selected)))
    return pd.concat(selected).reset_index(drop=True)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    if "<end_of_turn>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
    end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tok))
    model.eval()

    stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl,  lines=True)
    df = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
    df.rename(columns={
        "season_haiku":    "season",
        "haiku":           "ref_haiku",
        "5_mora_segment_1":"m5_1",
        "7_mora_segment":  "m7",
        "5_mora_segment_2":"m5_2",
    }, inplace=True)
    df = df[df["haiku_structure"] == "Regular"].copy()
    df["ref_haiku"] = df[["m5_1","m7","m5_2"]].apply(make_575, axis=1)

    few_shot_k = 6
    num_targets = 20
    max_new_tokens = 20
    num_beams = 6
    num_return_sequences = 5
    num_beam_groups = 3
    diversity_penalty = 0.7

    stats = []
    raw_outputs = []

    for _, tgt in df.sample(num_targets, random_state=42).iterrows():
        exs = retrieve_examples(tgt, df, few_shot_k)
        prompt = build_prompt_gemma(tgt, exs)
        inp = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                early_stopping=True,
                stopping_criteria=stop_criteria
            )

        outputs, ppls, bleus, final_scores = [], [], [], []
        for g in gen:
            raw = tok.decode(g, skip_special_tokens=False).strip()
            h = clean_haiku(raw)
            outputs.append(h)
            ppl = compute_perplexity(model, tok, h, device) if h else float("inf")
            bleu = compute_bleu(tgt.ref_haiku, h) if h else 0.0
            ppls.append(ppl)
            bleus.append(bleu)
            final_scores.append(bleu - 0.2 * math.log(ppl) if math.isfinite(ppl) and ppl > 0 else -float("inf"))
            raw_outputs.append({
                "prompt_kigo":   tgt.word,
                "raw_output":    raw,
                "cleaned_haiku": h
            })

        best_idx = final_scores.index(max(final_scores))
        repr_haiku = outputs[best_idx]
        repr_bleu = bleus[best_idx]
        avg_ppl = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
        mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
        word = tgt.word or ""
        kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)

        stats.append({
            "kigo": tgt.word,
            "season": tgt.season,
            "haiku_structure": tgt.haiku_structure,
            "m5_1": tgt.m5_1,
            "m7": tgt.m7,
            "m5_2": tgt.m5_2,
            "ref_haiku": tgt.ref_haiku,
            "repr_haiku": repr_haiku,
            "avg_ppl": avg_ppl,
            "mora_rate": mora_rate,
            "kigo_rate": kigo_rate,
            "bleu": repr_bleu
        })

    pd.DataFrame(stats).to_csv(
        os.path.join(args.output_dir, "fewshot_eval.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(raw_outputs).to_csv(
        os.path.join(args.output_dir, "fewshot_raw_outputs.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("✅ Done. Results saved in output_dir.")

if __name__ == "__main__":
    main()


# haiku_beamsearch_STABLELM_Gemma.py - prova 14
# Haiku generation with StableLM (Gemma) — Beam Search + PPL+BLEU reranking

import os
import re
import string
import math
import time
import argparse
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import LlamaTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--kigo_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("俳句の例:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            cleaned = cleaned.replace(" ", "")
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""

def build_example_block_gemma(ex):
    return (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n\n"
        f"季語: {ex['word']}\n"
        f"季節: {ex['season']}\n"
        f"構造: {ex['haiku_structure']}\n"
        f"俳句:\n{ex['ref_haiku']}\n"
        "<end_of_turn>\n"
    )

def build_prompt_gemma(target, examples):
    prompt_blocks = [build_example_block_gemma(ex) for _, ex in examples.iterrows()]
    target_block = (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n\n"
        f"季語: {target.word}\n"
        f"季節: {target.season}\n"
        f"構造: {target.haiku_structure}\n"
    )
    if target.haiku_structure == "Regular" and all(pd.notnull([target.m5_1, target.m7, target.m5_2])):
        target_block += f"ヒント: {target.m5_1} / {target.m7} / {target.m5_2}\n"
    target_block += "俳句:\n<end_of_turn>\n"
    return "\n".join(prompt_blocks + [target_block])

def retrieve_examples(target, pool, few_shot_k=6):
    pool = pool[pool.season == target.season].copy()
    pool = pool[pool.haiku_id != target.haiku_id]
    same = pool[pool.kigo_id == target.kigo_id]
    if len(same) >= few_shot_k:
        pool = same
    unique_authors = pool['author'].dropna().unique()
    selected = []
    rng = pd.Series(unique_authors).sample(frac=1, random_state=target.name)
    for author in rng:
        samples = pool[pool.author == author]
        if not samples.empty:
            selected.append(samples.sample(1, random_state=target.name))
        if len(selected) == few_shot_k:
            break
    while len(selected) < few_shot_k:
        selected.append(pool.sample(1, random_state=target.name + len(selected)))
    return pd.concat(selected).reset_index(drop=True)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)

    if "<end_of_turn>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
    end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tok))
    model.eval()

    stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl,  lines=True)
    df = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
    df.rename(columns={
        "season_haiku":    "season",
        "haiku":           "ref_haiku",
        "5_mora_segment_1":"m5_1",
        "7_mora_segment":  "m7",
        "5_mora_segment_2":"m5_2",
    }, inplace=True)
    df = df[df["haiku_structure"] == "Regular"].copy()
    df["ref_haiku"] = df[["m5_1","m7","m5_2"]].apply(make_575, axis=1)

    few_shot_k = 6
    num_targets = 20
    max_new_tokens = 20
    num_beams = 6
    num_return_sequences = 5
    num_beam_groups = 3
    diversity_penalty = 0.7

    stats = []
    raw_outputs = []

    for _, tgt in df.sample(num_targets, random_state=42).iterrows():
        exs = retrieve_examples(tgt, df, few_shot_k)
        prompt = build_prompt_gemma(tgt, exs)
        inp = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                early_stopping=True,
                stopping_criteria=stop_criteria
            )

        outputs, ppls, bleus, final_scores = [], [], [], []
        for g in gen:
            raw = tok.decode(g, skip_special_tokens=False).strip()
            h = clean_haiku(raw)
            outputs.append(h)
            ppl = compute_perplexity(model, tok, h, device) if h else float("inf")
            bleu = compute_bleu(tgt.ref_haiku, h) if h else 0.0
            ppls.append(ppl)
            bleus.append(bleu)
            final_scores.append(bleu - 0.2 * math.log(ppl) if math.isfinite(ppl) and ppl > 0 else -float("inf"))
            raw_outputs.append({
                "prompt_kigo":   tgt.word,
                "raw_output":    raw,
                "cleaned_haiku": h
            })

        best_idx = final_scores.index(max(final_scores))
        repr_haiku = outputs[best_idx]
        repr_bleu = bleus[best_idx]
        avg_ppl = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
        mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
        word = tgt.word or ""
        kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)

        stats.append({
            "kigo": tgt.word,
            "season": tgt.season,
            "haiku_structure": tgt.haiku_structure,
            "m5_1": tgt.m5_1,
            "m7": tgt.m7,
            "m5_2": tgt.m5_2,
            "ref_haiku": tgt.ref_haiku,
            "repr_haiku": repr_haiku,
            "avg_ppl": avg_ppl,
            "mora_rate": mora_rate,
            "kigo_rate": kigo_rate,
            "bleu": repr_bleu
        })

    pd.DataFrame(stats).to_csv(
        os.path.join(args.output_dir, "fewshot_eval.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(raw_outputs).to_csv(
        os.path.join(args.output_dir, "fewshot_raw_outputs.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("✅ Done. Results saved in output_dir.")

if __name__ == "__main__":
    main()



# prova 15 (Gemma with prompt optimized by Claude Sonnet4 option 1 optimized)
# MODIFIED BEAM SEARCH for Haiku generation — reranking by perplexity, BLEU evaluation

import os
import re
import string
import math
import time
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Set output directory on Google Drive
drive_output_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/fewshot_results"
os.makedirs(drive_output_dir, exist_ok=True)

# Patch torch.compile if using Python 3.12
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("俳句の例:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            cleaned = cleaned.replace(" ", "") #removing spaces
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

# Manual args setup for Colab
model_dir     = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/tokenizer_model"
tokenizer_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/tokenizer_model"
train_jsonl   = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/train_df_ready.jsonl"
kigo_jsonl    = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/kigo_df_ready.jsonl"

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, local_files_only=True)
if "<end_of_turn>" not in tok.get_vocab():
    tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True
)
model.resize_token_embeddings(len(tok))
model.eval()

stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

# Load and prepare data
df_h = pd.read_json(train_jsonl, lines=True)
df_k = pd.read_json(kigo_jsonl,  lines=True)
df   = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
df.rename(columns={
    "season_haiku":    "season",
    "haiku":           "ref_haiku",
    "5_mora_segment_1":"m5_1",
    "7_mora_segment":  "m7",
    "5_mora_segment_2":"m5_2",
}, inplace=True)

# Filter only Regular haiku
df = df[df["haiku_structure"] == "Regular"].copy()

# Format ref_haiku as 3 lines
def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""
df["ref_haiku"] = df[["m5_1","m7","m5_2"]].apply(make_575, axis=1)

# Gemma‑style few‑shot prompt builders
def build_example_block_gemma(ex):
    """
    Build a single training example block with optimized structure and clearer formatting.
    """
    # More natural Japanese instruction
    block = (
        "季語と季節を使って、美しい俳句を作成してください。\n\n"
        f"季語: {ex['word']}\n"
        f"季節: {ex['season']}\n"
    )

    # Only include structure if it's not Regular (since that's the default expectation)
    if ex.get('haiku_structure') and ex['haiku_structure'] != "Regular":
        block += f"構造: {ex['haiku_structure']}\n"

    block += f"\n俳句:\n{ex['ref_haiku']}\n<end_of_turn>\n"
    return block

def build_prompt_gemma(target, examples):
    """
    Build the complete prompt with optimized example ordering and clearer instructions.
    """
    # Sort examples by quality/relevance if possible
    # Priority: same kigo > same season > same author > similar structure
    sorted_examples = examples.copy()
    if hasattr(target, 'kigo_id') and 'kigo_id' in examples.columns:
        # Prioritize examples with same kigo
        sorted_examples = sorted_examples.sort_values(
            by=['kigo_id', 'season'],
            key=lambda x: x == getattr(target, 'kigo_id', None) if x.name == 'kigo_id' else x,
            ascending=[False, True]
        )

    # Build example blocks
    prompt_blocks = [build_example_block_gemma(ex) for _, ex in sorted_examples.iterrows()]

    # Build target instruction with enhanced clarity
    target_block = (
        "以下の条件に従って、5-7-5の音律を持つ美しい俳句を一つ作成してください。\n\n"
        f"季語: {target.word}\n"
        f"季節: {target.season}\n"
    )

    # Add structure only if non-regular
    if hasattr(target, 'haiku_structure') and target.haiku_structure != "Regular":
        target_block += f"構造: {target.haiku_structure}\n"

    # Enhanced hints for regular haiku with better formatting
    if (hasattr(target, 'haiku_structure') and target.haiku_structure == "Regular" and
        all(pd.notnull([getattr(target, 'm5_1', None),
                       getattr(target, 'm7', None),
                       getattr(target, 'm5_2', None)]))):
        target_block += f"\n参考語句:\n"
        target_block += f"上句(5音): {target.m5_1}\n"
        target_block += f"中句(7音): {target.m7}\n"
        target_block += f"下句(5音): {target.m5_2}\n"

    # Clear, focused final instruction
    target_block += "\n俳句を三行で書いてください:\n<end_of_turn>\n"

    return "\n".join(prompt_blocks + [target_block])


# Beam search & evaluation settings
few_shot_k         = 6
num_targets        = 20
max_new_tokens     = 20
num_beams          = 6
num_return_sequences = 5
num_beam_groups    = 3
diversity_penalty  = 0.7

def retrieve_examples(target):
    pool = df[df.season == target.season].copy()
    pool = pool[pool.haiku_id != target.haiku_id]

    # If many same-kigo haiku exist, use them (as before)
    same = pool[pool.kigo_id == target.kigo_id]
    if len(same) >= few_shot_k:
        pool = same

    # Prioritize diverse authors
    unique_authors = pool['author'].dropna().unique()
    selected = []

    rng = pd.Series(unique_authors).sample(frac=1, random_state=target.name)  # shuffle

    for author in rng:
        samples = pool[pool.author == author]
        if not samples.empty:
            selected.append(samples.sample(1, random_state=target.name))
        if len(selected) == few_shot_k:
            break

    # Fallback if not enough diverse authors
    while len(selected) < few_shot_k:
        selected.append(pool.sample(1, random_state=target.name + len(selected)))

    return pd.concat(selected).reset_index(drop=True)


# Run evaluation
stats = []
raw_outputs = []

for _, tgt in df.sample(num_targets, random_state=42).iterrows():
    exs    = retrieve_examples(tgt)
    prompt = build_prompt_gemma(tgt, exs)
    inp    = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            early_stopping=True,
            stopping_criteria=stop_criteria
        )

    outputs, ppls, bleus, final_scores = [], [], [], []
    for g in gen:
        raw = tok.decode(g, skip_special_tokens=False).strip()
        h   = clean_haiku(raw)

        outputs.append(h)
        ppl  = compute_perplexity(model, tok, h, device) if h else float("inf")
        bleu = compute_bleu(tgt.ref_haiku, h) if h else 0.0

        ppls.append(ppl)
        bleus.append(bleu)
        final_scores.append(
            bleu - 0.2 * math.log(ppl)
            if math.isfinite(ppl) and ppl>0 else -float("inf")
        )

        raw_outputs.append({
            "prompt_kigo":   tgt.word,
            "raw_output":    raw,
            "cleaned_haiku": h
        })

    best_idx   = final_scores.index(max(final_scores))
    repr_haiku = outputs[best_idx]
    repr_bleu  = bleus[best_idx]

    avg_ppl   = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
    mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
    word      = tgt.word or ""
    kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)

    stats.append({
        "kigo":           tgt.word,
        "season":         tgt.season,
        "haiku_structure":tgt.haiku_structure,
        "m5_1":           tgt.m5_1,
        "m7":             tgt.m7,
        "m5_2":           tgt.m5_2,
        "ref_haiku":      tgt.ref_haiku,
        "repr_haiku":     repr_haiku,
        "avg_ppl":        avg_ppl,
        "mora_rate":      mora_rate,
        "kigo_rate":      kigo_rate,
        "bleu":           repr_bleu
    })

# Save results
pd.DataFrame(stats).to_csv(
    os.path.join(drive_output_dir, "fewshot_eval.csv"),
    index=False, encoding="utf-8-sig"
)
pd.DataFrame(raw_outputs).to_csv(
    os.path.join(drive_output_dir, "fewshot_raw_outputs.csv"),
    index=False, encoding="utf-8-sig"
)

print("\n✅ Done. Results saved in fewshot_results on your Drive.")




# prova 16 (Gemma with prompt optimized by Claude Sonnet4 option 2 concise)
# MODIFIED BEAM SEARCH for Haiku generation — reranking by perplexity, BLEU evaluation

import os
import re
import string
import math
import time
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Set output directory on Google Drive
drive_output_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/fewshot_results"
os.makedirs(drive_output_dir, exist_ok=True)

# Patch torch.compile if using Python 3.12
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("俳句の例:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            cleaned = cleaned.replace(" ", "") #removing spaces
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

# Manual args setup for Colab
model_dir     = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/tokenizer_model"
tokenizer_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/tokenizer_model"
train_jsonl   = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/train_df_ready.jsonl"
kigo_jsonl    = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/kigo_df_ready.jsonl"

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, local_files_only=True)
if "<end_of_turn>" not in tok.get_vocab():
    tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True
)
model.resize_token_embeddings(len(tok))
model.eval()

stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

# Load and prepare data
df_h = pd.read_json(train_jsonl, lines=True)
df_k = pd.read_json(kigo_jsonl,  lines=True)
df   = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
df.rename(columns={
    "season_haiku":    "season",
    "haiku":           "ref_haiku",
    "5_mora_segment_1":"m5_1",
    "7_mora_segment":  "m7",
    "5_mora_segment_2":"m5_2",
}, inplace=True)

# Filter only Regular haiku
df = df[df["haiku_structure"] == "Regular"].copy()

# Format ref_haiku as 3 lines
def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""
df["ref_haiku"] = df[["m5_1","m7","m5_2"]].apply(make_575, axis=1)

# Gemma‑style few‑shot prompt builders
def build_example_block_gemma(ex):
    """
    Concise version focusing on essential information only.
    """
    return f"季語: {ex['word']} ({ex['season']})\n俳句:\n{ex['ref_haiku']}\n<end_of_turn>\n"

def build_prompt_gemma(target, examples):
    """
    Concise version that reduces token usage while maintaining effectiveness.
    """
    # Build concise examples
    prompt_blocks = [build_example_block_gemma(ex) for _, ex in examples.iterrows()]

    # Concise target instruction
    target_block = f"季語: {target.word} ({target.season})\n"

    # Add hints if available
    if (hasattr(target, 'haiku_structure') and target.haiku_structure == "Regular" and
        all(pd.notnull([getattr(target, 'm5_1', None),
                       getattr(target, 'm7', None),
                       getattr(target, 'm5_2', None)]))):
        target_block += f"語句: {target.m5_1} / {target.m7} / {target.m5_2}\n"

    target_block += "俳句:\n<end_of_turn>\n"

    return "\n".join(prompt_blocks + [target_block])

# Beam search & evaluation settings
few_shot_k         = 6
num_targets        = 20
max_new_tokens     = 20
num_beams          = 6
num_return_sequences = 5
num_beam_groups    = 3
diversity_penalty  = 0.7

def retrieve_examples(target):
    pool = df[df.season == target.season].copy()
    pool = pool[pool.haiku_id != target.haiku_id]

    # If many same-kigo haiku exist, use them (as before)
    same = pool[pool.kigo_id == target.kigo_id]
    if len(same) >= few_shot_k:
        pool = same

    # Prioritize diverse authors
    unique_authors = pool['author'].dropna().unique()
    selected = []

    rng = pd.Series(unique_authors).sample(frac=1, random_state=target.name)  # shuffle

    for author in rng:
        samples = pool[pool.author == author]
        if not samples.empty:
            selected.append(samples.sample(1, random_state=target.name))
        if len(selected) == few_shot_k:
            break

    # Fallback if not enough diverse authors
    while len(selected) < few_shot_k:
        selected.append(pool.sample(1, random_state=target.name + len(selected)))

    return pd.concat(selected).reset_index(drop=True)


# Run evaluation
stats = []
raw_outputs = []

for _, tgt in df.sample(num_targets, random_state=42).iterrows():
    exs    = retrieve_examples(tgt)
    prompt = build_prompt_gemma(tgt, exs)
    inp    = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            early_stopping=True,
            stopping_criteria=stop_criteria
        )

    outputs, ppls, bleus, final_scores = [], [], [], []
    for g in gen:
        raw = tok.decode(g, skip_special_tokens=False).strip()
        h   = clean_haiku(raw)

        outputs.append(h)
        ppl  = compute_perplexity(model, tok, h, device) if h else float("inf")
        bleu = compute_bleu(tgt.ref_haiku, h) if h else 0.0

        ppls.append(ppl)
        bleus.append(bleu)
        final_scores.append(
            bleu - 0.2 * math.log(ppl)
            if math.isfinite(ppl) and ppl>0 else -float("inf")
        )

        raw_outputs.append({
            "prompt_kigo":   tgt.word,
            "raw_output":    raw,
            "cleaned_haiku": h
        })

    best_idx   = final_scores.index(max(final_scores))
    repr_haiku = outputs[best_idx]
    repr_bleu  = bleus[best_idx]

    avg_ppl   = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
    mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
    word      = tgt.word or ""
    kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)

    stats.append({
        "kigo":           tgt.word,
        "season":         tgt.season,
        "haiku_structure":tgt.haiku_structure,
        "m5_1":           tgt.m5_1,
        "m7":             tgt.m7,
        "m5_2":           tgt.m5_2,
        "ref_haiku":      tgt.ref_haiku,
        "repr_haiku":     repr_haiku,
        "avg_ppl":        avg_ppl,
        "mora_rate":      mora_rate,
        "kigo_rate":      kigo_rate,
        "bleu":           repr_bleu
    })

# Save results
pd.DataFrame(stats).to_csv(
    os.path.join(drive_output_dir, "fewshot_eval.csv"),
    index=False, encoding="utf-8-sig"
)
pd.DataFrame(raw_outputs).to_csv(
    os.path.join(drive_output_dir, "fewshot_raw_outputs.csv"),
    index=False, encoding="utf-8-sig"
)

print("\n✅ Done. Results saved in fewshot_results on your Drive.")



# prova 17 (Gemma with prompt optimized by Claude Sonnet4 option 3 enhanced)
# MODIFIED BEAM SEARCH for Haiku generation — reranking by perplexity, BLEU evaluation

import os
import re
import string
import math
import time
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Set output directory on Google Drive
drive_output_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/fewshot_results"
os.makedirs(drive_output_dir, exist_ok=True)

# Patch torch.compile if using Python 3.12
if hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn
try:
    import torch._dynamo
    torch._dynamo.disable()
except ImportError:
    pass

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("俳句の例:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            cleaned = cleaned.replace(" ", "") #removing spaces
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

# Manual args setup for Colab
model_dir     = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/tokenizer_model"
tokenizer_dir = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/tokenizer_model"
train_jsonl   = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/train_df_ready.jsonl"
kigo_jsonl    = "/content/drive/MyDrive/DATA SCIENCE 23 24/THESIS/haiku_project/kigo_df_ready.jsonl"

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, local_files_only=True)
if "<end_of_turn>" not in tok.get_vocab():
    tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True
)
model.resize_token_embeddings(len(tok))
model.eval()

stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

# Load and prepare data
df_h = pd.read_json(train_jsonl, lines=True)
df_k = pd.read_json(kigo_jsonl,  lines=True)
df   = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
df.rename(columns={
    "season_haiku":    "season",
    "haiku":           "ref_haiku",
    "5_mora_segment_1":"m5_1",
    "7_mora_segment":  "m7",
    "5_mora_segment_2":"m5_2",
}, inplace=True)

# Filter only Regular haiku
df = df[df["haiku_structure"] == "Regular"].copy()

# Format ref_haiku as 3 lines
def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""
df["ref_haiku"] = df[["m5_1","m7","m5_2"]].apply(make_575, axis=1)

# Gemma‑style few‑shot prompt builders
def build_example_block_gemma(ex):
    """
    Enhanced version with better seasonal context and poetic guidance.
    """
    # Season-specific instruction variations
    season_instructions = {
        "春": "春の美しさを表現した俳句を作成してください。",
        "夏": "夏の情感を込めた俳句を作成してください。",
        "秋": "秋の趣を表現した俳句を作成してください。",
        "冬": "冬の静寂と美しさを表現した俳句を作成してください。"
    }

    season = ex.get('season', '春')
    instruction = season_instructions.get(season, "季節感豊かな俳句を作成してください。")

    return (
        f"{instruction}\n"
        f"季語: {ex['word']}\n"
        f"俳句:\n{ex['ref_haiku']}\n<end_of_turn>\n"
    )

def build_prompt_gemma(target, examples):
    """
    Enhanced version with better seasonal context and structure.
    """
    # Group examples by season for better context
    same_season = examples[examples['season'] == target.season]
    other_season = examples[examples['season'] != target.season]

    # Prioritize same season examples
    ordered_examples = pd.concat([same_season, other_season]).head(6)

    prompt_blocks = [build_example_block_gemma(ex) for _, ex in ordered_examples.iterrows()]

    # Enhanced target instruction
    season_contexts = {
        "春": "新緑の季節、生命力あふれる春の俳句を",
        "夏": "暑さと生命力の季節、夏の情景を込めた俳句を",
        "秋": "実りと物悲しさの季節、秋の風情を表現した俳句を",
        "冬": "静寂と清浄の季節、冬の美しさを表現した俳句を"
    }

    season_context = season_contexts.get(target.season, "季節感豊かな俳句を")

    target_block = (
        f"{season_context}5-7-5の音律で作成してください。\n\n"
        f"季語: {target.word}\n"
        f"季節: {target.season}\n"
    )

    # Enhanced hints with poetic guidance
    if (hasattr(target, 'haiku_structure') and target.haiku_structure == "Regular" and
        all(pd.notnull([getattr(target, 'm5_1', None),
                       getattr(target, 'm7', None),
                       getattr(target, 'm5_2', None)]))):
        target_block += (
            f"\n表現の手がかり:\n"
            f"• {target.m5_1} (5音)\n"
            f"• {target.m7} (7音)\n"
            f"• {target.m5_2} (5音)\n"
        )

    target_block += "\n俳句:\n<end_of_turn>\n"

    return "\n".join(prompt_blocks + [target_block])

# Beam search & evaluation settings
few_shot_k         = 6
num_targets        = 20
max_new_tokens     = 20
num_beams          = 6
num_return_sequences = 5
num_beam_groups    = 3
diversity_penalty  = 0.7

def retrieve_examples(target):
    pool = df[df.season == target.season].copy()
    pool = pool[pool.haiku_id != target.haiku_id]

    # If many same-kigo haiku exist, use them (as before)
    same = pool[pool.kigo_id == target.kigo_id]
    if len(same) >= few_shot_k:
        pool = same

    # Prioritize diverse authors
    unique_authors = pool['author'].dropna().unique()
    selected = []

    rng = pd.Series(unique_authors).sample(frac=1, random_state=target.name)  # shuffle

    for author in rng:
        samples = pool[pool.author == author]
        if not samples.empty:
            selected.append(samples.sample(1, random_state=target.name))
        if len(selected) == few_shot_k:
            break

    # Fallback if not enough diverse authors
    while len(selected) < few_shot_k:
        selected.append(pool.sample(1, random_state=target.name + len(selected)))

    return pd.concat(selected).reset_index(drop=True)


# Run evaluation
stats = []
raw_outputs = []

for _, tgt in df.sample(num_targets, random_state=42).iterrows():
    exs    = retrieve_examples(tgt)
    prompt = build_prompt_gemma(tgt, exs)
    inp    = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            early_stopping=True,
            stopping_criteria=stop_criteria
        )

    outputs, ppls, bleus, final_scores = [], [], [], []
    for g in gen:
        raw = tok.decode(g, skip_special_tokens=False).strip()
        h   = clean_haiku(raw)

        outputs.append(h)
        ppl  = compute_perplexity(model, tok, h, device) if h else float("inf")
        bleu = compute_bleu(tgt.ref_haiku, h) if h else 0.0

        ppls.append(ppl)
        bleus.append(bleu)
        final_scores.append(
            bleu - 0.2 * math.log(ppl)
            if math.isfinite(ppl) and ppl>0 else -float("inf")
        )

        raw_outputs.append({
            "prompt_kigo":   tgt.word,
            "raw_output":    raw,
            "cleaned_haiku": h
        })

    best_idx   = final_scores.index(max(final_scores))
    repr_haiku = outputs[best_idx]
    repr_bleu  = bleus[best_idx]

    avg_ppl   = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
    mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
    word      = tgt.word or ""
    kigo_rate = sum(1 for h in outputs if word in (h or "")) / len(outputs)

    stats.append({
        "kigo":           tgt.word,
        "season":         tgt.season,
        "haiku_structure":tgt.haiku_structure,
        "m5_1":           tgt.m5_1,
        "m7":             tgt.m7,
        "m5_2":           tgt.m5_2,
        "ref_haiku":      tgt.ref_haiku,
        "repr_haiku":     repr_haiku,
        "avg_ppl":        avg_ppl,
        "mora_rate":      mora_rate,
        "kigo_rate":      kigo_rate,
        "bleu":           repr_bleu
    })

# Save results
pd.DataFrame(stats).to_csv(
    os.path.join(drive_output_dir, "fewshot_eval.csv"),
    index=False, encoding="utf-8-sig"
)
pd.DataFrame(raw_outputs).to_csv(
    os.path.join(drive_output_dir, "fewshot_raw_outputs.csv"),
    index=False, encoding="utf-8-sig"
)

print("\n✅ Done. Results saved in fewshot_results on your Drive.")



# prova 11 + PromptOptimizer = prova_18
# MODIFIED BEAM SEARCH for Haiku generation with automatic prompt optimization using a custom scoring loop

import os
import re
import string
import math
import argparse
import torch
import pandas as pd
import pyopenjtalk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# === CLI ARGUMENTS ===
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--kigo_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()

class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]

def clean_haiku(raw: str) -> str:
    haiku_started = False
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if "<end_of_turn>" in line:
            break
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        if line.startswith("俳句:") or line.startswith("俳句の例:"):
            haiku_started = True
            continue
        if haiku_started:
            cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line).strip()
            cleaned = cleaned.replace(" ", "")
            if cleaned:
                haiku_lines.append(cleaned)
            if len(haiku_lines) == 3:
                break
    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""

def build_prompt_with_template(template, target, examples):
    blocks = [template.format(kigo=ex['word'], season=ex['season']) + f"\n俳句:\n{ex['ref_haiku']}\n<end_of_turn>" for _, ex in examples.iterrows()]
    tgt_block = template.format(kigo=target.word, season=target.season)
    if target.haiku_structure == "Regular" and all(pd.notnull([target.m5_1, target.m7, target.m5_2])):
        tgt_block += f"\nヒント: {target.m5_1} / {target.m7} / {target.m5_2}"
    tgt_block += "\n俳句:\n<end_of_turn>"
    return "\n\n".join(blocks + [tgt_block])

def retrieve_examples(target, df, few_shot_k=6):
    pool = df[df.season == target.season].copy()
    pool = pool[pool.haiku_id != target.haiku_id]
    same = pool[pool.kigo_id == target.kigo_id]
    if len(same) >= few_shot_k:
        pool = same
    unique_authors = pool['author'].dropna().unique()
    selected = []
    rng = pd.Series(unique_authors).sample(frac=1, random_state=target.name)
    for author in rng:
        samples = pool[pool.author == author]
        if not samples.empty:
            selected.append(samples.sample(1, random_state=target.name))
        if len(selected) == few_shot_k:
            break
    while len(selected) < few_shot_k:
        selected.append(pool.sample(1, random_state=target.name + len(selected)))
    return pd.concat(selected).reset_index(drop=True)

def score_fn(model, tok, prompt, response, ref, device):
    gen = clean_haiku(response)
    bleu = compute_bleu(ref, gen)
    ppl = compute_perplexity(model, tok, gen, device)
    return bleu - 0.2 * math.log(ppl) if ppl > 0 and math.isfinite(ppl) else -float("inf")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    if "<end_of_turn>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
    end_token_id = tok.convert_tokens_to_ids("<end_of_turn>")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tok))
    model.eval()

    stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl,  lines=True)
    df = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
    df.rename(columns={
        "season_haiku":    "season",
        "haiku":           "ref_haiku",
        "5_mora_segment_1":"m5_1",
        "7_mora_segment":  "m7",
        "5_mora_segment_2":"m5_2",
    }, inplace=True)
    df = df[df["haiku_structure"] == "Regular"].copy()
    df["ref_haiku"] = df[["m5_1","m7","m5_2"]].apply(make_575, axis=1)

    PROMPT_POOL = [
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。\n季語: {kigo}\n季節: {season}",
        "次のキーワードを使って、自然を表す三行の俳句を作成してください。\n季語: {kigo}\n季節: {season}",
        "日本語の俳句を作ってください。\n季語: {kigo}\n季節: {season}",
        "次の季語と季節を含めて、三行の俳句を作成してください。\n季語: {kigo}\n季節: {season}"
    ]

    train_subset = df.sample(8, random_state=123)
    prompt_scores = {}
    for template in PROMPT_POOL:
        scores = []
        for _, row in train_subset.iterrows():
            exs = retrieve_examples(row, df)
            prompt = build_prompt_with_template(template, row, exs)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=20, num_beams=4, early_stopping=True)
            resp = tok.decode(output[0], skip_special_tokens=False)
            score = score_fn(model, tok, prompt, resp, row.ref_haiku, device)
            scores.append(score)
        prompt_scores[template] = sum(scores) / len(scores)

    best_prompt_template = max(prompt_scores, key=prompt_scores.get)

    stats, raw_outputs = [], []
    num_targets = 20

    for _, tgt in df.sample(num_targets, random_state=42).iterrows():
        exs = retrieve_examples(tgt, df)
        prompt = build_prompt_with_template(best_prompt_template, tgt, exs)
        inp = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen = model.generate(
                **inp,
                max_new_tokens=20,
                num_beams=6,
                num_return_sequences=5,
                num_beam_groups=3,
                diversity_penalty=0.7,
                early_stopping=True,
                stopping_criteria=stop_criteria
            )

        outputs, ppls, bleus, final_scores = [], [], [], []
        for g in gen:
            raw = tok.decode(g, skip_special_tokens=False).strip()
            h = clean_haiku(raw)
            outputs.append(h)
            ppl = compute_perplexity(model, tok, h, device) if h else float("inf")
            bleu = compute_bleu(tgt.ref_haiku, h) if h else 0.0
            ppls.append(ppl)
            bleus.append(bleu)
            final_scores.append(bleu - 0.2 * math.log(ppl) if math.isfinite(ppl) and ppl > 0 else -float("inf"))

            raw_outputs.append({
                "prompt_kigo": tgt.word,
                "raw_output": raw,
                "cleaned_haiku": h
            })

        best_idx = final_scores.index(max(final_scores))
        repr_haiku = outputs[best_idx]
        repr_bleu = bleus[best_idx]
        avg_ppl = sum(p for p in ppls if math.isfinite(p)) / len(ppls)
        mora_rate = sum(is_575(h) for h in outputs if h) / len(outputs)
        kigo_rate = sum(1 for h in outputs if tgt.word in (h or "")) / len(outputs)

        stats.append({
            "kigo": tgt.word,
            "season": tgt.season,
            "haiku_structure": tgt.haiku_structure,
            "m5_1": tgt.m5_1,
            "m7": tgt.m7,
            "m5_2": tgt.m5_2,
            "ref_haiku": tgt.ref_haiku,
            "repr_haiku": repr_haiku,
            "avg_ppl": avg_ppl,
            "mora_rate": mora_rate,
            "kigo_rate": kigo_rate,
            "bleu": repr_bleu
        })

    pd.DataFrame(stats).to_csv(os.path.join(args.output_dir, "fewshot_eval.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(raw_outputs).to_csv(os.path.join(args.output_dir, "fewshot_raw_outputs.csv"), index=False, encoding="utf-8-sig")
    print("\u2705 Done. Results saved in output_dir.")

if __name__ == "__main__":
    main()