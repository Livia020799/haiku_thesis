# haiku_beamsearch_STABLELM_Gamma.py (iterative 3-step generation)
# Haiku generation with StableLM (Gemma) — Iterative Line-by-Line Beam Search + PPL+BLEU reranking
#iterative_eval_1

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
    return "\n".join(prompt_blocks)


def retrieve_examples(target, pool, few_shot_k=6):
    # Step 1: try exact match of season, kigo_id, author
    pool = pool[pool["haiku_id"] != target["haiku_id"]]
    subset = pool[
        (pool['season']  == target['season'])  &
        (pool['kigo_id'] == target['kigo_id']) &
        (pool['author']  == target['author'])
    ]

    # Step 2: if too few, drop author constraint
    if len(subset) < few_shot_k:
        subset = pool[
            (pool['season']  == target['season']) &
            (pool['kigo_id'] == target['kigo_id'])
        ]

    # Step 3: if still too few, drop kigo constraint
    if len(subset) < few_shot_k:
        subset = pool[pool['season'] == target['season']]

    # Now sample one per unique author in that subset
    selected = []
    for auth in subset['author'].unique():
        auth_samples = subset[subset['author'] == auth]
        # pick one from this author (seeded)
        selected.append(auth_samples.sample(1, random_state=target.name))
        if len(selected) == few_shot_k:
            break

    # If you still have fewer than k, fill in with whatever remains
    while len(selected) < few_shot_k and not subset.empty:
        # seed offset ensures reproducibility but different picks
        selected.append(subset.sample(1, random_state=target.name + len(selected)))

    return pd.concat(selected).reset_index(drop=True)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

        # Setup model
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

    # Load and merge data
    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl, lines=True)
    df = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
    df.rename(columns={
        "season_haiku": "season",
        "haiku": "ref_haiku",
        "5_mora_segment_1": "m5_1",
        "7_mora_segment": "m7",
        "5_mora_segment_2": "m5_2",
    }, inplace=True)
    df = df[df['haiku_structure'] == "Regular"].copy()
    df["ref_haiku"] = df[["m5_1","m7","m5_2"]].apply(make_575, axis=1)

    # Prefilter to master poets for few-shot examples
    #masters = ["松尾芭蕉", "与謝蕪村", "小林一茶", "正岡子規"]
    #df_masters = df[df['author'].isin(masters)].reset_index(drop=True)
    df_masters = df.copy()

    # Generation settings
    few_shot_k = 6
    num_targets = 20
    num_beams = 6
    num_beam_groups = 3
    diversity_penalty = 0.7

    stats = []
    raw_outputs = []

    # Iterate targets from full df, but examples from df_masters
    for _, tgt in df.sample(num_targets, random_state=42).iterrows():
        exs = retrieve_examples(tgt, df_masters, few_shot_k)

        header = (
            f"作者: {tgt['author']}\n"
            f"季語: {tgt['word']} (ID: {tgt['kigo_id']})\n"
            f"季節: {tgt['season']}\n\n"
        )
        base_prompt = header + build_prompt_gemma(tgt, exs)

        # Generate first line (5 mora)
        inp1 = tok(base_prompt + "\n俳句:\n<end_of_turn>\n", return_tensors="pt").to(device)
        gen1 = model.generate(
            **inp1,
            max_new_tokens=20,
            num_beams=num_beams,
            early_stopping=True,
            eos_token_id=end_token_id,
            stopping_criteria=stop_criteria
        )
        line1 = clean_haiku(tok.decode(gen1[0], skip_special_tokens=False))

        # Generate second line (7 mora)
        prompt2 = base_prompt + f"\n前の行: {line1}\n2行目(7モーラ):\n<end_of_turn>\n"
        gen2 = model.generate(
            **tok(prompt2, return_tensors="pt").to(device),
            max_new_tokens=20,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            num_return_sequences=5,
            early_stopping=True,
            eos_token_id=end_token_id,
            stopping_criteria=stop_criteria
        )
        lines2 = [clean_haiku(tok.decode(g, skip_special_tokens=False)) for g in gen2]

        # Full haiku rerank
        best_score = -float('inf')
        best_haiku = None
        for l2 in lines2:
            prompt3 = base_prompt + f"\n前の行: {l2}\n3行目(5モーラ):\n<end_of_turn>\n"
            gen3 = model.generate(
                **tok(prompt3, return_tensors="pt").to(device),
                max_new_tokens=20,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                num_return_sequences=5,
                early_stopping=True,
                eos_token_id=end_token_id,
                stopping_criteria=stop_criteria
            )
            lines3 = [clean_haiku(tok.decode(g, skip_special_tokens=False)) for g in gen3]
            for l3 in lines3:
                full = f"{line1}\n{l2}\n{l3}"
                parts = full.split("\n")[:3]
                candidate = "\n".join(parts)
                score = compute_bleu(tgt['ref_haiku'], candidate) - 0.2 * math.log(compute_perplexity(model, tok, candidate, device))
                if score > best_score:
                    best_score = score
                    best_haiku = candidate

        # Record
        # After you’ve found best_haiku, compute the extra metrics:
        avg_ppl = compute_perplexity(model, tok, best_haiku, device)
        bleu_score = compute_bleu(tgt['ref_haiku'], best_haiku)
        mora_rate = 1.0 if is_575(best_haiku) else 0.0
        # does the kigo word actually appear?
        kigo_rate = 1.0 if tgt['word'] in best_haiku else 0.0

        stats.append({
            "kigo":            tgt['word'],
            "season":          tgt['season'],
            "haiku_structure": tgt['haiku_structure'],
            "m5_1":            tgt['m5_1'],
            "m7":              tgt['m7'],
            "m5_2":            tgt['m5_2'],
            "ref_haiku":       tgt['ref_haiku'],
            "repr_haiku":      best_haiku,
            "avg_ppl":         avg_ppl,
            "mora_rate":       mora_rate,
            "kigo_rate":       kigo_rate,
            "bleu":            bleu_score,
        })
        raw_outputs.append({"prompt_kigo": tgt['word'], "cleaned_haiku": best_haiku})

    # Save
    pd.DataFrame(stats).to_csv(os.path.join(args.output_dir, "iterative_eval.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(raw_outputs).to_csv(os.path.join(args.output_dir, "iterative_raw_outputs.csv"), index=False, encoding="utf-8-sig")
    print("Done. Iterative results saved.")

if __name__ == "__main__":
    main()




#iterative_eval_mora

# haiku_beamsearch_STABLELM_Gamma_mora_constrained_debug.py
# Iterative 3-step Haiku generation with Mora constraints, cleaning, and debug logging

import os
import re
import math
import argparse
import string
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

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--kigo_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()

# Stop generation on our custom end token
class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

# Count mora by vowel + 'N' occurrences
def count_mora(jp_text: str) -> int:
    phonemes = pyopenjtalk.g2p(jp_text)
    mora = sum(1 for c in phonemes if c in "aeiouN")
    print(f"DEBUG count_mora: '{jp_text}' -> phonemes={phonemes} -> mora={mora}")
    return mora

# Check full haiku structure 5-7-5
def is_575(haiku: str) -> bool:
    lines = haiku.split("\n")
    if len(lines) != 3:
        return False
    return [count_mora(l) for l in lines] == [5, 7, 5]

# Clean an individual haiku line with full-haiku logic per line
def clean_line(line: str) -> str:
    raw = line.strip()
    # skip markers
    if not raw or '<end_of_turn>' in raw or raw.startswith('#') or raw.startswith('*') or 'ヒント' in raw:
        print(f"DEBUG clean_line skip: '{raw}' -> ''")
        return ''
    # remove any preceding label up to colon
    if ':' in raw:
        raw = raw.split(':', 1)[1].strip()
    print(f"DEBUG clean_line raw after label strip: '{raw}'")
    # remove ascii, digits, punctuation
    cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", raw)
    cleaned = cleaned.replace(' ', '').strip()
    print(f"DEBUG clean_line cleaned: '{cleaned}'")
    return cleaned

# Clean a full raw haiku text (multi-line)
def clean_haiku(raw: str) -> str:
    print(f"DEBUG clean_haiku raw block:\n{raw}\n--- end raw")
    haiku_started = False
    lines = []
    for row in raw.splitlines():
        row = row.strip()
        if '<end_of_turn>' in row:
            break
        if not row or row.startswith('#') or row.startswith('*') or 'ヒント' in row:
            continue
        if row.startswith('俳句:') or row.startswith('俳句の例:'):
            haiku_started = True
            continue
        if haiku_started:
            cl = clean_line(row)
            if cl:
                lines.append(cl)
            if len(lines) == 3:
                break
    while len(lines) < 3:
        lines.append("")
    haiku = '\n'.join(lines)
    print(f"DEBUG clean_haiku result:\n{haiku}\n--- end clean")
    return haiku

# Reconstruct reference haiku from mora segments
def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""

# Compute perplexity
def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

# Compute unigram BLEU
def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

# Extract generated line from model output
def extract_generated(raw_ids: torch.LongTensor, prompt_len: int, tok) -> str:
    new_ids = raw_ids[prompt_len:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    # only keep after any colon
    if ':' in text:
        text = text.split(':',1)[1]
    text = text.replace('<end_of_turn>','').strip()
    # first line only
    first_line = text.split('\n')[0]
    return clean_line(first_line)

# Build few-shot prompt block
def build_example_block(ex):
    return (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n\n"
        f"季語: {ex['word']}\n"
        f"季節: {ex['season']}\n"
        f"構造: {ex['haiku_structure']}\n"
        f"俳句:\n{ex['ref_haiku']}\n"
        "<end_of_turn>\n"
    )

# Retrieve few-shot examples
def retrieve_examples(target, pool, k=6):
    pool = pool[pool['haiku_id'] != target['haiku_id']]
    subset = pool[(pool['season']==target['season']) & (pool['kigo_id']==target['kigo_id']) & (pool['author']==target['author'])]
    if len(subset) < k:
        subset = pool[(pool['season']==target['season']) & (pool['kigo_id']==target['kigo_id'])]
    if len(subset) < k:
        subset = pool[pool['season']==target['season']]
    selected = []
    for auth in subset['author'].unique():
        auth_samples = subset[subset['author']==auth]
        selected.append(auth_samples.sample(1, random_state=target.name))
        if len(selected) == k:
            break
    while len(selected) < k and not subset.empty:
        selected.append(subset.sample(1, random_state=target.name+len(selected)))
    return pd.concat(selected).reset_index(drop=True)

# Main
if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    if '<end_of_turn>' not in tok.get_vocab(): tok.add_special_tokens({'additional_special_tokens':['<end_of_turn>']})
    end_id = tok.convert_tokens_to_ids('<end_of_turn>')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map='auto',
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.resize_token_embeddings(len(tok))
    model.eval()
    stopper = StoppingCriteriaList([StopOnEndOfTurn(end_id)])

    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl, lines=True)
    df = df_h.merge(df_k, on='haiku_id', how='left', suffixes=('_haiku','_kigo'))
    df.rename(columns={
        'season_haiku':'season',
        'haiku':'ref_haiku',
        '5_mora_segment_1':'m5_1',
        '7_mora_segment':'m7',
        '5_mora_segment_2':'m5_2'
    }, inplace=True)
    df = df[df['haiku_structure']=='Regular'].copy()
    df['ref_haiku'] = df[['m5_1','m7','m5_2']].apply(make_575, axis=1)
    pool = df.copy()

    beams, ret = 20, 20  # ensure return <= beams
    stats, raw_outputs = [], []

    for _, tgt in df.sample(20, random_state=42).iterrows():
        exs = retrieve_examples(tgt, pool)
        header = f"作者: {tgt['author']}\n季語: {tgt['word']}\n季節: {tgt['season']}\n\n"
        prompt_base = header + ''.join(build_example_block(ex) for _, ex in exs.iterrows())

        lines = []
        for label, max_m in [('俳句:',5),('2行目(7モーラ):',7),('3行目(5モーラ):',5)]:
            p = prompt_base + (f"前の行: {lines[-1]}\n" if lines else '') + f"{label}\n<end_of_turn>\n"
            enc = tok(p, return_tensors='pt').to(device)
            outs = model.generate(**enc, max_new_tokens=16, num_beams=beams, num_return_sequences=ret, stopping_criteria=stopper)
            cands = [extract_generated(o, enc['input_ids'].shape[1], tok) for o in outs]
            valid = [c for c in cands if count_mora(c)==max_m]
            choice = valid[0] if valid else cands[0]
            lines.append(choice)

        haiku_out = '\n'.join(lines)
        stats.append({
            'kigo':tgt['word'], 'season':tgt['season'], 'haiku_structure':tgt['haiku_structure'],
            'm5_1':tgt['m5_1'], 'm7':tgt['m7'], 'm5_2':tgt['m5_2'], 'ref_haiku':tgt['ref_haiku'],
            'repr_haiku':haiku_out, 'avg_ppl':compute_perplexity(model,tok,haiku_out,device),
            'mora_rate':1.0 if is_575(haiku_out) else 0.0, 'kigo_rate':1.0 if tgt['word'] in haiku_out else 0.0,
            'bleu':compute_bleu(tgt['ref_haiku'], haiku_out)
        })
        raw_outputs.append({'prompt_kigo':tgt['word'], 'cleaned_haiku':haiku_out})

    pd.DataFrame(stats).to_csv(os.path.join(args.output_dir,'iterative_eval_mora.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(raw_outputs).to_csv(os.path.join(args.output_dir,'iterative_raw_mora.csv'), index=False, encoding='utf-8-sig')
    print("Done. Mora-constrained debug results saved.")


#iterative_eval_mora_2

# Iterative 3-step Haiku generation with Mora constraints, per-line cleaning, example retrieval, and debug logging
# File: haiku_beamsearch_STABLELM_Gamma_mora_constrained_debug.py

import os
import re
import math
import argparse
import string
import random
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

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--kigo_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()

# Stop generation on our custom end token
class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.end_token_id

# Count mora by vowel + 'N' occurrences
def count_mora(jp_text: str) -> int:
    phonemes = pyopenjtalk.g2p(jp_text)
    mora = sum(1 for c in phonemes if c in "aeiouN")
    print(f"DEBUG count_mora: '{jp_text}' -> phonemes={phonemes} -> mora={mora}")
    return mora

# Clean an individual haiku line (single-line clean-up)
def clean_line(raw: str) -> str:
    """
    Strip metadata, ascii letters, digits, punctuation, spaces,
    empty/comment/hint lines, and any leading labels.
    Returns the "pure" Japanese text (or empty if discarded).
    """
    line = raw.strip()

    # 0) drop any remaining copies of our few-shot prompt header
    if "以下の情報をもとに" in line:
        print(f"DEBUG clean_line skip (prompt header): '{line}' -> ''")
        return ""

    # 1) stop if end-of-turn marker shows up
    if "<end_of_turn>" in line:
        print(f"DEBUG clean_line skip (end marker): '{line}' -> ''")
        return ""

    # 2) drop blank or comment/hint lines
    if not line or line.startswith(("#", "*")) or "ヒント" in line:
        print(f"DEBUG clean_line skip (blank/comment/hint): '{line}' -> ''")
        return ""

    # 3) drop any leading labels up to the first colon (ASCII or full-width)
    line = re.sub(r"^.*?[:：]\s*", "", line)
    print(f"DEBUG clean_line after label strip: '{line}'")

    # 4) remove ALL ASCII letters, digits, punctuation, and spaces
    cleaned = re.sub(
        r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+",
        "",
        line
    ).replace(" ", "").strip()
    print(f"DEBUG clean_line cleaned: '{cleaned}'")
    return cleaned

# Clean a full raw haiku text (multi-line, legacy support)
def clean_haiku(raw: str) -> str:
    """
    Legacy multi-line cleaner: logs raw input and applies clean_line to extract
    three Japanese lines while ignoring labels and metadata.
    """
    print(f"DEBUG clean_haiku raw block:\n{raw}\n--- end raw")
    started = False
    lines = []
    for row in raw.splitlines():
        row = row.strip()
        if '<end_of_turn>' in row:
            break
        if not row or row.startswith('#') or row.startswith('*') or 'ヒント' in row:
            continue
        if row.startswith('俳句:') or row.startswith('俳句の例:'):
            started = True
            continue
        if started:
            cl = clean_line(row)
            if cl:
                lines.append(cl)
            if len(lines) == 3:
                break
    while len(lines) < 3:
        lines.append("")
    haiku = '\n'.join(lines)
    print(f"DEBUG clean_haiku result:\n{haiku}\n--- end clean")
    return haiku

# Reconstruct reference haiku from mora segments
def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""

# Compute perplexity
def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

# Compute unigram BLEU
def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

# Extract generated line from model output
def extract_generated(raw_ids: torch.LongTensor, prompt_len: int, tok) -> str:
    new_ids = raw_ids[prompt_len:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    # drop any leading label
    if ':' in text:
        text = text.split(':', 1)[1]
    text = text.replace('<end_of_turn>', '').strip()
    first_line = text.split('\n')[0]
    return clean_line(first_line)

# Build few-shot prompt block
def build_example_block(ex):
    return (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。季語と季節を含めてください。\n\n"
        f"季語: {ex['word']}\n"
        f"季節: {ex['season']}\n"
        f"構造: {ex['haiku_structure']}\n"
        f"俳句:\n{ex['ref_haiku']}\n"
        "<end_of_turn>\n"
    )

# Retrieve few-shot examples
def retrieve_examples(target, pool, k=6):
    pool = pool[pool['haiku_id'] != target['haiku_id']]
    subset = pool[(pool['season']==target['season']) & (pool['kigo_id']==target['kigo_id']) & (pool['author']==target['author'])]
    if len(subset) < k:
        subset = pool[(pool['season']==target['season']) & (pool['kigo_id']==target['kigo_id'])]
    if len(subset) < k:
        subset = pool[pool['season']==target['season']]
    selected = []
    for auth in subset['author'].unique():
        auth_samples = subset[subset['author']==auth]
        selected.append(auth_samples.sample(1, random_state=target.name))
        if len(selected) == k:
            break
    while len(selected) < k and not subset.empty:
        selected.append(subset.sample(1, random_state=target.name+len(selected)))
    return pd.concat(selected).reset_index(drop=True)

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    if '<end_of_turn>' not in tok.get_vocab():
        tok.add_special_tokens({'additional_special_tokens':['<end_of_turn>']})
    end_id = tok.convert_tokens_to_ids('<end_of_turn>')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map='auto',
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.resize_token_embeddings(len(tok))
    model.eval()
    stopper = StoppingCriteriaList([StopOnEndOfTurn(end_id)])

    # load data
    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl, lines=True)
    df = df_h.merge(df_k, on='haiku_id', how='left', suffixes=('_haiku','_kigo'))
    df.rename(columns={
        'season_haiku':'season',
        'haiku':'ref_haiku',
        '5_mora_segment_1':'m5_1',
        '7_mora_segment':'m7',
        '5_mora_segment_2':'m5_2'
    }, inplace=True)
    df = df[df['haiku_structure']=='Regular'].copy()
    df['ref_haiku'] = df[['m5_1','m7','m5_2']].apply(make_575, axis=1)
    pool = df.copy()

    beams, ret = 20, 20
    stats, raw_outputs = [], []

    for _, tgt in df.sample(20, random_state=42).iterrows():
        exs = retrieve_examples(tgt, pool)
        header = f"作者: {tgt['author']}\n季語: {tgt['word']}\n季節: {tgt['season']}\n\n"
        prompt_base = header + ''.join(build_example_block(ex) for _, ex in exs.iterrows())

        lines = []
        for label, max_m in [('俳句:',5),('2行目(7モーラ):',7),('3行目(5モーラ):',5)]:
            p = (
                prompt_base +
                (f"前の行: {lines[-1]}\n" if lines else '') +
                f"{label}\n<end_of_turn>\n"
            )
            enc = tok(p, return_tensors='pt').to(device)
            outs = model.generate(
                **enc,
                max_new_tokens=16,
                num_beams=beams,
                num_return_sequences=ret,
                stopping_criteria=stopper
            )
            cands = [extract_generated(o, enc['input_ids'].shape[1], tok) for o in outs]
            valid = [c for c in cands if count_mora(c) == max_m]
            choice = valid[0] if valid else cands[0]
            lines.append(choice)

        haiku_out = '\n'.join(lines)
        stats.append({
            'kigo': tgt['word'],
            'season': tgt['season'],
            'haiku_structure': tgt['haiku_structure'],
            'm5_1': tgt['m5_1'], 'm7': tgt['m7'], 'm5_2': tgt['m5_2'],
            'ref_haiku': tgt['ref_haiku'],
            'repr_haiku': haiku_out,
            'avg_ppl': compute_perplexity(model, tok, haiku_out, device),
            'mora_rate': 1.0 if count_mora(haiku_out.split('\n')[0]) == 5 and \
                          count_mora(haiku_out.split('\n')[1]) == 7 and \
                          count_mora(haiku_out.split('\n')[2]) == 5 else 0.0,
            'kigo_rate': 1.0 if tgt['word'] in haiku_out else 0.0,
            'bleu': compute_bleu(tgt['ref_haiku'], haiku_out)
        })
        raw_outputs.append({'prompt_kigo': tgt['word'], 'cleaned_haiku': haiku_out})

    pd.DataFrame(stats).to_csv(
        os.path.join(args.output_dir, 'iterative_eval_mora.csv'),
        index=False, encoding='utf-8-sig'
    )
    pd.DataFrame(raw_outputs).to_csv(
        os.path.join(args.output_dir, 'iterative_raw_mora.csv'),
        index=False, encoding='utf-8-sig'
    )
    print("Done. Mora-constrained debug results saved.")

    


# haiku_beamsearch_STABLELM_Gamma.py (iterative 3-step generation)
# Haiku generation with StableLM (Gamma) — Iterative Line-by-Line Beam Search + PPL+BLEU reranking
# Revised: use <example_end> delimiter, enforce 5-7-5, avoid premature EOS, decode only new tokens

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
    m = re.search(r"俳句:(.*)", raw, re.S)
    block = m.group(1) if m else raw
    block = block.split("<example_end>")[0]

    lines = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", line)
        cleaned = cleaned.replace(" ", "")
        if cleaned:
            lines.append(cleaned)
        if len(lines) == 3:
            break

    while len(lines) < 3:
        lines.append("")

    return "\n".join(lines)


def compute_perplexity(model, tok, text, device):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())


def compute_bleu(ref, hyp):
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    return sentence_bleu(
        ref_chars, hyp_chars,
        smoothing_function=SmoothingFunction().method1,
        weights=(1.0,),
    )


def make_575(segments):
    if all(pd.notnull(segments)):
        return f"{segments[0]}\n{segments[1]}\n{segments[2]}"
    return ""


def build_example_block_gemma(ex):
    return (
        "以下の情報をもとに俳句を作ってください。季語と季節を含めてください。\n"
        f"季語: {ex['word']}\n"
        f"季節: {ex['season']}\n"
        f"構造: {ex['haiku_structure']}\n"
        f"俳句:\n{ex['ref_haiku']}\n"
        "<example_end>\n"
    )


def build_prompt_gemma(target, examples):
    # Scaffold
    prompt = (
        "以下の情報をもとに、美しい日本語の俳句を三行で一つ作ってください。\n"
        "必ず 5-7-5 のモーラ構造で書いてください。\n\n"
        f"季語: {target['word']}\n"
        f"季節: {target['season']}\n"
    )

    # Hint for regular structure
    if (
        target['haiku_structure'] == "Regular"
        and pd.notnull(target['m5_1'])
        and pd.notnull(target['m7'])
        and pd.notnull(target['m5_2'])
    ):
        prompt += f"ヒント: {target['m5_1']} / {target['m7']} / {target['m5_2']}\n\n"
    else:
        prompt += "\n"

    # Few-shot examples
    for _, ex in examples.iterrows():
        prompt += build_example_block_gemma(ex)

    # Final generation instruction
    prompt += "俳句:\n<end_of_turn>\n"
    return prompt


def retrieve_examples(target, pool, few_shot_k=6):
    pool = pool[pool['haiku_id'] != target['haiku_id']]
    subset = pool[
        (pool['season'] == target['season']) &
        (pool['kigo_id'] == target['kigo_id']) &
        (pool['author'] == target['author'])
    ]

    if len(subset) < few_shot_k:
        subset = pool[
            (pool['season'] == target['season']) & (pool['kigo_id'] == target['kigo_id'])
        ]

    if len(subset) < few_shot_k:
        subset = pool[pool['season'] == target['season']]

    selected = []
    for auth in subset['author'].unique():
        sel = subset[subset['author'] == auth]
        selected.append(sel.sample(1, random_state=target.name))
        if len(selected) == few_shot_k:
            break

    while len(selected) < few_shot_k and not subset.empty:
        selected.append(subset.sample(1, random_state=target.name + len(selected)))

    return pd.concat(selected).reset_index(drop=True)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device & tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        use_fast=False,
        local_files_only=True
    )

    if '<end_of_turn>' not in tok.get_vocab():
        tok.add_special_tokens({'additional_special_tokens': ['<end_of_turn>']})

    end_token_id = tok.convert_tokens_to_ids('<end_of_turn>')

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map='auto',
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        local_files_only=True
    )

    model.resize_token_embeddings(len(tok))
    model.eval()
    stop_criteria = StoppingCriteriaList([StopOnEndOfTurn(end_token_id)])

    # Load data
    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl, lines=True)
    df = df_h.merge(
        df_k,
        on='haiku_id',
        how='left',
        suffixes=('_haiku','_kigo')
    )

    df.rename(
        columns={
            'season_haiku': 'season',
            'haiku': 'ref_haiku',
            '5_mora_segment_1': 'm5_1',
            '7_mora_segment': 'm7',
            '5_mora_segment_2': 'm5_2'
        },
        inplace=True
    )

    df = df[df['haiku_structure'] == 'Regular'].copy()
    df['ref_haiku'] = df[['m5_1','m7','m5_2']].apply(make_575, axis=1)

    # Few-shot pool
    df_masters = df.copy()

    # Generation settings
    few_shot_k, num_targets = 6, 20
    num_beams, num_beam_groups, diversity_penalty = 6, 3, 0.7

    stats, raw_outputs = [], []
    for _, tgt in df.sample(num_targets, random_state=42).iterrows():
        exs = retrieve_examples(tgt, df_masters, few_shot_k)

        header = (
            f"作者: {tgt['author']}\n"
            f"季語: {tgt['word']} (ID: {tgt['kigo_id']})\n"
            f"季節: {tgt['season']}\n\n"
        )

        prompt = header + build_prompt_gemma(tgt, exs)
        inp = tok(prompt, return_tensors='pt').to(device)

        # Generate line 1
        gen1 = model.generate(
            **inp,
            max_new_tokens=10,
            num_beams=num_beams,
            early_stopping=True,
            eos_token_id=end_token_id,
            stopping_criteria=stop_criteria
        )
        new1 = gen1[0, inp.input_ids.shape[-1]:]
        line1 = clean_haiku(tok.decode(new1, skip_special_tokens=True))

        # Generate line 2
        prompt2 = prompt + f"前の行: {line1}\n2行目(7モーラ):\n"
        inp2 = tok(prompt2, return_tensors='pt').to(device)

        gen2 = model.generate(
            **inp2,
            max_new_tokens=14,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            num_return_sequences=5,
            early_stopping=True,
            eos_token_id=end_token_id,
            stopping_criteria=stop_criteria
        )
        lines2 = [
            clean_haiku(tok.decode(seq[inp2.input_ids.shape[-1]:], skip_special_tokens=True))
            for seq in gen2
        ]

        # Rerank with line 3
        best_score, best_haiku = -float('inf'), None
        for l2 in lines2:
            prompt3 = prompt2 + f"前の行: {l2}\n3行目(5モーラ):\n"
            inp3 = tok(prompt3, return_tensors='pt').to(device)

            gen3 = model.generate(
                **inp3,
                max_new_tokens=10,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                num_return_sequences=5,
                early_stopping=True,
                eos_token_id=end_token_id,
                stopping_criteria=stop_criteria
            )
            for seq3 in gen3:
                new3 = seq3[inp3.input_ids.shape[-1]:]
                l3 = clean_haiku(tok.decode(new3, skip_special_tokens=True))
                cand = f"{line1}\n{l2}\n{l3}"
                ppl = compute_perplexity(model, tok, cand, device)
                bleu = compute_bleu(tgt['ref_haiku'], cand)
                score = bleu - 0.2 * math.log(ppl)
                if score > best_score:
                    best_score, best_haiku = score, cand

        avg_ppl = compute_perplexity(model, tok, best_haiku, device)
        bleu_score = compute_bleu(tgt['ref_haiku'], best_haiku)
        mora_rate = 1.0 if is_575(best_haiku) else 0.0
        kigo_rate = 1.0 if tgt['word'] in best_haiku else 0.0

        stats.append({
            'kigo': tgt['word'],
            'season': tgt['season'],
            'haiku_structure': tgt['haiku_structure'],
            'm5_1': tgt['m5_1'],
            'm7': tgt['m7'],
            'm5_2': tgt['m5_2'],
            'ref_haiku': tgt['ref_haiku'],
            'repr_haiku': best_haiku,
            'avg_ppl': avg_ppl,
            'mora_rate': mora_rate,
            'kigo_rate': kigo_rate,
            'bleu': bleu_score,
        })
        raw_outputs.append({
            'prompt_kigo': tgt['word'],
            'cleaned_haiku': best_haiku
        })

    pd.DataFrame(stats).to_csv(
        os.path.join(args.output_dir, 'iterative_eval.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    pd.DataFrame(raw_outputs).to_csv(
        os.path.join(args.output_dir, 'iterative_raw_outputs.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    print('Done. Iterative results saved.')


if __name__ == '__main__':
    main()

# Only use same-kigo when you’ve got ≥ k examples
'''
def retrieve_examples_strict_kigo(target, pool, few_shot_k=6):
    # don’t include the target itself
    pool = pool[pool.haiku_id != target.haiku_id]

    # all same-season candidates
    season_pool = pool[pool.season == target.season]

    # only restrict to same kigo if we can fill all k
    same_kigo = season_pool[season_pool.kigo_id == target.kigo_id]
    if len(same_kigo) >= few_shot_k:
        subset = same_kigo
    else:
        subset = season_pool

    # now sample one per author until k
    selected = []
    for auth in subset.author.dropna().unique():
        auth_samples = subset[subset.author == auth]
        selected.append(auth_samples.sample(1, random_state=target.name))
        if len(selected) == few_shot_k:
            break

    # if still short, fill randomly from subset
    while len(selected) < few_shot_k and not subset.empty:
        selected.append(subset.sample(1, random_state=target.name + len(selected)))

    return pd.concat(selected).reset_index(drop=True)

'''









# A mixed-pool for retrival examples: 2 strict + 4 loose
# fewshot_eval_mixed_pool

# haiku_beamsearch_STABLELM_Gemma.py
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

def retrieve_examples(target, pool, few_shot_k=6, strict_k=2):
    # don’t include the target itself
    pool = pool[pool.haiku_id != target.haiku_id]

    # same-season pool
    season_pool = pool[pool.season == target.season]
    # same-kigo subset
    kigo_pool   = season_pool[season_pool.kigo_id == target.kigo_id]

    # pick up to strict_k from kigo_pool
    strict_n = min(len(kigo_pool), strict_k)
    strict_sel = (
        kigo_pool
        .groupby("author")
        .apply(lambda df: df.sample(1, random_state=target.name))
        .reset_index(drop=True)
        .sample(strict_n, random_state=target.name)
    )

    # fill the rest from the broader season_pool (excluding already chosen)
    remaining_pool = season_pool.drop(strict_sel.index, errors="ignore")
    loose_sel = remaining_pool.sample(few_shot_k - strict_n, random_state=target.name + 1)

    return pd.concat([strict_sel, loose_sel]).reset_index(drop=True)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    tok = LlamaTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)

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
        os.path.join(args.output_dir, "fewshot_eval_same_kigo.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(raw_outputs).to_csv(
        os.path.join(args.output_dir, "fewshot_raw_outputs_same_kigo.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("✅ Done. Results saved in output_dir.")

if __name__ == "__main__":
    main()





#fewshot_eval_mixed_filtered

# haiku_beamsearch_STABLELM_Gemma.py
# Haiku generation with StableLM (Gemma) — Beam Search + PPL+BLEU reranking
# Retrieval: mixed (same-season with up-to-k same-kigo), filtered to 5-7-5 examples when possible

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


# ---------- Mora utilities ----------
def count_mora(japanese_text: str) -> int:
    phonemes = pyopenjtalk.g2p(japanese_text)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = haiku.strip().split("\n")
    return len(lines) == 3 and [count_mora(l) for l in lines] == [5, 7, 5]


# ---------- Cleaning ----------
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


# ---------- Metrics ----------
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


# ---------- Prompt helpers ----------
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


# ---------- Retrieval (filtered mixed) ----------
def retrieve_examples(target, pool, few_shot_k=6, strict_k=3, enforce_is575=True):
    """
    Mixed retrieval with global one-per-author and strict-first ordering.

    - always same season
    - up to `strict_k` from same kigo (one per author)
    - fill remainder from same season (one per author, excluding authors already used)
    - prefer examples that pass is_575; relax filtering if not enough
    """
    # 1) exclude target itself
    pool = pool[pool.haiku_id != target.haiku_id]

    # 2) same-season pool & same-kigo subset (unfiltered)
    season_pool_all = pool[pool.season == target.season]
    kigo_pool_all   = season_pool_all[season_pool_all.kigo_id == target.kigo_id]

    # helper: apply 5-7-5 filter if requested
    def filt(df):
        if not enforce_is575 or df.empty:
            return df
        if "is575" in df.columns:
            return df[df.is575]
        # fallback compute if column missing
        return df[df.ref_haiku.apply(is_575)]

    season_pool = filt(season_pool_all)
    kigo_pool   = filt(kigo_pool_all)

    # deterministic seed per target row
    seed = int(getattr(target, "name", 0)) if pd.notnull(getattr(target, "name", None)) else 0

    def one_per_author(df, n, seed, exclude_authors=None):
        """Sample up to n rows, at most 1 per author, excluding given authors."""
        if df.empty or n <= 0:
            return df.sample(0)
        d = df.copy()
        d = d[d.author.notna()]
        if exclude_authors:
            d = d[~d.author.isin(exclude_authors)]
        if d.empty:
            return d.sample(0)
        per_author = (
            d.groupby("author", dropna=True)
             .apply(lambda g: g.sample(1, random_state=seed))
             .reset_index(drop=True)
        )
        take = min(n, len(per_author))
        return per_author.sample(take, random_state=seed)

    # 3) strict selection (filtered, same kigo, one per author)
    strict_sel = one_per_author(kigo_pool, strict_k, seed)
    used_authors = set(strict_sel.author.dropna().tolist())

    # 4) loose selection (filtered, same season, one per author, exclude used authors)
    need = max(0, few_shot_k - len(strict_sel))
    loose_sel = one_per_author(season_pool, need, seed + 1, exclude_authors=used_authors)

    # 5) if still short, relax filtering in stages
    if len(strict_sel) + len(loose_sel) < few_shot_k:
        # try unfiltered kigo to top up strict
        more_need = max(0, strict_k - len(strict_sel))
        if more_need > 0 and not kigo_pool_all.empty:
            more_strict = one_per_author(kigo_pool_all, more_need, seed + 2, exclude_authors=used_authors)
            if not more_strict.empty:
                strict_sel = pd.concat([strict_sel, more_strict], ignore_index=True)
                used_authors.update(more_strict.author.dropna().tolist())

        # then unfiltered season to finish
        need = max(0, few_shot_k - len(strict_sel) - len(loose_sel))
        if need > 0 and not season_pool_all.empty:
            more_loose = one_per_author(season_pool_all, need, seed + 3, exclude_authors=used_authors)
            if not more_loose.empty:
                loose_sel = pd.concat([loose_sel, more_loose], ignore_index=True)
                used_authors.update(more_loose.author.dropna().tolist())

    # 6) if still short, drop one-per-author just to fill from season_all
    out = pd.concat([strict_sel, loose_sel], ignore_index=True)
    if len(out) < few_shot_k and not season_pool_all.empty:
        extra_need = few_shot_k - len(out)
        remaining = season_pool_all[~season_pool_all.haiku_id.isin(out.haiku_id)]
        if not remaining.empty:
            extra = remaining.sample(min(extra_need, len(remaining)), random_state=seed + 4)
            out = pd.concat([out, extra], ignore_index=True)

    return out.head(few_shot_k).reset_index(drop=True)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    tok = LlamaTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)

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

    # ---------- Data ----------
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

    # Precompute 5-7-5 for retrieval filtering
    df["is575"] = df["ref_haiku"].apply(is_575)

    few_shot_k = 6
    strict_k = 3
    num_targets = 20
    max_new_tokens = 20
    num_beams = 6
    num_return_sequences = 5
    num_beam_groups = 3
    diversity_penalty = 0.7

    stats = []
    raw_outputs = []

    for _, tgt in df.sample(num_targets, random_state=42).iterrows():
        exs = retrieve_examples(tgt, df, few_shot_k=few_shot_k, strict_k=strict_k, enforce_is575=True)
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
            final_scores.append(bleu - 0.2 * (math.log(ppl) if (math.isfinite(ppl) and ppl > 0) else 0.0))
            raw_outputs.append({
                "prompt_kigo":   tgt.word,
                "raw_output":    raw,
                "cleaned_haiku": h
            })

        best_idx = final_scores.index(max(final_scores))
        repr_haiku = outputs[best_idx]
        repr_bleu = bleus[best_idx]
        finite_p = [p for p in ppls if math.isfinite(p)]
        avg_ppl = (sum(finite_p) / len(finite_p)) if finite_p else float("inf")
        mora_rate = sum(is_575(h) for h in outputs if h) / (len(outputs) or 1)
        word = tgt.word or ""
        kigo_rate = sum(1 for h in outputs if word in (h or "")) / (len(outputs) or 1)

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
        os.path.join(args.output_dir, "fewshot_eval_mixed_filtered.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(raw_outputs).to_csv(
        os.path.join(args.output_dir, "fewshot_raw_outputs_mixed_filtered.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("✅ Done. Results saved in output_dir.")


if __name__ == "__main__":
    main()



#fewshot_eval_mixed_filtered_2

# A mixed-pool for retrival examples: 2 strict + 4 loose
# fewshot_eval_mixed_pool

# haiku_beamsearch_STABLELM_Gemma.py
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


def retrieve_examples(target, pool, few_shot_k=6, strict_k=2, enforce_is575=True):
    # exclude the target itself
    pool = pool[pool.haiku_id != target.haiku_id]

    # always same-season
    season_pool = pool[pool.season == target.season]

    # ---- 5-7-5 filtering (preferred, with graceful fallback) ----
    if enforce_is575 and not season_pool.empty:
        if "is575" in season_pool.columns:
            season_filtered = season_pool[season_pool.is575]
        else:
            # fallback: compute on the fly using your existing is_575 + ref_haiku
            season_filtered = season_pool[season_pool.ref_haiku.apply(is_575)]
        # only replace if we actually have some 5-7-5 examples
        if not season_filtered.empty:
            season_pool = season_filtered

    # same-kigo subset (inherits the filter from season_pool)
    kigo_pool = season_pool[season_pool.kigo_id == target.kigo_id]

    # deterministic seed per target row (same as your previous code)
    seed = int(getattr(target, "name", 0)) if pd.notnull(getattr(target, "name", None)) else 0

    # pick up to strict_k from kigo_pool, one per author
    strict_n = min(len(kigo_pool), strict_k)
    if strict_n > 0 and not kigo_pool.empty:
        per_author_kigo = (
            kigo_pool[kigo_pool.author.notna()]
            .groupby("author", dropna=True)
            .apply(lambda df: df.sample(1, random_state=seed))
        )
        # flatten MultiIndex if present and sample up to strict_n
        if isinstance(per_author_kigo.index, pd.MultiIndex):
            per_author_kigo.index = per_author_kigo.index.get_level_values(-1)
        strict_sel = per_author_kigo.sample(min(strict_n, len(per_author_kigo)), random_state=seed)
    else:
        strict_sel = season_pool.iloc[0:0]  # empty frame with same columns

    # exclude strict picks before loose sampling (use IDs, not indices)
    strict_ids = set(strict_sel.haiku_id.tolist())
    remaining_pool = season_pool[~season_pool.haiku_id.isin(strict_ids)]

    # fill the rest from the broader season_pool, one per author
    need = max(0, few_shot_k - len(strict_sel))
    if need > 0 and not remaining_pool.empty:
        per_author_season = (
            remaining_pool[remaining_pool.author.notna()]
            .groupby("author", dropna=True)
            .apply(lambda df: df.sample(1, random_state=seed + 1))
        )
        if isinstance(per_author_season.index, pd.MultiIndex):
            per_author_season.index = per_author_season.index.get_level_values(-1)
        loose_sel = per_author_season.sample(min(need, len(per_author_season)), random_state=seed + 1)
    else:
        loose_sel = remaining_pool.iloc[0:0]

    # if still short, relax one-per-author just to fill quota
    out = pd.concat([strict_sel, loose_sel], ignore_index=False)
    if len(out) < few_shot_k and not remaining_pool.empty:
        extra_need = few_shot_k - len(out)
        filler = remaining_pool[~remaining_pool.haiku_id.isin(set(out.haiku_id.tolist()))]
        if not filler.empty:
            out = pd.concat([out, filler.sample(min(extra_need, len(filler)), random_state=seed + 2)], ignore_index=False)

    return out.head(few_shot_k).reset_index(drop=True)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    tok = LlamaTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)

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
        os.path.join(args.output_dir, "fewshot_eval_mixed_pool_2.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(raw_outputs).to_csv(
        os.path.join(args.output_dir, "fewshot_raw_outputs_mixed_pool_2.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("✅ Done. Results saved in output_dir.")

if __name__ == "__main__":
    main()


# iterative_eval_2

# haiku_beamsearch_STABLELM_Gamma.py (iterative 3-step generation)
# Haiku generation with StableLM (Gemma) — Iterative Line-by-Line Beam Search + PPL+BLEU reranking

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
    return "\n".join(prompt_blocks)


def retrieve_examples(target, pool, few_shot_k=6, enforce_is575=True):
    """
    Season-first retrieval with optional 5-7-5 preference.

    Order of narrowing:
      1) same season + same kigo + same author (prefer 5-7-5)
      2) same season + same kigo          (prefer 5-7-5)
      3) same season only                 (prefer 5-7-5)

    Sampling:
      - one per author first (deterministic per target), then fill the remainder.
    """

    # 0) exclude the target itself
    pool = pool[pool["haiku_id"] != target["haiku_id"]]

    # 1) same-season base
    season = pool[pool["season"] == target["season"]]

    # helper: prefer rows that pass is_575; if none, return original df
    def prefer_575(df):
        if not enforce_is575 or df.empty:
            return df
        if "is575" in df.columns:
            filtered = df[df["is575"]]
        else:
            # compute on the fly if not precomputed
            filtered = df[df["ref_haiku"].apply(is_575)]
        return filtered if not filtered.empty else df

    # build progressively looser subsets
    subset = season[
        (season["kigo_id"] == target["kigo_id"]) &
        (season["author"]  == target["author"])
    ]
    subset = prefer_575(subset)

    if len(subset) < few_shot_k:
        subset = prefer_575(season[season["kigo_id"] == target["kigo_id"]])

    if len(subset) < few_shot_k:
        subset = prefer_575(season)

    # deterministic seed per target (works with iterrows() index in target.name)
    seed = int(getattr(target, "name", 0)) if pd.notnull(getattr(target, "name", None)) else 0

    # 2) sample one per author first
    selected = []
    authors = subset["author"].dropna().unique()
    # Shuffle authors deterministically
    if len(authors) > 0:
        authors = pd.Series(authors).sample(frac=1.0, random_state=seed).tolist()
        for auth in authors:
            auth_rows = subset[subset["author"] == auth]
            if not auth_rows.empty:
                selected.append(auth_rows.sample(1, random_state=seed))
            if len(selected) == few_shot_k:
                break

    # 3) if still short, fill from whatever remains (allow repeats per author)
    if len(selected) < few_shot_k and not subset.empty:
        need = few_shot_k - len(selected)
        # avoid re-picking the same rows if possible
        already_ids = set(pd.concat(selected)["haiku_id"]) if selected else set()
        filler_pool = subset[~subset["haiku_id"].isin(already_ids)]
        if filler_pool.empty:
            filler_pool = subset
        selected.append(filler_pool.sample(min(need, len(filler_pool)), random_state=seed + 1))

    if not selected:
        # absolute fallback: return an empty-but-shaped frame (prevents concat errors)
        return subset.iloc[0:0].reset_index(drop=True)

    return pd.concat(selected, ignore_index=True).head(few_shot_k)



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

        # Setup model
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

    # Load and merge data
    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl, lines=True)
    df = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku","_kigo"))
    df.rename(columns={
        "season_haiku": "season",
        "haiku": "ref_haiku",
        "5_mora_segment_1": "m5_1",
        "7_mora_segment": "m7",
        "5_mora_segment_2": "m5_2",
    }, inplace=True)
    df = df[df['haiku_structure'] == "Regular"].copy()
    df["ref_haiku"] = df[["m5_1","m7","m5_2"]].apply(make_575, axis=1)

    # Prefilter to master poets for few-shot examples
    #masters = ["松尾芭蕉", "与謝蕪村", "小林一茶", "正岡子規"]
    #df_masters = df[df['author'].isin(masters)].reset_index(drop=True)
    df_masters = df.copy()

    # Generation settings
    few_shot_k = 6
    num_targets = 20
    num_beams = 6
    num_beam_groups = 3
    diversity_penalty = 0.7

    stats = []
    raw_outputs = []

    # Iterate targets from full df, but examples from df_masters
    for _, tgt in df.sample(num_targets, random_state=42).iterrows():
        exs = retrieve_examples(tgt, df_masters, few_shot_k)

        header = (
            f"作者: {tgt['author']}\n"
            f"季語: {tgt['word']} (ID: {tgt['kigo_id']})\n"
            f"季節: {tgt['season']}\n\n"
        )
        base_prompt = header + build_prompt_gemma(tgt, exs)

        # Generate first line (5 mora)
        inp1 = tok(base_prompt + "\n俳句:\n<end_of_turn>\n", return_tensors="pt").to(device)
        gen1 = model.generate(
            **inp1,
            max_new_tokens=20,
            num_beams=num_beams,
            early_stopping=True,
            eos_token_id=end_token_id,
            stopping_criteria=stop_criteria
        )
        line1 = clean_haiku(tok.decode(gen1[0], skip_special_tokens=False))

        # Generate second line (7 mora)
        prompt2 = base_prompt + f"\n前の行: {line1}\n2行目(7モーラ):\n<end_of_turn>\n"
        gen2 = model.generate(
            **tok(prompt2, return_tensors="pt").to(device),
            max_new_tokens=20,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            num_return_sequences=5,
            early_stopping=True,
            eos_token_id=end_token_id,
            stopping_criteria=stop_criteria
        )
        lines2 = [clean_haiku(tok.decode(g, skip_special_tokens=False)) for g in gen2]

        # Full haiku rerank
        best_score = -float('inf')
        best_haiku = None
        for l2 in lines2:
            prompt3 = base_prompt + f"\n前の行: {l2}\n3行目(5モーラ):\n<end_of_turn>\n"
            gen3 = model.generate(
                **tok(prompt3, return_tensors="pt").to(device),
                max_new_tokens=20,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                num_return_sequences=5,
                early_stopping=True,
                eos_token_id=end_token_id,
                stopping_criteria=stop_criteria
            )
            lines3 = [clean_haiku(tok.decode(g, skip_special_tokens=False)) for g in gen3]
            for l3 in lines3:
                full = f"{line1}\n{l2}\n{l3}"
                parts = full.split("\n")[:3]
                candidate = "\n".join(parts)
                score = compute_bleu(tgt['ref_haiku'], candidate) - 0.2 * math.log(compute_perplexity(model, tok, candidate, device))
                if score > best_score:
                    best_score = score
                    best_haiku = candidate

        # Record
        # After you’ve found best_haiku, compute the extra metrics:
        avg_ppl = compute_perplexity(model, tok, best_haiku, device)
        bleu_score = compute_bleu(tgt['ref_haiku'], best_haiku)
        mora_rate = 1.0 if is_575(best_haiku) else 0.0
        # does the kigo word actually appear?
        kigo_rate = 1.0 if tgt['word'] in best_haiku else 0.0

        stats.append({
            "kigo":            tgt['word'],
            "season":          tgt['season'],
            "haiku_structure": tgt['haiku_structure'],
            "m5_1":            tgt['m5_1'],
            "m7":              tgt['m7'],
            "m5_2":            tgt['m5_2'],
            "ref_haiku":       tgt['ref_haiku'],
            "repr_haiku":      best_haiku,
            "avg_ppl":         avg_ppl,
            "mora_rate":       mora_rate,
            "kigo_rate":       kigo_rate,
            "bleu":            bleu_score,
        })
        raw_outputs.append({"prompt_kigo": tgt['word'], "cleaned_haiku": best_haiku})

    # Save
    pd.DataFrame(stats).to_csv(os.path.join(args.output_dir, "iterative_eval.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(raw_outputs).to_csv(os.path.join(args.output_dir, "iterative_raw_outputs.csv"), index=False, encoding="utf-8-sig")
    print("Done. Iterative results saved.")

if __name__ == "__main__":
    main()