############
#QWEN MODEL#
############

#PROVA 1

import os
import re
import math
import argparse
import string
import unicodedata
import time
import torch
import pandas as pd
import pyopenjtalk
from typing import List, Iterable, Set, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessor,
    LogitsProcessorList,
)

# ---------- keep PyTorch sane on HPC
os.environ.setdefault("TRANSFORMERS_NO_TORCH_COMPILE", "1")
if hasattr(torch, "compile"):
    def _no_compile(fn, *args, **kwargs):
        return fn
    torch.compile = _no_compile
torch.set_grad_enabled(False)

# ---------- global knob (overridden by CLI)
MAX_NEW_TOKENS = 64  # default headroom; set via --gen_max_new_tokens

# ---------- modest per-stage budgets (used unless you override globally)
GEN_TOKENS_L1 = 32
GEN_TOKENS_L2 = 40
GEN_TOKENS_L3 = 40

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--kigo_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_targets", type=int, default=20)  # quick check
    p.add_argument("--few_shot_k", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_line_attempts", type=int, default=100)  # per line (L1/L2)
    p.add_argument("--max_haiku_attempts", type=int, default=800) # 0 = infinite tries for L3
    p.add_argument("--bleu_threshold", type=float, default=0.85)   # we can try 0.93 via CLI
    p.add_argument("--kigo_hint", action="store_true", default=True,
               help="If set, add a gentle instruction to include the kigo in line 3 when not present in L1/L2.")
    # NEW: generation budget knob for output (still overridable here)
    p.add_argument("--gen_max_new_tokens", type=int, default=128,
               help="Budget for generation per attempt. Used unless smaller per-stage caps apply.")
    return p.parse_args()

# -----------------------
# Stop on <end_of_turn>
# -----------------------
class StopOnEndOfTurn(StoppingCriteria):
    def __init__(self, end_token_id: int):
        super().__init__()
        self.end_token_id = end_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() == self.end_token_id

# -----------------------
# Japanese mora utils
# -----------------------
def count_mora(japanese_text: str) -> int:
    s = re.sub(r"\s+", "", japanese_text)
    s = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", s)
    phonemes = pyopenjtalk.g2p(s)
    return sum(1 for c in phonemes if c in "aeiouN")

def is_575(haiku: str) -> bool:
    lines = [l for l in haiku.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    try:
        return [count_mora(l) for l in lines[:3]] == [5, 7, 5]
    except Exception:
        return False

# -----------------------
# Cleaning + normalization
# -----------------------
def _strip_unicode_punct_symbol(s: str) -> str:
    return "".join(ch for ch in s if (unicodedata.category(ch)[0] not in ("P", "S")))

def clean_line(s: str) -> str:
    s = s.strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)  # remove all whitespace (incl. full-width)
    s = _strip_unicode_punct_symbol(s)
    s = re.sub(r"[0-9A-Za-z" + re.escape(string.punctuation) + r"]+", "", s)
    return s

# --- Think-aloud friendly extractor (still useful if model emits `俳句:` block)
_HAIKU_BLOCK_RX = re.compile(r"俳句:\s*\n(.+?)\n(.+?)\n(.+?)\s*(?:\n|$)")

def extract_haiku(output_text: str) -> Optional[str]:
    """
    If the model printed:
        俳句:
        <L1>
        <L2>
        <L3>
    …return exactly those three lines.
    """
    m = _HAIKU_BLOCK_RX.search(output_text)
    if m:
        return "\n".join(m.groups())
    return None

def clean_haiku(raw: str) -> str:
    # If a labeled block exists, trust it and skip the rest
    haiku_from_block = extract_haiku(raw)
    if haiku_from_block:
        return haiku_from_block

    # Fallback: first 3 clean lines (pre-<end_of_turn>)
    haiku_lines = []
    for line in raw.splitlines():
        if "<end_of_turn>" in line:
            break
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("*") or "ヒント" in line:
            continue
        # ignore label-ish lines if they appear, but don't require them
        if line.startswith("俳句:") or line.startswith("俳句の例:"):
            continue

        cleaned = clean_line(line)
        if cleaned:
            haiku_lines.append(cleaned)
        if len(haiku_lines) == 3:
            break

    while len(haiku_lines) < 3:
        haiku_lines.append("")
    return "\n".join(haiku_lines)

def normalize_jp(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)
    return s

# -----------------------
# Kigo detection (string OR phonetic match)
# -----------------------
def contains_kigo(text: str, kigo: str) -> bool:
    """Return True if kigo appears in `text` as exact string or by reading (pyopenjtalk g2p)."""
    tn = normalize_jp(text)
    kn = normalize_jp(kigo)
    if not tn or not kn:
        return False
    # direct (kanji) containment
    if kn in tn:
        return True
    # phonetic containment (robust to kana/kanji variants)
    try:
        tph = pyopenjtalk.g2p(tn).replace(" ", "")
        kph = pyopenjtalk.g2p(kn).replace(" ", "")
        return kph in tph
    except Exception:
        return False

# -----------------------
# Prompt-bleed guards: forbidden sets & continuation-only extraction
# -----------------------
# (3) stronger: treat label-like *starts* and ordinal line labels as meta
_LABEL_RX = re.compile(
    r"^((作者|季語|季節|構造|俳句|出力|一行目|二行目|三行目|四行目)|第?[一二三四五六七八九十0-9]+行目)[：: ]?"
)

def build_forbidden_set(texts: Iterable[str]) -> Set[str]:
    """Collect normalized lines we must NOT accept (examples, headers, earlier lines)."""
    forbid = set()
    for t in texts:
        if t is None:
            continue
        for ln in str(t).splitlines():
            raw = ln.strip()
            if not raw or _LABEL_RX.search(raw):
                continue
            nrm = clean_line(raw)
            if nrm:
                forbid.add(nrm)
    return forbid

# (2) ban terms that must never appear inside poem lines (plus regex for ordinals)
_BANNED_IN_OUTPUT = {
    "季語","kigo","俳句","五七五","５７５","七五","7-5-7","七五の和歌","和歌","短歌","切れ字","モーラ","行目"
}
_BANNED_ORDINAL_RX = re.compile(r"(第?[一二三四五六七八九十0-9]+)行(目)?")

def _has_banned_term(s: str) -> bool:
    return any(b in s for b in _BANNED_IN_OUTPUT) or bool(_BANNED_ORDINAL_RX.search(s))

# Extra ASCII guard (prevents English/meta scaffolding lines)
_ASCII_LETTER_RX = re.compile(r"[A-Za-z]")
def _has_ascii_letters(s: str) -> bool:
    return bool(_ASCII_LETTER_RX.search(s))

# === OPTION 1: prefer kigo-containing line in the extractor (used on L3 only)
def extract_first_fresh_line(new_only: str, forbidden: Set[str], prefer_kigo: Optional[str] = None) -> str:
    """Read only from the continuation and pick a line.
    If `prefer_kigo` is given, first prefer the first line that contains that kigo (normalized),
    as long as it is not exactly one of the forbidden example lines.
    """
    seg = new_only.split("<end_of_turn>")[0]
    kigo_norm = normalize_jp(prefer_kigo) if prefer_kigo else None

    # 1) Prefer a line that contains the kigo (normalized)
    if kigo_norm:
        for ln in seg.splitlines():
            raw = ln.strip()
            if not raw:
                continue
            if _LABEL_RX.search(raw):
                continue
            if re.search(r"[<>#{};=\[\]()]", raw):
                continue
            nrm = clean_line(raw)
            if not nrm:
                continue
            if kigo_norm in normalize_jp(nrm):
                if nrm not in forbidden:
                    return nrm

    # 2) Fallback to the original "first fresh, non-meta" behavior
    for ln in seg.splitlines():
        raw = ln.strip()
        if not raw:
            continue
        if _LABEL_RX.search(raw):
            continue
        if re.search(r"[<>#{};=\[\]()]", raw):
            continue
        nrm = clean_line(raw)
        if not nrm or nrm in forbidden:
            continue
        return nrm
    return ""

# -----------------------
# Metrics
# -----------------------
def strip_ascii_punct(s: str) -> str:
    s = re.sub(r"[A-Za-z0-9" + re.escape(string.punctuation) + r"]+", "", s)
    return s.replace(" ", "").strip()

def compute_perplexity(model, tok, text, device):
    s = "\n".join([strip_ascii_punct(l) for l in text.splitlines()])
    inputs = tok(s, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return math.exp(loss.item())

def compute_bleu(ref: str, hyp: str) -> float:
    smoothie = SmoothingFunction().method1
    ref_chars = [list(ref.replace("\n", ""))]
    hyp_chars = list(hyp.replace("\n", ""))
    return sentence_bleu(ref_chars, hyp_chars, smoothing_function=smoothie, weights=(1.0,))

def max_bleu_vs_list(candidate: str, refs: List[str]) -> float:
    smoothie = SmoothingFunction().method1
    hyp = list(candidate.replace("\n",""))
    mx = 0.0
    for ref in refs:
        mx = max(mx, sentence_bleu([list(ref.replace("\n",""))], hyp,
                                   smoothing_function=smoothie, weights=(1.0,)))
    return mx

def make_575(segments):
    a, b, c = segments
    if pd.notnull(a) and pd.notnull(b) and pd.notnull(c):
        return f"{a}\n{b}\n{c}"
    return ""

# -----------------------
# CHANGE: helpers to ignore kigo in the BLEU "too-similar" gate
# -----------------------
def _strip_kigo_for_bleu(text: str, kigo: str) -> str:
    t = normalize_jp(text)
    k = normalize_jp(kigo)
    return t.replace(k, "")

def _max_bleu_ignore_kigo(candidate: str, refs: List[str], kigo: str) -> float:
    stripped_cand = _strip_kigo_for_bleu(candidate, kigo)
    stripped_refs = [_strip_kigo_for_bleu(r, kigo) for r in refs]
    return max_bleu_vs_list(stripped_cand, stripped_refs)

# -----------------------
# --- Haiku style helpers ---
# -----------------------
_KIREJI = ["や", "かな", "けり"]
_BAD_STYLE = [
    "参照", "参照すべき", "参考", "例として", "データ", "アルゴリズム", "プロンプト",
    "いやそうじゃない", "つまり", "しかし", "ですから", "してください", "べき", "入力", "出力"
]

_KATAKANA_RX = re.compile(r"[ァ-ヺー]")
_KANJI_RX    = re.compile(r"[一-龯]")

def _kanji_ratio(s: str) -> float:
    s = s.replace("\n","")
    n = len(s)
    if n == 0: return 0.0
    return len(_KANJI_RX.findall(s))/n

def _katakana_ratio(s: str) -> float:
    s = s.replace("\n","")
    n = len(s)
    if n == 0: return 0.0
    return len(_KATAKANA_RX.findall(s))/n

def _has_kireji(cand: str) -> bool:
    lines = [l for l in cand.splitlines() if l.strip()]
    if len(lines) < 3: return False
    l1, l3 = lines[0], lines[2]
    return any(k in l1 for k in _KIREJI) or any(k in l3 for k in _KIREJI)

def _repetition_penalty(s: str) -> float:
    t = s.replace("\n","")
    pen = 0.0
    for n in (2,3):
        seen = {}
        for i in range(len(t)-n+1):
            chunk = t[i:i+n]
            seen[chunk] = seen.get(chunk, 0)+1
        repeats = sum(1 for _,v in seen.items() if v >= 3)
        pen += 0.15 * repeats
    return pen

def _contains_bad_style(s: str) -> bool:
    return any(b in s for b in _BAD_STYLE)

# -----------------------
# Prompt builders
# -----------------------
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
    blocks = [build_example_block_gemma(ex) for _, ex in examples.iterrows()]
    return "\n".join(blocks)

# -----------------------
# Retrieval
# -----------------------
def retrieve_examples(target, pool, few_shot_k=6, enforce_is575=True):
    pool = pool[pool["haiku_id"] != target["haiku_id"]]
    season = pool[pool["season"] == target["season"]]

    def prefer_575(df):
        if not enforce_is575 or df.empty:
            return df
        if "is575" in df.columns:
            filt = df[df["is575"]]
        else:
            filt = df[df["ref_haiku"].apply(is_575)]
        return filt if not filt.empty else df

    subset = season[(season["kigo_id"] == target["kigo_id"]) & (season["author"] == target["author"])]
    subset = prefer_575(subset)
    if len(subset) < few_shot_k:
        subset = prefer_575(season[season["kigo_id"] == target["kigo_id"]])
    if len(subset) < few_shot_k:
        subset = prefer_575(season)

    seed = int(getattr(target, "name", 0)) if pd.notnull(getattr(target, "name", None)) else 0

    selected = []
    authors = subset["author"].dropna().unique()
    if len(authors) > 0:
        authors = pd.Series(authors).sample(frac=1.0, random_state=seed).tolist()
        for auth in authors:
            rows = subset[subset["author"] == auth]
            if not rows.empty:
                selected.append(rows.sample(1, random_state=seed))
            if len(selected) == few_shot_k:
                break

    if len(selected) < few_shot_k and not subset.empty:
        need = few_shot_k - len(selected)
        already = set(pd.concat(selected)["haiku_id"]) if selected else set()
        filler = subset[~subset["haiku_id"].isin(already)]
        if filler.empty:
            filler = subset
        selected.append(filler.sample(min(need, len(filler)), random_state=seed + 1))

    if not selected:
        exs = subset.iloc[0:0].reset_index(drop=True)
    else:
        exs = pd.concat(selected, ignore_index=True).head(few_shot_k)

    # Soft style priming: prefer refs with kireji at the top
    if not exs.empty:
        def _row_has_kireji(row): return any(k in str(row["ref_haiku"]) for k in _KIREJI)
        exs = pd.concat([exs[exs.apply(_row_has_kireji, axis=1)],
                         exs[~exs.apply(_row_has_kireji, axis=1)]], ignore_index=True)
    return exs

# -----------------------
# Anti-copy: n-gram blocker
# -----------------------
def build_example_ngrams(tok, examples_texts, n_min=8, n_max=12):
    blocked = set()
    for txt in examples_texts:
        ids = tok(txt, add_special_tokens=False).input_ids
        L = len(ids)
        for n in range(n_min, n_max + 1):
            for i in range(L - n + 1):
                blocked.add(tuple(ids[i:i+n]))
    return blocked

class NoCopyFromExamples(LogitsProcessor):
    """Block any next-token that would complete an n-gram seen in the examples."""
    def __init__(self, blocked_ngrams, n_min=8, n_max=12):
        self.n_min, self_n_max = n_min, n_max
        self.prefix_map = {}
        for gram in blocked_ngrams:
            if not (n_min <= len(gram) <= n_max):
                continue
            pref = gram[:-1]
            self.prefix_map.setdefault(pref, []).append(gram[-1])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        bsz, cur_len = input_ids.shape
        for b in range(bsz):
            for n in range(8, 13):
                if cur_len >= n - 1:
                    pref = tuple(input_ids[b, cur_len - (n - 1):cur_len].tolist())
                    if pref in self.prefix_map:
                        for bad_next in self.prefix_map[pref]:
                            scores[b, bad_next] = -float("inf")
        return scores

# -----------------------
# Helper: decode only the continuation (for debugging)
# -----------------------
def decode_new(generated_ids: torch.Tensor, prompt_ids: torch.Tensor, tok: AutoTokenizer) -> str:
    if generated_ids.dim() == 1:
        generated_ids = generated_ids.unsqueeze(0)
    new_ids = generated_ids[:, prompt_ids.shape[1]:]
    return tok.decode(new_ids[0], skip_special_tokens=False)

# -----------------------
# Sampler: one line with target mora (ALWAYS prints debug)
# -----------------------
def sample_line(model, tok, device, prompt, end_id, want_mora=None,
                logits_processors=None, bad_words_ids=None,
                tries=1, tag="", forbidden=None, prefer_kigo: Optional[str] = None,
                max_new_tokens: Optional[int] = None):
    """
    Added: per-call max_new_tokens override.
    """
    forbidden = forbidden or set()
    if logits_processors is None:
        logits_processors = []
    lp = LogitsProcessorList(list(logits_processors))
    for attempt in range(1, tries+1):
        inp = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inp,
            max_new_tokens=(max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS),
            do_sample=True,
            temperature=0.7,                 # tweaked
            top_p=0.90,
            top_k=50,                        # tweaked
            repetition_penalty=1.15,         # tweaked
            eos_token_id=end_id,
            pad_token_id=tok.pad_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnEndOfTurn(end_id)]),
            bad_words_ids=bad_words_ids,
            logits_processor=lp,
        )
        full_txt = tok.decode(out[0], skip_special_tokens=False)
        new_only = decode_new(out, inp["input_ids"], tok)

        # continuation-only + forbidden filter (+OPTION 1: prefer kigo line if provided)
        line = extract_first_fresh_line(new_only, forbidden, prefer_kigo=prefer_kigo)

        prompt_len = inp["input_ids"].shape[1]
        gen_len    = out[0].shape[0] - prompt_len
        print(f"\n[DEBUG {tag}] attempt {attempt}")
        print(f"  tokens: prompt={prompt_len} new={gen_len}")
        print(f"  produced_EOT_in_new={('<end_of_turn>' in new_only)}")
        print(f"  full_txt[:120]= {repr(full_txt[:120])}")
        print(f"  new_only[:120]= {repr(new_only[:120])}")
        print(f"  picked_line={repr(line)}")

        if not line:
            print("  -> reject: empty or forbidden after clean")
            continue
        if re.search(r"[<>#{};=\[\]()]", line):
            print("  -> reject: meta/code pattern")
            continue
        if _has_ascii_letters(line):
            print("  -> reject: contains ASCII letters")
            continue
        if _has_banned_term(line):
            print("  -> reject: banned meta/ordinal term in line")
            continue
        if want_mora is not None:
            try:
                m = count_mora(line)
                if m != want_mora:
                    print(f"  -> reject: mora {m} != {want_mora}")
                    continue
            except Exception:
                print("  -> reject: mora counter exception")
                continue

        print(f"  -> ACCEPT line={repr(line)}")
        return line

    print(f"[DEBUG {tag}] FAILED after {tries} tries")
    return ""

# -----------------------
# Main
# -----------------------
def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()

    # update global generation budget from CLI
    global MAX_NEW_TOKENS
    MAX_NEW_TOKENS = int(args.gen_max_new_tokens)

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.perf_counter()  # ---- runtime start
    torch.manual_seed(args.seed)

    # Model / tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    if "<end_of_turn>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tok))
    model.eval()

    # Data
    df_h = pd.read_json(args.train_jsonl, lines=True)
    df_k = pd.read_json(args.kigo_jsonl,  lines=True)
    df = df_h.merge(df_k, on="haiku_id", how="left", suffixes=("_haiku", "_kigo"))
    df.rename(columns={
        "season_haiku": "season",
        "haiku": "ref_haiku",
        "5_mora_segment_1": "m5_1",
        "7_mora_segment":  "m7",
        "5_mora_segment_2": "m5_2",
    }, inplace=True)
    df = df[df["haiku_structure"] == "Regular"].copy()
    df["ref_haiku"] = df[["m5_1", "m7", "m5_2"]].apply(make_575, axis=1)
    df["is575"] = df["ref_haiku"].apply(is_575)
    df_masters = df.copy()

    # (1) basic bad-words to avoid meta/code + meta/genre self-reference + ordinal labels
    BAD_STRINGS = [
        # meta/instructional Japanese
        "以下の情報", "出力", "一行目", "二行目", "三行目", "俳句:", "結果:", "正解です",
        # code / markup
        "<?php","<html","<head","<body","<!DOCTYPE","#include","#pragma",
        "namespace ","class ","public ","private ","package ","import ","using ",
        "std::","var ","const ","function ",
        # misc prompts
        "上の作例は参考です", "日本語の二行", "五七五形式",
        "次の行", "前の行", "後の行", "行目", "モーラ",
        # NEW: hard-block meta/genre/self-referential terms
        "季語","kigo","俳句","五七五","５７５","七五","7-5-7","七五の和歌","和歌","短歌","切れ字",
        # NEW: style offenders
        "参照", "参照すべき", "参考", "例として", "アルゴリズム", "プロンプト", "入力", "出力",
        "いやそうじゃない", "ですから", "してください", "べき", "解析", "評価結果", "テスト",
        # NEW: ordinal label variants
        "一行目","二行目","三行目","四行目","第1行","第2行","第3行","第4行","第一行","第二行","第三行","第四行",
        # NEW: reasoning/tool markers & variants you saw in logs
        "</think>", "<think>", "<tool_call>", "<|im_start|>", "<|im_end|>",
        "以下のように", "以下のように考えました", "手順", "ステップ", "問題文", "ユーザー",
        # Chinese/variant words that bled into output
        "俳曲", "季语", "季节", "切れ文字", "モーリー", "注意:", "確認して"
    ]
    bad_words_ids = tok(BAD_STRINGS, add_special_tokens=False).input_ids

    skip_counters = {
        "l1_fail": 0,
        "l2_fail": 0,
        "accepted": 0,
    }

    stats = []
    raw_outputs = []

    targets = df.sample(args.num_targets, random_state=args.seed)

    for t_idx, (_, tgt) in enumerate(targets.iterrows(), start=1):
        print("\n" + "="*70)
        print(f"[TARGET {t_idx}] kigo={tgt['word']}  season={tgt['season']}  author={tgt['author']}")
        print("="*70, flush=True)

        exs = retrieve_examples(tgt, df_masters, few_shot_k=args.few_shot_k)
        example_texts = [ex["ref_haiku"] for _, ex in exs.iterrows()]
        print(f"[INFO] Retrieved {len(example_texts)} examples for prompt.")
        for i, ex in enumerate(example_texts, 1):
            print(f"  EX{i}:\n{ex}\n---")

        header = (
            f"作者: {tgt['author']}\n"
            f"季語: {tgt['word']} (ID: {tgt['kigo_id']})\n"
            f"季節: {tgt['season']}\n\n"
            # IMPORTANT: do not print reasoning, only outputs in the requested format
            "重要: 思考は出力せず、最終出力のみを表示してください。"
            "説明、理由、箇条書き、英語、ツールタグ、マークアップは出力しないでください。\n"
            "出力は指示された行のみ、または次の形式の俳句ブロックのみです：\n"
            "俳句:\n"
            "（一行目 5モーラ）\n"
            "（二行目 7モーラ）\n"
            "（三行目 5モーラ）\n\n"
        )
        base_prompt = header + build_prompt_gemma(tgt, exs)

        blocked_ngrams = build_example_ngrams(tok, example_texts, n_min=8, n_max=12)
        print(f"[INFO] Blocked n-grams: {len(blocked_ngrams)}")
        no_copy_proc = NoCopyFromExamples(blocked_ngrams)

        near_pool = df[(df["season"] == tgt["season"]) & (df["kigo_id"] == tgt["kigo_id"])]
        near_refs = near_pool["ref_haiku"].dropna().tolist()
        print(f"[INFO] Near-pool refs: {len(near_refs)}")

        # -------- L1: keep sampling until a 5-mora line
        prompt1 = (
            base_prompt +
            "上の作例は参考です。これから新しい俳句を作ります。\n"
            "出力: 日本語の一行のみ（漢字かなのみ／数字・英字・記号・注釈なし）。\n"
            "「や」「かな」「けり」などの切れ字を自然に用い、端的で映像的な言葉遣いにしてください。\n"
            "考えは出力せず、行そのものだけを書いてください。\n"
            "一行目（5モーラ）のみを書き、直後に<end_of_turn>。\n\n"
            "一行目:\n"
        )
        forbidden1 = build_forbidden_set([base_prompt])
        print("\n[STEP1] Sampling first line (5 mora, keep trying)...")
        line1 = ""
        for gl_try in range(1, args.max_line_attempts + 1):
            l1 = sample_line(
                model, tok, device, prompt1, end_id, want_mora=5,
                logits_processors=[no_copy_proc],
                bad_words_ids=bad_words_ids,
                tries=1, tag=f"T{t_idx}-L1-{gl_try}",
                forbidden=forbidden1,
                prefer_kigo=None,
                max_new_tokens=min(GEN_TOKENS_L1, MAX_NEW_TOKENS)
            )
            if l1:
                line1 = l1
                break
        if not line1:
            skip_counters["l1_fail"] += 1
            print("[STEP1] FAILED to meet 5 mora. Restarting this target is recommended.")
        print(f"[STEP1] PICK: {repr(line1)}  (mora={(count_mora(line1) if line1 else -1)})\n")

        # -------- L2: keep sampling until a 7-mora line
        prompt2 = (
            base_prompt +
            f"ここまでに作った一行目:\n{line1}\n\n"
            "考えは出力せず、行そのものだけを書いてください。\n"
            "出力: 二行目（7モーラ）のみ。直後に<end_of_turn>。\n\n"
            "二行目:\n"
        )
        forbidden2 = build_forbidden_set([base_prompt, line1])
        print("[STEP2] Sampling second line (7 mora, keep trying)...")
        line2 = ""
        for gl_try in range(1, args.max_line_attempts + 1):
            l2 = sample_line(
                model, tok, device, prompt2, end_id, want_mora=7,
                logits_processors=[no_copy_proc],
                bad_words_ids=bad_words_ids,
                tries=1, tag=f"T{t_idx}-L2-{gl_try}",
                forbidden=forbidden2,
                prefer_kigo=None,
                max_new_tokens=min(GEN_TOKENS_L2, MAX_NEW_TOKENS)
            )
            if l2:
                line2 = l2
                break
        if not line2:
            skip_counters["l2_fail"] += 1
            print("[STEP2] FAILED to meet 7 mora. Continuing to L3 regardless.")
        print(f"[STEP2] PICK: {repr(line2)}  (mora={(count_mora(line2) if line2 else -1)})\n")

        # -------- L3: keep sampling until an ACCEPTABLE full haiku (by sampling line 3)
        has_kigo_12 = contains_kigo(f"{line1}\n{line2}", tgt["word"])
        kigo_hint = (
            f"注意: まだ季語「{tgt['word']}」が含まれていません。三行目に必ず一度だけ「{tgt['word']}」を入れてください。\n"
            if (args.kigo_hint and not has_kigo_12) else
            ""
        )

        prompt3 = (
            base_prompt +
            f"ここまでに作った一行目と二行目:\n{line1}\n{line2}\n\n" +
            kigo_hint +
            "可能なら切れ字を自然に用い、説明口調や会話調は避けてください。\n"
            "考えは出力せず、行そのものだけを書いてください。\n"
            "出力: 三行目（5モーラ）のみ。直後に<end_of_turn>。\n\n"
            "三行目:\n"
        )
        forbidden3 = build_forbidden_set([base_prompt, line1, line2])

        def acceptable(cand: str) -> bool:
            if not is_575(cand):
                return False
            if not contains_kigo(cand, tgt["word"]):
                return False
            if _has_banned_term(cand):
                return False
            return True

        print("[STEP3] Sampling third line (5 mora) until ACCEPTABLE...")
        best_seen = None
        best_seen_kigo = None
        best_score = -1e9
        best_score_kigo = -1e9

        # meter-safe bests to protect mora_rate
        best_seen_575 = None
        best_score_575 = -1e9
        best_seen_kigo_575 = None
        best_score_kigo_575 = -1e9

        def style_score(c):
            ppl = compute_perplexity(model, tok, c, device)
            sim_ex  = max_bleu_vs_list(c, example_texts)
            sim_loc = max_bleu_vs_list(c, near_refs)
            meter   = 1.0 if is_575(c) else 0.0
            has_kigo= 1.0 if contains_kigo(c, tgt["word"]) else 0.0

            # diction features
            kireji_bonus   = 0.35 if _has_kireji(c) else 0.0
            kanji_bonus    = min(_kanji_ratio(c), 0.35)
            katakana_pen   = min(_katakana_ratio(c), 0.30)
            repeat_pen     = _repetition_penalty(c)

            if _contains_bad_style(c) or _has_banned_term(c):
                return -1e9

            return (
                -math.log(ppl + 1e-9)
                + 0.8*meter + 0.4*has_kigo
                + kireji_bonus + 0.5*kanji_bonus
                - 0.8*katakana_pen - 0.6*repeat_pen
                - 1.2*max(sim_ex, sim_loc)
            )

        accepted = None
        gl_try = 0
        while accepted is None:
            gl_try += 1
            l3 = sample_line(
                model, tok, device, prompt3, end_id, want_mora=5,
                logits_processors=[no_copy_proc],
                bad_words_ids=bad_words_ids,
                tries=1, tag=f"T{t_idx}-L3-{gl_try}",
                forbidden=forbidden3,
                prefer_kigo=tgt["word"],
                max_new_tokens=min(GEN_TOKENS_L3, MAX_NEW_TOKENS)
            )
            if not l3:
                if args.max_haiku_attempts and gl_try >= args.max_haiku_attempts:
                    print("[STEP3] Reached cap without any valid line; breaking.")
                    break
                continue

            cand_raw = f"{line1}\n{line2}\n{l3}"
            cand = clean_haiku(cand_raw)

            if _has_banned_term(cand) or _has_ascii_letters(cand):
                print("  -> reject: banned/meta/ASCII in candidate")
                if args.max_haiku_attempts and gl_try >= args.max_haiku_attempts:
                    print("[STEP3] Reached cap (banned/meta); breaking.")
                    break
                continue

            # track best seen regardless
            sc = style_score(cand)
            if sc > best_score:
                best_score, best_seen = sc, cand
            if contains_kigo(cand, tgt["word"]) and sc > best_score_kigo:
                best_score_kigo, best_seen_kigo = sc, cand

            # track meter-safe bests
            if is_575(cand):
                if sc > best_score_575:
                    best_score_575, best_seen_575 = sc, cand
                if contains_kigo(cand, tgt["word"]) and sc > best_score_kigo_575:
                    best_score_kigo_575, best_seen_kigo_575 = sc, cand

            if not acceptable(cand):
                if args.max_haiku_attempts and gl_try >= args.max_haiku_attempts:
                    print("[STEP3] Reached cap (not acceptable yet); breaking.")
                    break
                continue

            # similarity gate that ignores the kigo itself
            b_ex  = _max_bleu_ignore_kigo(cand, example_texts, tgt["word"])
            b_loc = _max_bleu_ignore_kigo(cand, near_refs,     tgt["word"])
            print(f"  [L3-CAND] acceptable=True  BLEU_ex(¬kigo)={b_ex:.3f}  BLEU_loc(¬kigo)={b_loc:.3f}  is575={is_575(cand)}")
            if max(b_ex, b_loc) >= args.bleu_threshold:
                if args.max_haiku_attempts and gl_try >= args.max_haiku_attempts:
                    print("[STEP3] Reached cap (too similar); breaking.")
                    break
                continue

            accepted = cand
            break

        # Meter-safe fallback ordering (protects mora_rate without losing kigo gains)
        if accepted is None:
            if best_seen_kigo_575 is not None:
                print("[STEP3] Fallback: best 5-7-5 WITH kigo.")
                best_haiku = best_seen_kigo_575
            elif best_seen_575 is not None:
                print("[STEP3] Fallback: best 5-7-5 (kigo may or may not be present).")
                best_haiku = best_seen_575
            elif best_seen_kigo is not None:
                print("[STEP3] Fallback: best with kigo (meter relaxed).")
                best_haiku = best_seen_kigo
            else:
                if args.max_haiku_attempts == 0:
                    print("[STEP3] WARNING: infinite mode but no accept; using best seen (no-kigo).")
                else:
                    print("[STEP3] NOTE: No acceptable candidate within caps — using best seen (no-kigo).")
                best_haiku = best_seen if best_seen is not None else clean_haiku(f"{line1}\n{line2}\n")
        else:
            best_haiku = accepted

        skip_counters["accepted"] += 1

        print("\n[RESULT] Best candidate (cleaned):\n" + best_haiku)
        print(f"  is_575={is_575(best_haiku)}")
        print(f"  contains_kigo={contains_kigo(best_haiku, tgt['word'])}")

        # metrics
        avg_ppl   = compute_perplexity(model, tok, best_haiku, device)
        bleu_score= compute_bleu(tgt["ref_haiku"], best_haiku)
        mora_rate = 1.0 if is_575(best_haiku) else 0.0
        kigo_rate = 1.0 if contains_kigo(best_haiku, tgt["word"]) else 0.0

        stats.append({
            "kigo":            tgt["word"],
            "season":          tgt["season"],
            "haiku_structure": tgt["haiku_structure"],
            "m5_1":            tgt["m5_1"],
            "m7":              tgt["m7"],
            "m5_2":            tgt["m5_2"],
            "ref_haiku":       tgt["ref_haiku"],
            "repr_haiku":      best_haiku,
            "avg_ppl":         avg_ppl,
            "mora_rate":       mora_rate,
            "kigo_rate":       kigo_rate,
            "bleu_vs_ref":     bleu_score,
        })
        raw_outputs.append({"prompt_kigo": tgt["word"], "cleaned_haiku": best_haiku})

    # Save
    pd.DataFrame(stats).to_csv(os.path.join(args.output_dir, "iterative_eval.csv"),
                               index=False, encoding="utf-8-sig")
    pd.DataFrame(raw_outputs).to_csv(os.path.join(args.output_dir, "iterative_raw_outputs.csv"),
                                     index=False, encoding="utf-8-sig")

    # ---- runtime end
    elapsed = time.perf_counter() - start_time
    print("\n=== DEBUG SUMMARY ===")
    print(skip_counters)
    print(f"\n=== RUNTIME ===\nElapsed: {elapsed:.2f} s ({_fmt_hms(elapsed)})")
    try:
        with open(os.path.join(args.output_dir, "run_time.txt"), "w", encoding="utf-8") as f:
            f.write(f"{elapsed:.2f} seconds ({_fmt_hms(elapsed)})\n")
    except Exception as e:
        print(f"[WARN] Failed to write run_time.txt: {e}")

    print("Done. Iterative results saved.", flush=True)

if __name__ == "__main__":
    main()


#PROVA 2 - disabled the "thinking" in the output