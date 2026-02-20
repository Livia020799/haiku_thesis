import os
import re
import math
import argparse
import unicodedata
import time
import torch
import pandas as pd
import pyopenjtalk
from functools import lru_cache
from typing import List, Iterable, Set, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- keep PyTorch sane on HPC
os.environ.setdefault("TRANSFORMERS_NO_TORCH_COMPILE", "1")
if hasattr(torch, "compile"):
    def _no_compile(fn, *args, **kwargs):
        return fn
    torch.compile = _no_compile
torch.set_grad_enabled(False)

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
    p.add_argument("--num_targets", type=int, default=20)
    p.add_argument("--few_shot_k", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_haiku_attempts", type=int, default=550)
    p.add_argument("--bleu_threshold", type=float, default=0.80)
    p.add_argument("--kigo_hint", action="store_true", default=True,
                   help="If set, add a gentle instruction to include the kigo in the haiku.")
    return p.parse_args()

# -----------------------
# Japanese mora utils
# -----------------------
@lru_cache(maxsize=4096)
def _g2p_cached(s: str) -> str:
    try:
        return pyopenjtalk.g2p(s or "").replace(" ", "")
    except Exception:
        return ""

_SMALL_KANA = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮヵヶ")
_SOKUON     = set("っッ")
_MORAIC_N   = set("んン")
_PROLONG    = "ー"
_VOWELS  = {"a","e","i","o","u","A","E","I","O","U"}
_SPECIAL = {"N","cl","q"}  # uppercase N

def _sanitize_jp(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[A-Za-z0-9" + re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""") + r"]+", "", s)
    return s

@lru_cache(maxsize=4096)
def _g2p_kana(s: str) -> str:
    try:
        return pyopenjtalk.g2p(s, kana=True) or ""
    except Exception:
        return ""

@lru_cache(maxsize=4096)
def _g2p_tokens(s: str) -> list:
    try:
        txt = pyopenjtalk.g2p(s) or ""
        txt = txt.strip()
        return txt.split() if txt else []
    except Exception:
        return []

def _count_mora_from_kana(kana: str) -> int:
    k = re.sub(r"\s+", "", kana or "")
    m = 0
    for ch in k:
        if ch in _SMALL_KANA:
            continue
        if ch in _SOKUON or ch in _MORAIC_N or ch == _PROLONG:
            m += 1
        elif ("ぁ" <= ch <= "ゔ") or ("ァ" <= ch <= "ヴ"):
            m += 1
    return m

def _mora_from_tokens(toks: list) -> int:
    m = 0
    for t in toks:
        if t in _VOWELS or t in _SPECIAL:
            m += 1
    return m

def count_mora(japanese_text: str) -> int:
    s = _sanitize_jp(japanese_text)
    kana = _g2p_kana(s)
    if kana:
        return _count_mora_from_kana(kana)
    toks = _g2p_tokens(s)
    if toks:
        return _mora_from_tokens(toks)
    return _count_mora_from_kana(s)

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
    out = []
    for ch in s:
        cat0 = unicodedata.category(ch)[0]
        if cat0 in ("P", "S"):
            if ch in "・ー、。！？":
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)

def clean_line(s: str) -> str:
    s = s.strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)
    s = _strip_unicode_punct_symbol(s)
    s = re.sub(r"[0-9A-Za-z" + re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""") + r"]+", "", s)
    return s

def clean_haiku(raw: str) -> str:
    haiku_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("俳句:", "俳句の例:", "作者:", "季語:", "季節:", "構造:", "出力:")):
            continue
        # strip labels before cleaning
        line = re.sub(r"^(一行目|二行目|三行目)[：:]\s*", "", line)
        nrm = clean_line(line)
        if nrm:
            haiku_lines.append(nrm)
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
# Kigo detection
# -----------------------
def contains_kigo(text: str, kigo: str) -> bool:
    tn = normalize_jp(text)
    kn = normalize_jp(kigo)
    if not tn or not kn:
        return False
    if kn in tn:
        return True
    try:
        tph = _g2p_cached(tn)
        kph = _g2p_cached(kn)
        return kph in tph
    except Exception:
        return False

# -----------------------
# Prompt-bleed guards
# -----------------------
_LABEL_RX = re.compile(r"^((作者|季語|季節|構造|俳句|出力)|第?[一二三四五六七八九十0-9]+行目)[：: ]?")
def build_forbidden_set(texts: Iterable[str]) -> Set[str]:
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

_BANNED_IN_OUTPUT = {"俳句", "季語", "季節", "作者", "行目", "モーラ"}
_BANNED_ORDINAL_RX = re.compile(r"(第?[一二三四五六七八九十0-9]+)行(目)?")
def _has_banned_term(s: str) -> bool:
    return any(b in s for b in _BANNED_IN_OUTPUT) or bool(_BANNED_ORDINAL_RX.search(s))

# Raw line-wise meta/ordinal check on continuation (before cleaning)
def _has_banned_term_linewise(raw_text: str) -> bool:
    for ln in (raw_text or "").splitlines():
        raw = ln.strip()
        if not raw:
            continue
        if _LABEL_RX.search(raw):  # skip explicit labels
            continue
        if any(b in raw for b in _BANNED_IN_OUTPUT) or _BANNED_ORDINAL_RX.search(raw):
            return True
    return False

# -----------------------
# Metrics
# -----------------------
def strip_ascii_punct(s: str) -> str:
    s = re.sub(r"[A-Za-z" + re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""") + r"]+", "", s)
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
    hyp = list(candidate.replace("\n", ""))
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
# BLEU ignoring kigo
# -----------------------
def _strip_kigo_for_bleu(text: str, kigo: str) -> str:
    t = normalize_jp(text)
    k = normalize_jp(kigo)
    return t.replace(k, "")

def _max_bleu_ignore_kigo_early(candidate: str, refs: List[str], kigo: str, stop_at: float) -> float:
    smoothie = SmoothingFunction().method1
    hyp = list(_strip_kigo_for_bleu(candidate, kigo).replace("\n",""))
    mx = 0.0
    for ref in refs:
        sc = sentence_bleu(
            [list(_strip_kigo_for_bleu(ref, kigo).replace("\n",""))],
            hyp, smoothing_function=smoothie, weights=(1.0,)
        )
        if sc > mx:
            mx = sc
            if mx >= stop_at:
                break
    return mx

# -----------------------
# Haiku style helpers
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
    return f"{ex['ref_haiku']}\n———\n"

def build_prompt_gemma(target, examples):
    if examples is None or examples.empty:
        return ""
    blocks = [build_example_block_gemma(ex) for _, ex in examples.iterrows()]
    return "".join(blocks)

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

    if not exs.empty:
        def _row_has_kireji(row): return any(k in str(row["ref_haiku"]) for k in _KIREJI)
        exs = pd.concat([exs[exs.apply(_row_has_kireji, axis=1)],
                         exs[~exs.apply(_row_has_kireji, axis=1)]], ignore_index=True)
    return exs

# -----------------------
# Helpers
# -----------------------
def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def _decode_continuation(out_ids: torch.Tensor, in_ids: torch.Tensor, tok: AutoTokenizer) -> str:
    if out_ids.dim() == 1:
        out_ids = out_ids.unsqueeze(0)
    new_ids = out_ids[:, in_ids.shape[1]:]
    return tok.decode(new_ids[0], skip_special_tokens=False)

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.perf_counter()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    eos_id = tok.eos_token_id

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
    df = df_h.merge(df_k, on=("haiku_id"), how="left", suffixes=("_haiku", "_kigo"))
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

    skip_counters = {"accepted": 0}
    stats, raw_outputs = [], []

    targets = df.sample(args.num_targets, random_state=args.seed)

    for t_idx, (_, tgt) in enumerate(targets.iterrows(), start=1):
        print("\n" + "="*70)
        print(f"[TARGET {t_idx}] kigo={tgt['word']} season={tgt['season']} author={tgt['author']}")
        print("="*70, flush=True)

        exs = retrieve_examples(tgt, df_masters, few_shot_k=args.few_shot_k)
        example_texts = [ex["ref_haiku"] for _, ex in exs.iterrows()]

        # Instruction
        instruction = (
            "あなたは日本語の俳人です。説明は書かず、俳句のみを出力してください。\n"
            "出力は三行のみ：一行目5モーラ、二行目7モーラ、三行目5モーラ。\n"
            "数字・英字・記号・注釈・作者名・ラベルは禁止。自然に切れ字を用いてください。\n"
            f"季節は「{tgt['season']}」。季語「{tgt['word']}」を一度だけ含めてください。\n"
            "以下の作例は参考のみ。語句・比喩・構図を繰り返さないでください。\n"
            "行頭に「一行目:」「二行目:」「三行目:」を付け、その後に句のみを書いてください。\n\n"
        )

        examples_block = build_prompt_gemma(tgt, exs)
        base_prompt = instruction
        if examples_block.strip():
            base_prompt += "【作例】\n" + examples_block + "\n"
        base_prompt += "【新作】\n"

        # Style scorer (used to rank fallbacks)
        near_pool = df[(df["season"] == tgt["season"]) & (df["kigo_id"] == tgt["kigo_id"])]
        near_refs = near_pool["ref_haiku"].dropna().tolist()
        if len(near_refs) > 50:
            near_refs = pd.Series(near_refs).sample(50, random_state=args.seed).tolist()

        def style_score(c):
            sim_ex  = max_bleu_vs_list(c, example_texts) if example_texts else 0.0
            sim_loc = max_bleu_vs_list(c, near_refs)     if near_refs else 0.0
            meter   = 1.0 if is_575(c) else 0.0
            has_kigo= 1.0 if contains_kigo(c, tgt["word"]) else 0.0
            kireji_bonus = 0.35 if _has_kireji(c) else 0.0
            kanji_bonus  = min(_kanji_ratio(c), 0.35)
            katakana_pen = min(_katakana_ratio(c), 0.30)
            repeat_pen   = _repetition_penalty(c)
            if _contains_bad_style(c) or _has_banned_term(c):
                return -1e9
            return (
                0.8*meter + 0.4*has_kigo
                + kireji_bonus + 0.5*kanji_bonus
                - 0.8*katakana_pen - 0.6*repeat_pen
                - 1.0*max(sim_ex, sim_loc)
            )

        # Nudge state
        gen_prompt = base_prompt
        nudged = False

        # Acceptance buckets (for fallback ladder)
        best_strict = None       # 5-7-5 + kigo + below BLEU threshold
        best_575_kigo = None     # 5-7-5 + kigo (even if too similar)
        best_575 = None          # 5-7-5 (kigo optional)
        best_kigo = None         # kigo present (meter optional)
        best_any = None          # anything non-empty
        sc_strict = sc_575k = sc_575 = sc_kigo = sc_any = -1e9

        gl_try = 0
        while gl_try < args.max_haiku_attempts:
            gl_try += 1

            # Build inputs (chat template if available)
            use_template = hasattr(tok, "apply_chat_template")
            if use_template:
                try:
                    messages = [
                        {"role": "system", "content": "あなたは簡潔に俳句を作る詩人です。余計な説明は書きません。"},
                        {"role": "user",   "content": gen_prompt}
                    ]
                    prompt_text = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inp = tok(prompt_text, return_tensors="pt").to(device)
                except Exception:
                    inp = tok(gen_prompt, return_tensors="pt").to(device)
            else:
                inp = tok(gen_prompt, return_tensors="pt").to(device)

            # Decode
            out = model.generate(
                **inp,
                max_new_tokens=40,
                min_new_tokens=12,
                do_sample=True,
                temperature=0.90,
                top_p=0.94,
                top_k=20,
                repetition_penalty=1.02,
                eos_token_id=eos_id,
                pad_token_id=tok.pad_token_id,
            )

            full_txt = tok.decode(out[0], skip_special_tokens=False)
            new_only = _decode_continuation(out, inp["input_ids"], tok)
            new_only = "\n".join([ln for ln in new_only.splitlines() if ln.strip()])

            # Raw-line meta/ordinal guard
            if _has_banned_term_linewise(new_only):
                print("  -> reject: meta/ordinal term found in raw non-label line")
                continue

            cand = clean_haiku(new_only)

            print(f"\n[ONE-SHOT DEBUG] try {gl_try}")
            print(f"  raw[:160]= {repr(full_txt[:160])}")
            print(f"  new_only[:160]= {repr(new_only[:160])}")
            print(f"  cand=\n{cand}")

            if not cand.strip():
                print("  -> reject: empty after cleaning")
                continue
            if _has_banned_term(cand):
                print("  -> reject: banned meta/label term (post-clean)")
                continue

            # track best_any early
            sc = style_score(cand)
            if cand and sc > sc_any:
                best_any, sc_any = cand, sc

            m575 = is_575(cand)
            has_k = contains_kigo(cand, tgt["word"])

            # Optional one-time nudge if we got meter but missing kigo
            if m575 and not has_k and not nudged:
                nudged = True
                gen_prompt = (
                    base_prompt
                    + f"【注意】まだ季語「{tgt['word']}」が含まれていません。句のどこかに一度だけ自然に「{tgt['word']}」を入れてください。\n"
                    "【新作】\n"
                )
                print("  -> kigo nudge applied; retrying")
                # still keep this candidate as a fallback 5-7-5
                if sc > sc_575:
                    best_575, sc_575 = cand, sc
                continue

            # Fill buckets
            if m575 and has_k:
                # strict similarity gate
                b_ex  = _max_bleu_ignore_kigo_early(cand, example_texts, tgt["word"], args.bleu_threshold) if example_texts else 0.0
                b_loc = _max_bleu_ignore_kigo_early(cand, near_refs,     tgt["word"], args.bleu_threshold) if near_refs else 0.0
                too_sim = max(b_ex, b_loc) >= args.bleu_threshold

                if not too_sim and sc > sc_strict:
                    best_strict, sc_strict = cand, sc
                    print("  -> ACCEPT strict (5-7-5 + kigo + below similarity)")
                    break  # we can finish early with the best class

                if sc > sc_575k:
                    best_575_kigo, sc_575k = cand, sc
            elif m575:
                if sc > sc_575:
                    best_575, sc_575 = cand, sc
            elif has_k:
                if sc > sc_kigo:
                    best_kigo, sc_kigo = cand, sc
            # best_any already tracked

        # Fallback ladder (A -> E)
        if best_strict:
            best_haiku = best_strict
            fb_used = "strict"
        elif best_575_kigo:
            best_haiku = best_575_kigo
            fb_used = "5-7-5+kigo"
        elif best_575:
            best_haiku = best_575
            fb_used = "5-7-5"
        elif best_kigo:
            best_haiku = best_kigo
            fb_used = "kigo-only"
        else:
            best_haiku = best_any if best_any else ""
            fb_used = "any"

        if not best_haiku.strip():
            print("[ONE-SHOT] No valid haiku found; leaving empty string (should be rare).")
        else:
            print(f"[FALLBACK] selected bucket: {fb_used}")

        skip_counters["accepted"] += 1
        print("\n[RESULT] Best candidate (cleaned):\n" + best_haiku)
        print(f"  is_575={is_575(best_haiku)}")
        print(f"  contains_kigo={contains_kigo(best_haiku, tgt['word'])}")

        # Final metrics
        if best_haiku.strip():
            avg_ppl = compute_perplexity(model, tok, best_haiku, device)
            bleu_score = compute_bleu(tgt["ref_haiku"], best_haiku)
        else:
            avg_ppl = float("nan")
            bleu_score = 0.0

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
            "fallback_bucket": fb_used,
        })
        raw_outputs.append({"prompt_kigo": tgt["word"], "cleaned_haiku": best_haiku})

    # Save
    pd.DataFrame(stats).to_csv(os.path.join(args.output_dir, "iterative_eval.csv"),
                               index=False, encoding="utf-8-sig")
    pd.DataFrame(raw_outputs).to_csv(os.path.join(args.output_dir, "iterative_raw_outputs.csv"),
                                     index=False, encoding="utf-8-sig")

    elapsed = time.perf_counter() - start_time
    print("\n=== DEBUG SUMMARY ===")
    print(skip_counters)
    print(f"\n=== RUNTIME ===\nElapsed: {elapsed:.2f} s ({_fmt_hms(elapsed)})")
    try:
        with open(os.path.join(args.output_dir, "run_time.txt"), "w", encoding="utf-8") as f:
            f.write(f"{elapsed:.2f} seconds ({_fmt_hms(elapsed)})\n")
    except Exception as e:
        print(f"[WARN] Failed to write run_time.txt: {e}")
    print("Done. One-shot results saved.", flush=True)

if __name__ == "__main__":
    main()
