"""SinLlama (local GPU) helpers shared by llm/run_sinllama.py and rule_detection/run_sinllama.py.

SinLlama is a base (non-chat) Llama-3-8B model, so answers are read from
next-token probabilities instead of free generation.

Needs torch + transformers (+ bitsandbytes for 4-bit) with CUDA; anish_project's
venv has them (PYTHONPATH=~/anish_project/.python_pkgs ~/anish_project/venv/bin/python).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re

DEFAULT_MODEL_DIR = "/home/dilusha/anish_project/predictions/sinhala articles/models/SinLlama-Llama-3-8B-Merged"
YES_WORDS = (" YES", " Yes", " yes")
NO_WORDS = (" NO", " No", " no")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", default=os.getenv("SINLLAMA_MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit quantisation (less VRAM)")
    parser.add_argument("--max-input-tokens", type=int, default=8000)
    parser.add_argument(
        "--margin", type=float, default=0.5,
        help="per_rule: calibrated log-odds needed for VIOLATION (0.5 = odds x1.65 vs an empty article)",
    )


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sentences(content: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", content) if len(s.strip()) >= 10]


def snap_to_sentence(generated: str, content: str, min_prefix: int = 12) -> str:
    """The article sentence the generated quote starts with, or '' if none matches."""
    best, best_len = "", 0
    for sentence in _sentences(content):
        n = 0
        for a, b in zip(generated, sentence):
            if a != b:
                break
            n += 1
        if n >= min(min_prefix, len(sentence)) and n > best_len:
            best, best_len = sentence, n
    return best


class SinLlama:
    def __init__(self, model_dir: str, *, load_in_4bit: bool = False, max_input_tokens: int = 8000):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        if not torch.cuda.is_available():
            raise SystemExit("SinLlama needs a CUDA GPU")
        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        load_kwargs = {"device_map": "cuda", "dtype": torch.float16, "low_cpu_mem_usage": True}
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        print(f"loading {model_dir} ({'4-bit' if load_in_4bit else 'fp16'})", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs).eval()
        self.yes_ids = self._first_token_ids(YES_WORDS)
        self.no_ids = self._first_token_ids(NO_WORDS)
        self._rule_baseline: dict[str, float] = {}

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SinLlama":
        return cls(args.model_dir, load_in_4bit=args.load_in_4bit, max_input_tokens=args.max_input_tokens)

    def _first_token_ids(self, words: tuple[str, ...]) -> list[int]:
        return sorted({self.tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in words})

    def encode(self, text: str):
        return self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_input_tokens
        ).to(self.model.device)

    def score_yes_no(self, text: str) -> tuple[float, float]:
        """(P(YES) / (P(YES) + P(NO)), P(YES) + P(NO)) for the token after text."""
        with self.torch.inference_mode():
            logits = self.model(**self.encode(text), logits_to_keep=1).logits[0, -1].float()
        probs = self.torch.softmax(logits, dim=-1)
        p_yes = probs[self.yes_ids].sum().item()
        p_no = probs[self.no_ids].sum().item()
        return (p_yes / (p_yes + p_no) if p_yes + p_no > 0 else 0.5), p_yes + p_no

    def choose(self, text: str, candidates: list[str]) -> dict[str, float]:
        """Probability of each candidate continuation of text, normalised over the candidates."""
        torch = self.torch
        with torch.inference_mode():
            inputs = self.encode(text)
            prompt_len = inputs["input_ids"].shape[1]
            out = self.model(**inputs, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            scores = {}
            for cand in candidates:
                ids = self.tokenizer(cand, add_special_tokens=False)["input_ids"]
                total = first[ids[0]].item()
                if len(ids) > 1:
                    rest = torch.tensor([ids[:-1]], device=self.model.device)
                    logits = self.model(input_ids=rest, past_key_values=past, use_cache=True).logits[0].float()
                    logp = torch.log_softmax(logits, dim=-1)
                    total += sum(logp[i, ids[i + 1]].item() for i in range(len(ids) - 1))
                    past.crop(prompt_len)
                scores[cand] = total
        top = max(scores.values())
        exp = {c: math.exp(s - top) for c, s in scores.items()}
        norm = sum(exp.values())
        return {c: v / norm for c, v in exp.items()}

    def score_continuations(self, prefix: str, suffixes: list[str]) -> list[float]:
        """P(YES) / (P(YES) + P(NO)) after prefix + suffix, for each suffix (prefix encoded once)."""
        torch = self.torch
        inputs = self.encode(prefix)
        prefix_len = inputs["input_ids"].shape[1]
        if prefix_len >= self.max_input_tokens:
            print(f"warning: prompt truncated to {self.max_input_tokens} tokens", flush=True)
        out = []
        with torch.inference_mode():
            past = self.model(**inputs, use_cache=True, logits_to_keep=1).past_key_values
            for suffix in suffixes:
                ids = self.tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
                logits = self.model(
                    input_ids=ids.to(self.model.device), past_key_values=past, use_cache=True, logits_to_keep=1
                ).logits[0, -1].float()
                past.crop(prefix_len)
                probs = torch.softmax(logits, dim=-1)
                p_yes = probs[self.yes_ids].sum().item()
                p_no = probs[self.no_ids].sum().item()
                out.append(p_yes / (p_yes + p_no) if p_yes + p_no > 0 else 0.5)
        return out

    def quote(self, text: str, max_new_tokens: int = 80) -> str:
        """Greedy continuation of text, first line only."""
        with self.torch.inference_mode():
            inputs = self.encode(text)
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated.strip().split("\n")[0].strip().strip("\"'“”")

    def per_rule(self, article: dict, prompt, margin: float, *, quote_evidence: bool = True) -> tuple[str, float, dict]:
        """Score every rule of prompt.RULE_IDS separately, calibrated against an empty article.

        Returns (raw JSON in the per_rule format, max calibrated P(YES), {"rule_p", "baseline_p"}).
        """
        if not self._rule_baseline:
            empty = {"heading": "N/A", "content": "N/A"}
            self._rule_baseline = {r: self.score_yes_no(prompt.build_rule(empty, r))[0] for r in prompt.RULE_IDS}
        baseline = self._rule_baseline
        rules, rule_p = {}, {}
        for rule_id in prompt.RULE_IDS:
            text = prompt.build_rule(article, rule_id)
            p, mass = self.score_yes_no(text)
            shift = logit(p) - logit(baseline[rule_id])
            rule_p[rule_id] = 1 / (1 + math.exp(-shift))
            violated = shift >= margin
            evidence = ""
            if violated and quote_evidence:
                evidence = snap_to_sentence(self.quote(text + prompt.EVIDENCE_SUFFIX), article.get("content") or "")
            rules[rule_id] = {
                "verdict": "VIOLATION" if violated else "NO_VIOLATION",
                "evidence": evidence,
                "explanation": (
                    f"P(YES)={p:.3f} vs {baseline[rule_id]:.3f} for an empty article "
                    f"(odds x{math.exp(shift):.2f}); P(YES)+P(NO)={mass:.3f}"
                ),
            }
        overall = "YES" if any(r["verdict"] == "VIOLATION" for r in rules.values()) else "NO"
        raw = json.dumps({"rules": rules, "violation": overall}, ensure_ascii=False)
        return raw, max(rule_p.values()), {"rule_p": rule_p, "baseline_p": dict(baseline)}
