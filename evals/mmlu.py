#!/usr/bin/env python3
# Drives the MMLU eval inside the SGLang image.
#
# Why this file exists: in the image this lane serves, `python3 -m
# sglang.test.run_eval --eval-name mmlu` no longer runs anything itself. It
# shells out to an external `sgl-eval` binary (sgl-project/sgl-eval, first
# published on PyPI 2026-09-12) that the image does not ship, so it dies with
# FileNotFoundError before touching the server. Measured 2026-09-18.
#
# This file then called the MMLUEval class the image still carried, until SGLang
# removed it (sgl-project/sglang#41215, 2026-09-25: no simple_eval_mmlu in v0.5.21,
# the 27B lane's image, nor in the flash lane's nightly), and the import failed in
# both pinned images (found 2026-10-20). The class is now below, as SGLang v0.5.19
# had it, with the sampler and the helpers the images still ship: the same dataset,
# the same questions drawn with the same seed, the same prompt and answer pattern.
# Usage: see evals/README.md (it runs inside the serving image, not on the host).
import random
import re
import sys
import time
from typing import Optional

import pandas

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    HTML_JINJA,
    ChatCompletionSampler,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    format_multichoice_question,
)

URL = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"

# From here to the driver below: SGLang v0.5.19, python/sglang/test/simple_eval_mmlu.py
# (commit 0bcd822377da7b5718e674eaf9c870d349424dd1), Copyright 2023-2024 SGLang Team, Apache
# License 2.0 (LICENSE-SGLANG, beside this file), adapted there from openai/simple-evals,
# Copyright (c) 2024 OpenAI, MIT (LICENSE-SIMPLE-EVALS). Unchanged but for its imports, above,
# and its docstring, now the three lines below.
#
# Measuring Massive Multitask Language Understanding
# Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
# https://arxiv.org/abs/2009.03300

subject2category = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


class MMLUEval(Eval):
    def __init__(self, filename: str, num_examples: Optional[int], num_threads: int):
        if "://" in filename:
            df = pandas.read_csv(filename, storage_options={"timeout": 30})
        else:
            df = pandas.read_csv(filename)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(row), role="user"
                )
            ]
            response_text = sampler(prompt_messages)
            response_text = response_text or ""
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = subject2category.get(row["Subject"], "other")
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)


# The driver, as before.
model = sys.argv[1]
num_examples = int(sys.argv[2])
num_threads = int(sys.argv[3])
base_url = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:30000/v1"

sampler = ChatCompletionSampler(
    model=model,
    max_tokens=2048,
    top_p=1.0,
    base_url=base_url,
    temperature=0.0,
    record_meta_info=True,
)
eval_obj = MMLUEval(URL, num_examples, num_threads)

tic = time.perf_counter()
result = eval_obj(sampler)
latency = time.perf_counter() - tic

print(f"Total latency: {latency:.3f} s")
print(f"examples: {num_examples}, threads: {num_threads}")
print(f"'score': {result.score}")
print(f"metrics: {dict(result.metrics or {})}")
