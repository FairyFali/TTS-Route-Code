"""The trained proxy reward model, wrapped as an online verifier for the TTS planner.

Why this replaces agreement
---------------------------
Measured on best-route (600 pools x 16 fixed-width samples):

    plurality agreement  : 0.718   (width-1 = 0.715, so widening buys +0.002)
    ourRM best-of-8      : 0.741
    oracle any-of-16     : 0.925

Agreement over exact-string clusters is near-useless on free-form text -- a 16-sample
pool holds ~9.2 distinct answers, so the "plurality" wins with 2 votes and is close to a
random pick. Its share also *decays* as ~c/n with width, so widening moves the signal
away from any fixed threshold. A scalar verifier does not have either problem: it gives a
usable ordering at every width, and it is what the read-out should use anyway.

Cost accounting
---------------
The verifier is not free and is charged in the same 1B-call units as generation. It is an
encoder: one prefill pass over (prompt + response), no decode. So

    cost = cost_prefill(n_p + n_d, deberta_spec) / unit_q

which lands at roughly 0.2-0.4 units per scored response against a ~1 unit 1B
generation -- cheap, but not nothing, and a search that scores every sample pays it on
every sample.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from experiments_query.budget import ModelSpec, cost_prefill

logger = logging.getLogger("verifier")

# DeBERTa-v3-large: ~435M non-embedding, hidden 1024, 24 layers.
DEBERTA_SPEC = ModelSpec("deberta-v3-large-rm", params=4.35e8, hidden=1024, layers=24)

DEFAULT_PATH = "experiments_query/results/bestroute_rm/models/checkpoint-best"


class Verifier:
    """Scores (prompt, response) pairs with the trained proxy RM.

    The text format MUST match what the model was trained on in make_pairs.py --
    "Human: {prompt} Assistant: {response}" -- or the scores silently degrade.
    """

    def __init__(self, path: str = DEFAULT_PATH, device: Optional[str] = None,
                 batch_size: int = 16, max_length: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=1, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_scored = 0
        logger.info("verifier loaded from %s on %s", path, self.device)

    @torch.no_grad()
    def score(self, prompt: str, responses: List[str]) -> List[float]:
        if not responses:
            return []
        texts = [f"Human: {prompt} Assistant: {r}" for r in responses]
        out: List[float] = []
        for b in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[b:b + self.batch_size], return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_length).to(self.device)
            out += [float(x) for x in self.model(**enc).logits[:, 0].float().cpu()]
        self.n_scored += len(out)
        return out

    def cost_units(self, n_prompt_tok: int, n_resp_tok: int, unit_flops: float) -> float:
        """FLOPs of scoring ONE response, in units of one 1B call on this query."""
        if unit_flops <= 0:
            return 0.0
        return cost_prefill(float(n_prompt_tok + n_resp_tok), DEBERTA_SPEC) / unit_flops
