# 이 파일이 하는 일: 운영 지침을 읽어 오고, 지금 상황에 가까운 조항을 찾아 준다 (RAG의 "찾아오기").
#   지침은 두 군데에 있다: 팀이 준비한 기본 지침(knowledge/operations_guideline.md)과,
#   하루 정리에서 관리자가 승인해 더한 문장(settings.yaml의 guideline.learned, 실행하면서 쌓인다).
#   조항이 적을 때는 전부 읽고, 많아지면 글을 숫자로 바꾸는 모델(임베딩)로 상황에 가장 가까운 것만 골라 읽는다.
#   찾아온 조항을 읽고 어느 선택지가 맞는지 말하는 것은 지침 담당(agents.py)이 한다.
# 해당 기능: 핵심 기능 2·4의 판단을 돕는 부분. 기억(지난 정리를 다음 날에 씀)
# 해당 기준: C-23, C-30, C-37 (도구·자료 연동), C-38 (기억)
# 근거: D-27 (행사 전 정보는 팀이 준비하고 프로그램이 읽음), D-64 (하루 정리 → 승인한 문장을 지침에 더함 → 토론에서 읽음),
#   E-68 (임베딩 모델이 필요함, Ollama의 /api/embed, CrewAI Knowledge의 찾아오는 개수 기본값 3), E-71 (bge-m3로 돌려 본 결과)
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class Clause:
    id: str  # 조항 번호. 기본 지침은 G-01부터, 승인해 더한 문장은 L-01부터
    text: str
    source: str  # 어디서 온 조항인지 (기본 지침 / 며칠차 정리에서 승인)


def load_clauses(root: Path, cfg: dict) -> list[Clause]:
    """기본 지침과, 관리자가 승인해 더한 문장을 차례로 읽는다."""
    out: list[Clause] = []
    base = root / cfg["base"]
    if base.exists():
        lines = [line[2:].strip() for line in base.read_text(encoding="utf-8").splitlines() if line.startswith("- ")]
        out += [Clause(f"G-{i:02d}", text, "기본 지침") for i, text in enumerate(lines, 1)]
    learned = root / cfg["learned"]
    if learned.exists():
        rows = [json.loads(line) for line in learned.read_text(encoding="utf-8").splitlines() if line.strip()]
        out += [Clause(f"L-{i:02d}", r["text"], r["source"]) for i, r in enumerate(rows, 1)]
    return out


def add_learned(root: Path, cfg: dict, texts: list[str], source: str) -> int:
    """관리자가 승인한 문장을 지침에 더한다. 같은 문장이 이미 있으면 넣지 않는다. 더한 개수를 돌려준다."""
    have = {c.text for c in load_clauses(root, cfg)}
    new = [t.strip() for t in texts if t.strip() and t.strip() not in have]
    path = root / cfg["learned"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for t in dict.fromkeys(new):
            f.write(json.dumps({"text": t, "source": source, "added": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False) + "\n")
    return len(dict.fromkeys(new))


def clear_learned(root: Path, cfg: dict) -> None:
    """승인해 더한 문장을 모두 지운다 (기본 지침은 그대로)."""
    (root / cfg["learned"]).unlink(missing_ok=True)


class GuidelineIndex:
    """조항을 숫자(임베딩)로 바꿔 두고, 상황 글에 가장 가까운 조항을 찾는다."""

    def __init__(self, cfg: dict, base_url: str):
        self.cfg, self.base_url = cfg, base_url
        self._vectors: dict[str, np.ndarray] = {}  # 글 → 숫자. 같은 글을 두 번 바꾸지 않는다

    def _embed(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if todo:
            body = json.dumps({"model": self.cfg["embed_model"], "input": todo}).encode()
            req = urllib.request.Request(self.base_url + "/api/embed", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.cfg["timeout_seconds"]) as r:
                vectors = np.asarray(json.loads(r.read().decode())["embeddings"], dtype=np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
            self._vectors.update(zip(todo, vectors))
        return np.stack([self._vectors[t] for t in texts])

    def find(self, query: str, clauses: list[Clause]) -> dict:
        """상황(query)에 맞춰 읽을 조항을 고른다.

        조항이 top_k개 이하이면 전부 읽는다(찾을 것이 없다). 그보다 많으면 가장 가까운 top_k개만 찾아 읽는다.
        돌려주는 값: {"mode": "전부 읽음" | "찾아 읽음", "total": 조항 수, "clauses": [{id, text, source, score}]}
        """
        k = self.cfg["top_k"]
        if len(clauses) <= k:
            return {"mode": "전부 읽음", "total": len(clauses), "clauses": [dict(asdict(c), score=None) for c in clauses]}
        vectors = self._embed([c.text for c in clauses])
        scores = vectors @ self._embed([query])[0]
        best = np.argsort(-scores)[:k]
        return {"mode": "찾아 읽음", "total": len(clauses), "clauses": [dict(asdict(clauses[i]), score=round(float(scores[i]), 3)) for i in best]}
