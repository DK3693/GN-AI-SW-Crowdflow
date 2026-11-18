# 이 파일이 하는 일: 운영 지침을 읽고, 승인한 문장을 더하고, 상황에 가까운 조항을 고르는 부분(guideline.py)을 확인한다.
#   글을 숫자로 바꾸는 모델(임베딩)은 부르지 않고, 시험용 숫자를 대신 넣는다. 실제 모델로 돌린 결과는 E-71에 따로 있다.
# 해당 기능: 핵심 기능 2·4의 판단을 돕는 부분 (운영 지침 읽기), 기억
# 해당 기준: C-32, C-37, C-38
# 근거: D-64, E-68
# 출처: AI 코딩 도구로 작성
import numpy as np

from crowdflow.guideline import Clause, GuidelineIndex, add_learned, clear_learned, load_clauses

CFG = {"base": "base.md", "learned": "learned.jsonl", "embed_model": "none", "top_k": 3, "timeout_seconds": 1}


def write_base(tmp_path, n=2):
    lines = ["# 운영 지침 (시험용)", "", "설명 글은 조항이 아니다."] + [f"- 기본 조항 {i}" for i in range(1, n + 1)]
    (tmp_path / "base.md").write_text("\n".join(lines), encoding="utf-8")


def test_base_and_approved_sentences_are_read_in_order(tmp_path):
    write_base(tmp_path)
    assert [(c.id, c.text, c.source) for c in load_clauses(tmp_path, CFG)] == [("G-01", "기본 조항 1", "기본 지침"), ("G-02", "기본 조항 2", "기본 지침")]
    n = add_learned(tmp_path, CFG, ["새 문장", " 새 문장 ", "", "기본 조항 1"], "1일차 정리에서 제안 · 관리자 승인")
    assert n == 1  # 같은 문장, 빈 글, 이미 있는 조항은 넣지 않는다
    last = load_clauses(tmp_path, CFG)[-1]
    assert (last.id, last.text, last.source) == ("L-01", "새 문장", "1일차 정리에서 제안 · 관리자 승인")
    assert add_learned(tmp_path, CFG, ["새 문장"], "다시") == 0
    clear_learned(tmp_path, CFG)
    assert [c.id for c in load_clauses(tmp_path, CFG)] == ["G-01", "G-02"]  # 기본 지침은 그대로


def test_few_clauses_are_all_read_without_the_embedding_model():
    index = GuidelineIndex(CFG, "http://127.0.0.1:9")  # 닿지 않는 주소: 부르면 오류가 난다
    clauses = [Clause(f"G-{i:02d}", f"조항 {i}", "기본 지침") for i in range(1, 4)]
    found = index.find("아무 상황", clauses)
    assert found["mode"] == "전부 읽음" and found["total"] == 3 and [c["id"] for c in found["clauses"]] == ["G-01", "G-02", "G-03"]
    assert all(c["score"] is None for c in found["clauses"])


def test_many_clauses_are_ranked_by_closeness(monkeypatch):
    clauses = [Clause(f"G-{i:02d}", f"조항 {i}", "기본 지침") for i in range(1, 6)]
    fake = {"물음": [1.0, 0.0], "조항 1": [0.0, 1.0], "조항 2": [1.0, 0.1], "조항 3": [0.7, 0.7], "조항 4": [-1.0, 0.0], "조항 5": [1.0, 0.0]}

    def embed(self, texts):
        v = np.array([fake[t] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(GuidelineIndex, "_embed", embed)
    found = GuidelineIndex(CFG, "http://127.0.0.1:9").find("물음", clauses)
    assert found["mode"] == "찾아 읽음" and found["total"] == 5
    assert [c["id"] for c in found["clauses"]] == ["G-05", "G-02", "G-03"]  # 가까운 순서로 top_k개
    assert found["clauses"][0]["score"] == 1.0 and found["clauses"][0]["text"] == "조항 5"


def test_unparsed_ai_answer_raises_so_the_flow_falls_back_to_rules():
    # CrewAI가 답을 형식에 맞추지 못하고 빈 값을 주면, 오류로 바꿔 흐름이 규칙으로 대신하게 한다 (E-72)
    import types

    import pytest

    from crowdflow.agents import _parsed

    assert _parsed(types.SimpleNamespace(pydantic="ok")) == "ok"
    with pytest.raises(ValueError):
        _parsed(types.SimpleNamespace(pydantic=None))
