# 이 파일이 하는 일: CrewAI로 담당(Agent) 여섯을 만든다. 글을 읽어 조건으로 바꾸는 소통 담당, 선택지를 두고 토론하는
#   효율 담당과 안전 담당, 토론을 듣고 정하는 계획 담당, 결과를 보고 끝낼지 정하고 하루 정리의 글을 쓰는 확인 담당,
#   그리고 운영 지침에서 찾아온 조항이 지금 선택에 해당하는지 가려 읽어 주는 지침 담당("지침 읽기"를 켰을 때만 일한다).
#   담당은 사람을 세거나 조건을 어기는 배치를 만들지 않는다. 계산 도구가 만든 선택지 안에서만 고른다.
# 해당 기능: 핵심 기능 2 (말 이해하기), 4 (실행과 재확인의 판단 부분)
# 해당 기준: C-06, C-09, C-12, C-21, C-22, C-29, C-34, C-35, C-36, C-44
# 근거: D-21 (CrewAI로 담당을 나눔), D-05 (꼭 넣는 두 장면), D-07 (다친 사람은 따로 알림), D-08,
#   D-48 (글 읽는 AI: 이 컴퓨터의 Ollama, qwen3.8:27b), D-49 (담당 구성: 계획 담당 안에서 효율·안전 담당이 토론),
#   D-59 (효율·안전 담당이 서로의 글을 보지 않고 따로 의견을 냄. 되풀이를 본 기록은 E-63, 고친 뒤의 확인은 E-66),
#   D-63 (안전 담당의 눈을 "붐비는 구역이 확실히 풀리게, 여유 있게"로 고침. 두 담당의 답이 갈리는지 본 기록은 E-70),
#   D-64 (지침 담당과 하루 정리. 지침에는 관리자가 승인한 문장만 들어감. 돌려 본 결과는 E-71),
#   E-35 (CrewAI 공식 문서: Agent, Task, Crew, kickoff, response_format, Ollama 연결), E-51 (모델 시험)
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Literal

os.environ.setdefault("OTEL_SDK_DISABLED", "true")  # CrewAI가 사용 통계를 밖으로 보내지 않게 한다
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from pydantic import BaseModel, Field  # noqa: E402


class ManagerInstruction(BaseModel):
    """관리자의 글을 읽은 결과."""

    kind: Literal["set_min_staff", "other", "unclear"] = Field(description="최소 인원 지시면 set_min_staff, 다른 내용이면 other, 뜻을 알 수 없으면 unclear")
    zone: Literal["A", "B", "C", "D", "none"] = Field(description="지시가 가리키는 구역. 없으면 none")
    min_staff: int = Field(description="그 구역에 남겨야 할 최소 인원. 해당 없으면 0")
    restated: str = Field(description="읽은 내용을 관리자에게 확인받을 한 문장")


class StaffReply(BaseModel):
    """요원의 답을 읽은 결과."""

    can_go: Literal["yes", "no", "unclear"] = Field(description="지금 바로 갈 수 있으면 yes, 못 가면 no, 알 수 없으면 unclear")
    reason: str = Field(description="못 가는 이유를 짧게. 없으면 빈 글")
    injury_mentioned: bool = Field(description="다친 사람, 환자, 쓰러진 사람 이야기가 있으면 true")


class Opinion(BaseModel):
    """토론에서 한 담당이 낸 의견."""

    preferred_option: int = Field(description="고른 선택지의 번호")
    point: str = Field(description="고른 이유의 핵심을 한 줄로. 25자 안팎. 관리자가 한눈에 읽는 글")
    argument: str = Field(description="그렇게 고른 이유. 한두 문장. 선택지에 적힌 숫자를 근거로")


class PlanDecision(BaseModel):
    """계획 담당의 결정."""

    chosen_option: int = Field(description="최종으로 고른 선택지의 번호")
    headline: str = Field(description="결정의 핵심을 한 줄로. 30자 안팎. 관리자가 한눈에 읽는 글. 예: 예비 요원을 남기려고 가까운 요원을 보냄")
    reason: str = Field(description="두 담당의 의견을 어떻게 따져 정했는지. 한두 문장")
    steps: list[str] = Field(description="이 뒤에 할 일을 차례대로 세 개. 하나에 15자 안팎")


class Verdict(BaseModel):
    """확인 담당의 판단."""

    decision: Literal["close", "send_more", "keep_watching", "notify_manager"] = Field(
        description="풀렸으면 close, 사람을 더 보내야 하면 send_more, 더 지켜보면 keep_watching, 사람이 정해야 하면 notify_manager")
    reason: str = Field(description="그렇게 판단한 이유. 한두 문장")


class GuideNote(BaseModel):
    """지침 담당이 찾아온 조항을 읽고 낸 말."""

    applies: bool = Field(description="찾아온 조항 가운데 지금 상황과 선택지 고르기에 해당하는 것이 있으면 true")
    clause_ids: list[str] = Field(description="해당하는 조항의 번호. 없으면 빈 목록")
    preferred_option: int = Field(description="그 조항대로라면 맞는 선택지의 번호. 해당하는 조항이 없으면 0")
    point: str = Field(description="핵심을 한 줄로. 25자 안팎")
    argument: str = Field(description="어느 조항이 왜 해당하는지 한두 문장. 조항에 적힌 말만 쓴다")


class Review(BaseModel):
    """하루치 운영 기록을 정리한 글 (확인 담당). 숫자는 기록에서 코드가 센 것만 쓴다 (review.py)."""

    problems: list[str] = Field(description="오늘 문제였던 것. 한 문장씩, 많아야 세 개. 주어진 숫자를 근거로")
    needs: list[str] = Field(description="더 필요했던 것. 한 문장씩, 많아야 세 개")
    staffing: Literal["모자람", "적당함", "넉넉함"] = Field(description="요원 수가 어땠는지")
    staffing_reason: str = Field(description="그렇게 본 까닭. 한두 문장. 주어진 숫자를 근거로")
    suggestions: list[str] = Field(description="운영 지침에 더할 문장. 많아야 세 개. 어느 날에나 쓸 수 있는 규칙으로 쓰고, 오늘의 특정 시각이나 요원 번호는 넣지 않는다")


def read_rules(zones: list[str]) -> str:
    """관리자의 글을 읽는 규칙. 구역은 지금 행사장에 있는 것만 알려 준다."""
    return f"""읽는 규칙:
- 구역은 {', '.join(zones)}뿐이다. "에이", "비", "씨", "디"처럼 한글로 쓴 것도 같은 구역이다. 여기에 없는 구역이면 unclear로 한다.
- "~구역은 n명 유지", "최소 n명", "n명은 남겨", "n명은 있어야" 같은 말은 그 구역의 최소 인원 지시다.
- 최소 인원과 상관없는 말은 other, 무슨 뜻인지 알 수 없는 말은 unclear로 한다. 추측해서 채우지 않는다."""

REPLY_RULES = """읽는 규칙:
- "네", "넵", "ㅇㅋ", "알겠습니다", "이동합니다", "갑니다"처럼 받아들이는 말은 yes.
- 지금 하는 일 때문에 바로 못 간다는 말은 no. "5분 뒤에 가능"처럼 나중에야 된다는 말도 지금은 못 가는 것이므로 no.
- 되묻는 말이나 뜻을 알 수 없는 말은 unclear.
- 다친 사람, 환자, 쓰러진 사람, 119 이야기가 있으면 injury_mentioned는 true. 미아, 분실물처럼 다친 사람이 아닌 일은 false."""


def ensure_model(cfg: dict) -> None:
    """설정에 적힌 모델이 Ollama에 없으면 만든다. 바탕 모델(base_model)과 같은 모델이고, 한 번에 읽는 글의 길이 한도(num_ctx)만 다르다.

    새로 내려받는 것은 없다. 한도를 줄이면 모델이 GPU 한 장에 올라가, 다른 한 장을 사람 찾기에 쓸 수 있다 (E-59).
    CrewAI가 Ollama를 부르는 길로는 이 값을 요청마다 넘길 수 없어서(넘기면 오류), 값을 담은 이름을 따로 만들어 둔다.
    """
    if not cfg["model"].startswith("ollama") or "base_model" not in cfg:
        return
    name = cfg["model"].split("/", 1)[1]
    try:
        with urllib.request.urlopen(cfg["base_url"] + "/api/tags", timeout=5) as r:
            if any(m["name"] == name for m in json.loads(r.read().decode()).get("models", [])):
                return
        body = json.dumps({"model": name, "from": cfg["base_model"], "parameters": {"num_ctx": cfg["num_ctx"]}, "stream": False}).encode()
        req = urllib.request.Request(cfg["base_url"] + "/api/create", data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60).read()
    except Exception:  # Ollama가 꺼져 있음 등. 그때는 흐름이 규칙으로 대신한다
        pass


def _parsed(result):
    """정해진 형식으로 읽힌 답을 꺼낸다.

    CrewAI는 답을 형식에 맞추지 못해도 오류 없이 빈 값(None)을 줄 때가 있다 (설치한 crewai의 agent/core.py에서 확인, E-72).
    그때는 여기서 오류를 내서, 흐름(flow.py의 _ask)이 규칙으로 대신하고 기록하게 한다 (C-44).
    """
    if result.pydantic is None:
        raise ValueError("AI의 답을 정해진 형식으로 읽지 못함")
    return result.pydantic


class CrewBrain:
    """CrewAI 담당들. 이 컴퓨터의 Ollama 모델을 쓴다 (D-48)."""

    name = "CrewAI 담당"

    def __init__(self, llm_cfg: dict):
        from crewai import Agent, LLM  # 무거운 불러오기는 쓸 때만

        ensure_model(llm_cfg)
        self.llm = LLM(model=llm_cfg["model"], base_url=llm_cfg["base_url"], temperature=llm_cfg["temperature"],
                       timeout=llm_cfg["timeout_seconds"], reasoning_effort=llm_cfg["reasoning_effort"])
        common = dict(llm=self.llm, verbose=False, allow_delegation=False)
        # 토론의 효율·안전 의견만 가벼운 모델로 내고, 둘을 동시에 부를 수 있다 (D-65, 재 본 결과는 E-73).
        # 꺼 두면(llm.opinion.enabled: false) 모든 담당이 위의 모델 하나를 쓰고 차례대로 부른다.
        op = llm_cfg.get("opinion") or {}
        self.main_model = llm_cfg.get("base_model") or llm_cfg["model"].split("/", 1)[-1]  # 기록과 화면에 보여 줄 모델 이름
        self.opinion_llm, self.opinions_together, self.opinion_model = self.llm, False, None
        if op.get("enabled"):
            ensure_model(dict(llm_cfg, model=op["model"], base_model=op["base_model"], num_ctx=op["num_ctx"]))
            self.opinion_llm = LLM(model=op["model"], base_url=llm_cfg["base_url"], temperature=llm_cfg["temperature"],
                                   timeout=llm_cfg["timeout_seconds"], reasoning_effort=llm_cfg["reasoning_effort"])
            self.opinions_together, self.opinion_model = bool(op.get("together")), op["base_model"]
        self._light = dict(common, llm=self.opinion_llm)
        self.communicator = Agent(
            role="소통 담당", goal="관리자와 안전요원이 쓴 글을 읽어, 배치 계산에 쓸 수 있는 정해진 형식으로 정확히 바꾼다",
            backstory="행사 안전 상황실에서 현장의 글을 읽는 일을 맡았다. 적힌 것만 읽고, 모르는 것은 모른다고 한다.", **common)
        self._make_opinion_agents()
        self.planner = Agent(
            role="계획 담당", goal="효율 담당과 안전 담당의 의견을 듣고, 주어진 선택지 가운데 하나를 골라 처리 순서를 정한다",
            backstory="상황실의 조정자다. 선택지에 없는 안을 새로 만들지 않고, 어느 의견을 왜 따랐는지 분명히 적는다.", **common)
        self.evaluator = Agent(
            role="확인 담당", goal="요원이 도착한 뒤 다시 센 결과를 보고, 사건을 닫을지 더 대응할지 정한다",
            backstory="숫자만 보고 판단한다. 영상에서 사람이 준 것이 요원 덕분이라고 말하지 않는다.", **common)
        # 지침 담당 (D-64): 운영 지침에서 찾아온 조항을 읽어 준다. 조항을 찾아오는 일은 도구(guideline.py)가 하고, 이 담당은 어느 조항이 해당하는지를 가린다
        self.guide = Agent(
            role="지침 담당", goal="운영 지침에서 찾아온 조항 가운데 지금 상황에 해당하는 것을 가려 읽어 주고, 그 조항대로라면 어느 선택지가 맞는지 말한다",
            backstory="행사 운영 지침을 맡았다. 지침에 적힌 것만 말한다. 해당하는 조항이 없으면 없다고 한다.", **common)

    def _make_opinion_agents(self) -> None:
        """효율 담당과 안전 담당을 만든다. 둘은 같은 모델(self.opinion_llm)을 쓰고 지시문만 다르다."""
        from crewai import Agent

        self.efficiency = Agent(
            role="효율 담당", goal="붐비는 구역에 가장 빨리, 가장 적은 사람을 움직여 대응하게 한다",
            backstory="이동 시간과 움직이는 인원 수를 따진다. 필요 이상으로 사람을 움직이는 것을 낭비로 본다.", **self._light)
        # 안전 담당의 눈을 효율 담당과 다르게 둔다 (D-63). 처음에는 둘 다 "적게 움직이는 쪽"을 좋게 봐서 거의 늘 같은 선택지를 골랐다.
        # 효율 담당은 꼭 필요한 만큼만 빨리, 안전 담당은 붐비는 구역이 확실히 풀리도록 여유 있게. 어느 쪽을 따를지는 계획 담당이 정한다.
        self.safety = Agent(
            role="안전 담당", goal="붐비는 구역이 확실히 풀리도록 여유 있게 대응하게 하고, 그러면서도 다른 구역이 비지 않게 한다",
            backstory="붐비는 구역에 요원이 모자라 늦게 풀리는 것을 가장 큰 위험으로 본다. 단계가 높을수록 필요한 수보다 넉넉히 보내는 쪽을 먼저 살핀다. "
                      "다른 구역이 최소 인원만 남는 것, 휴게가 가까운 요원을 보내는 것도 위험으로 본다. 예비 요원은 이런 때 쓰려고 둔 인력으로 본다.", **self._light)

    def read_manager_text(self, text: str, min_staff: dict[str, int]) -> ManagerInstruction:
        now = ", ".join(f"{z}구역 {n}명" for z, n in min_staff.items())
        prompt = f"{read_rules(list(min_staff))}\n\n지금의 구역별 최소 인원: {now}\n관리자가 쓴 글: \"{text}\"\n\n이 글을 읽고 정해진 형식으로 답하라."
        return _parsed(self.communicator.kickoff(prompt, response_format=ManagerInstruction))

    def read_staff_reply(self, text: str, order: str) -> StaffReply:
        prompt = f"{REPLY_RULES}\n\n요원에게 보낸 지시: \"{order}\"\n요원이 보낸 답: \"{text}\"\n\n이 답을 읽고 정해진 형식으로 답하라."
        return _parsed(self.communicator.kickoff(prompt, response_format=StaffReply))

    def debate(self, situation: str, options_text: str, guideline: list[dict] | None = None) -> tuple[PlanDecision, list[dict]]:
        """효율 담당과 안전 담당이 서로의 글을 보지 않고 따로 의견을 내고, 계획 담당이 둘을 함께 읽고 정한다. (결정, 토론 기록)을 돌려준다.

        guideline: 운영 지침에서 찾아온 조항들 [{id, text, ...}]. 주어지면 지침 담당이 읽어 주고, 계획 담당이 그것까지 보고 정한다 (D-64).

        처음에는 안전 담당이 효율 담당의 의견을 먼저 읽고 동의하거나 반박하게 했다. 그런데 실제 실행에서 안전 담당이 그 글을
        글자 그대로 되풀이하는 일이 있었다 (E-63). 그래서 따로 의견을 내게 하고, 담당마다 근거로 들 숫자를 정해 주었다 (D-59).
        """
        base =f"상황: {situation}\n\n계산 도구가 만든 선택지 (모두 휴게와 최소 인원 조건을 지킨다. 이 밖의 안은 만들 수 없다):\n{options_text}\n\n"
        # 담당을 하나씩 바로 불러 보았으나(Crew와 Task를 쓰지 않고) 걸리는 시간이 같아서(16.6초와 18.6초), Crew와 Task로 묶는 방식을 그대로 둔다 (E-63)
        from crewai import Crew, Process, Task

        # context=[]: 앞의 일(Task)의 답을 보여 주지 않는다. 적지 않으면 CrewAI가 앞의 답을 모두 보여 준다 (설치한 crewai의 crew.py에서 확인)
        # async_execution: 켜면 두 의견의 일이 동시에 돌고, 뒤의 계획 담당의 일은 둘이 끝나기를 기다린다 (CrewAI 공식 문서의 Task 설명, 설치한 crew.py에서 확인. E-72)
        together = self.opinions_together
        t_eff = Task(description=base + "효율의 눈으로 선택지 하나를 고르고, 이유를 한두 문장으로 말하라. 근거는 선택지에 적힌 이동 시간과 움직이는 인원 수의 숫자로 들어라. 핵심 한 줄도 따로 적어라.",
                     expected_output="고른 선택지의 번호, 핵심 한 줄, 이유", agent=self.efficiency, context=[], output_pydantic=Opinion, async_execution=together)
        t_safe = Task(description=base + "안전의 눈으로 선택지 하나를 고르고, 이유를 한두 문장으로 말하라. 근거는 붐빔 단계와 보내는 인원 수(여유가 있는지), "
                                          "선택지에 적힌 보낸 뒤 남는 근무 인원, 다음 휴게까지 남은 시간의 숫자로 들어라. 핵심 한 줄도 따로 적어라.",
                      expected_output="고른 선택지의 번호, 핵심 한 줄, 이유", agent=self.safety, context=[], output_pydantic=Opinion, async_execution=together)
        guide_line, with_guide = None, ""
        if guideline:  # 지침 담당이 먼저, 찾아온 조항 가운데 해당하는 것을 가린다. 그 말을 계획 담당이 두 의견과 함께 읽는다
            listed = "\n".join(f"[{c['id']}] {c['text']}" for c in guideline)
            note = self.guide.kickoff(
                base + f"운영 지침에서 찾아온 조항:\n{listed}\n\n이 가운데 지금 선택지를 고르는 데 해당하는 조항을 가려내라.\n"
                "- 조항이 '누구를 먼저 보낼지'나 '몇 명을 보낼지'를 직접 말하고 있을 때만 해당한다고 한다.\n"
                "- 해당하는 조항이 있으면 그 조항대로 할 때 맞는 선택지의 번호를 말한다.\n"
                "- 해당하는 조항이 없으면 없다고 하고(applies는 false) 선택지 번호는 0으로 한다. 조항에 적히지 않은 말을 지어내지 않는다.\n"
                # 해당하는 조항이 없을 때 조항마다 긴 설명을 쓰느라 토론이 10초쯤 늦어졌다 (E-71). 짧게 쓰게 한다
                "- 답은 짧게 쓴다. 해당하는 조항이 없으면 argument에는 '해당하는 조항 없음'이라고만 쓰고, 조항마다 따로 설명하지 않는다.",
                response_format=GuideNote).pydantic
            if note is None:  # 지침 담당의 답을 형식대로 읽지 못함: 토론을 멈추지 않고 "해당 없음"으로 보고 이어 간다
                note = GuideNote(applies=False, clause_ids=[], preferred_option=0, point="", argument="지침 담당의 답을 읽지 못해 해당하는 조항이 없는 것으로 봄")
            known = {c["id"] for c in guideline}
            ids = [i.strip("[] ") for i in note.clause_ids if i.strip("[] ") in known]
            applies = bool(note.applies and ids and note.preferred_option > 0)  # 조항 번호나 선택지가 없으면 "해당 없음"으로 본다
            guide_line = {"who": "지침 담당", "option": note.preferred_option if applies else 0, "says": note.argument, "point": note.point if applies else "해당하는 조항 없음",
                          "clauses": ids if applies else []}
            if applies:
                cited = "\n".join(f"[{c['id']}] {c['text']}" for c in guideline if c["id"] in ids)
                with_guide = (f"지침 담당이 운영 지침에서 해당하는 조항을 찾아 읽어 주었다:\n{cited}\n지침 담당의 말: 이 조항대로라면 선택지 {note.preferred_option}. {note.argument}\n"
                              "운영 지침은 관리자가 승인한 것이므로 이 조항에 맞는 선택지를 고른다. 따르지 않을 때는 그 까닭을 분명히 적는다. ")
            else:
                with_guide = "지침 담당이 운영 지침을 찾아보았으나, 이 선택에 해당하는 조항은 없었다. "
        t_plan = Task(description=base + "함께 주어진 글은 두 담당이 서로의 글을 보지 않고 따로 낸 의견이다. 첫 번째가 효율 담당, 두 번째가 안전 담당의 의견이다. " + with_guide +
                                          "의견들이 같은 선택지를 골랐는지, 근거가 어디서 갈렸는지 보고 최종 선택지 하나를 정하라. "
                                          "결정의 핵심 한 줄을 적고, 어느 의견을 왜 따랐는지 한두 문장으로 적고, 이 뒤에 할 일을 차례대로 세 개 적어라.",
                      expected_output="최종 선택지의 번호, 핵심 한 줄, 이유, 할 일의 차례", agent=self.planner, context=[t_eff, t_safe], output_pydantic=PlanDecision)
        try:
            Crew(agents=[self.efficiency, self.safety, self.planner], tasks=[t_eff, t_safe, t_plan], process=Process.sequential, verbose=False).kickoff()
        except Exception:
            if together:  # 동시에 돌던 다른 의견의 일이 아직 끝나지 않았을 수 있다. 그 담당을 곧바로 다시 쓰면 CrewAI가 오류를 내므로(E-72) 두 담당을 새로 만든다
                self._make_opinion_agents()
            raise
        eff, safe, plan = _parsed(t_eff.output), _parsed(t_safe.output), _parsed(t_plan.output)
        light = self.opinion_model or self.main_model  # 어느 모델이 낸 말인지 기록에 남긴다 (E-07: 쓴 도구가 기록에 나와야 함)
        transcript = [
            {"who": "효율 담당", "option": eff.preferred_option, "says": eff.argument, "point": eff.point, "model": light},
            {"who": "안전 담당", "option": safe.preferred_option, "says": safe.argument, "point": safe.point, "model": light},
        ]
        if guide_line:
            transcript.append(dict(guide_line, model=self.main_model))
        transcript.append({"who": "계획 담당", "option": plan.chosen_option, "says": plan.reason, "point": plan.headline, "model": self.main_model})
        return plan, transcript

    def judge(self, facts: str) -> Verdict:
        prompt = (f"{facts}\n\n판단 규칙:\n- 붐빔 단계가 요원을 보내는 기준 아래로 내려갔으면 close.\n- 단계가 그대로이거나 올랐고 더 보낼 인원이 계산상 필요하면 send_more.\n"
                  "- 단계가 그대로이지만 필요한 인원은 이미 보냈으면 keep_watching.\n- 정해진 횟수만큼 다시 확인했는데도 풀리지 않았으면 notify_manager.\n\n정해진 형식으로 답하라.")
        return _parsed(self.evaluator.kickoff(prompt, response_format=Verdict))

    def review(self, facts: str, stats: dict) -> Review:
        """하루치 기록을 정리한다 (D-64). 숫자는 facts에 적힌 것(코드가 기록에서 센 것)만 쓰게 한다."""
        prompt = (f"{facts}\n\n위 숫자만 근거로 오늘의 운영을 정리하라.\n규칙:\n- 위에 없는 숫자나 일을 지어내지 않는다.\n"
                  "- 사람 수가 준 것을 요원 덕분이라고 쓰지 않는다 (녹화 영상이다).\n"
                  "- 요원 수는 이렇게 본다: 보낼 요원이 없던 일이나 교대 요원·최소 인원을 채우지 못한 일이 한 번이라도 있으면 '모자람', "
                  "그런 일이 없고 예비 요원을 다 쓰지 않았으면 '넉넉함', 그 밖에는 '적당함'.\n"
                  "- 효율 담당과 안전 담당의 선택이 갈린 것은 문제로 적지 않는다 (두 담당은 일부러 서로 다른 눈으로 보게 한 것이다).\n"
                  "- 지침에 더할 문장은 다음 날 토론에서 선택지를 고를 때 바로 쓸 수 있는 규칙으로 쓴다: 근무 요원과 예비 요원 가운데 누구를 먼저 보낼지, "
                  "필요한 수만 보낼지 한 명 더 보낼지, 예비 요원을 어떻게 아끼거나 쓸지. 위의 숫자에서 드러난 일을 근거로 삼는다. "
                  "'절차를 명확히 한다', '기준이 필요하다'처럼 무엇을 고르라는 것인지 알 수 없는 문장은 쓰지 않는다. "
                  "오늘의 특정 시각, 구역 이름, 요원 번호는 넣지 않는다. 프로그램을 고치라는 말은 쓰지 않는다.\n"
                  "- 쉬운 우리말로 쓴다. 영어 낱말이나 줄임말을 쓰지 않는다.\n\n정해진 형식으로 답하라.")
        review = _parsed(self.evaluator.kickoff(prompt, response_format=Review))
        review.problems, review.needs, review.suggestions = review.problems[:3], review.needs[:3], review.suggestions[:3]  # "많아야 세 개"를 넘기면 자른다
        return review


class RuleBrain:
    """AI 없이 규칙으로만 처리하는 대신할 것. 시험할 때와, AI가 답하지 못할 때(오류, 형식에 맞지 않는 답) 쓴다."""

    name = "규칙 (AI 없음)"
    ZONES = {"a": "A", "에이": "A", "b": "B", "비": "B", "c": "C", "씨": "C", "d": "D", "디": "D"}
    NUMBERS = {"한": 1, "두": 2, "세": 3, "네": 4, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}

    def read_manager_text(self, text: str, min_staff: dict[str, int]) -> ManagerInstruction:
        z = re.search(r"(a|b|c|d|에이|비|씨|디)\s*구역", text.lower())
        n = re.search(r"(\d|한|두|세|네)\s*명", text)
        if z and n and re.search(r"유지|최소|남겨|있어야", text):
            zone, num = self.ZONES[z.group(1)], self.NUMBERS[n.group(1)]
            return ManagerInstruction(kind="set_min_staff", zone=zone, min_staff=num, restated=f"{zone}구역의 최소 인원을 {num}명으로 바꿉니다.")
        return ManagerInstruction(kind="unclear", zone="none", min_staff=0, restated="무슨 뜻인지 읽지 못했습니다.")

    def read_staff_reply(self, text: str, order: str) -> StaffReply:
        injury = bool(re.search(r"다친|환자|쓰러|119|부상", text))
        if re.search(r"못\s*가|못\s*갑|불가|안\s*돼|어렵", text) or injury:
            return StaffReply(can_go="no", reason=text.strip(), injury_mentioned=injury)
        if re.search(r"^(네|넵|예|ㅇㅋ|오케이|알겠)|이동합니다|갑니다|가겠습니다", text.strip()):
            return StaffReply(can_go="yes", reason="", injury_mentioned=False)
        return StaffReply(can_go="unclear", reason="", injury_mentioned=False)

    def debate(self, situation: str, options_text: str, guideline: list[dict] | None = None) -> tuple[PlanDecision, list[dict]]:
        plan = PlanDecision(chosen_option=1, headline="기본 계산대로 가장 가까운 요원을 보냄", reason="AI 없이 기본 계산(선택지 1)을 그대로 씀.",
                            steps=["지시 보내기", "도착 확인", "다시 세기"])
        return plan, [{"who": "규칙", "option": 1, "says": plan.reason, "point": plan.headline}]

    def judge(self, facts: str) -> Verdict:
        if "기준 아래" in facts:
            return Verdict(decision="close", reason="붐빔 단계가 요원을 보내는 기준 아래로 내려감.")
        if "더 필요한 인원: 0명" in facts:
            return Verdict(decision="keep_watching", reason="필요한 인원은 이미 보냈고 단계가 아직 내려가지 않음.")
        return Verdict(decision="send_more", reason="단계가 내려가지 않았고 계산상 인원이 더 필요함.")

    def review(self, facts: str, stats: dict) -> Review:
        """AI 없이 쓰는 정리: 숫자를 정해 둔 문장에 넣기만 한다."""
        short = stats["no_staff"] + stats["no_relief"] + stats["fill_short"]
        problems = [f"조건을 지키면서 보낼 요원이 없어 관리자에게 넘긴 일이 {stats['no_staff']}번 있었음."] if stats["no_staff"] else []
        problems += [f"휴게 교대 요원을 구하지 못한 일이 {stats['no_relief']}번 있었음."] if stats["no_relief"] else []
        problems += [f"닫지 못한 사건이 {stats['incidents_opened'] - stats['incidents_closed']}건 남았음."] if stats["incidents_opened"] > stats["incidents_closed"] else []
        needs = [f"다친 사람 보고가 {stats['injuries']}건 있었음. 그런 날에 쓸 여유 인원이 필요함."] if stats["injuries"] else []
        staffing = "모자람" if short else "넉넉함" if stats["reserves_total"] and stats["reserves_used"] < stats["reserves_total"] else "적당함"
        return Review(problems=problems or ["기록에서 문제로 볼 일을 찾지 못함."], needs=needs or ["기록에서 더 필요했던 것을 찾지 못함."], staffing=staffing,
                      staffing_reason=f"보낼 요원이 없거나 교대·최소 인원을 채우지 못한 일이 {short}번, 예비 요원 {stats['reserves_total']}명 가운데 {stats['reserves_used']}명을 씀.",
                      suggestions=["보낼 요원이 없던 날에는 다음 날 예비 요원 한 명을 남겨 두고 시작한다."] if short else [])
