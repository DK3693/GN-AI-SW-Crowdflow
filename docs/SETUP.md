# 실행 환경 및 설치

우리가 돌려 본 환경은 이것 하나입니다: Windows 10 Pro, NVIDIA RTX 3090 24GB 두 장(드라이버 617.14), Python 3.13.9, uv 0.12.22, Ollama 0.32.13.

| 필요한 것 | 내용 |
|---|---|
| Python | 3.11 이상 3.14 미만 (`pyproject.toml`) |
| uv | 패키지 설치와 실행에 씁니다. https://docs.astral.sh/uv/ |
| NVIDIA GPU | 글 읽는 AI가 약 22~23.5GB(GPU 0), 사람 찾기가 약 1GB(GPU 1)를 썼습니다. 오래된 드라이버(560.94)에서는 GPU를 쓰지 못했습니다 |
| Ollama | 이 컴퓨터에서 언어 모델을 돌립니다. https://ollama.com/ |
| 언어 모델 | Ollama 공식 목록의 `qwen3.8:27b` (약 17GB, Apache 2.0) |
| 영상 | AI Hub "지능형 관제 서비스 CCTV 영상 데이터"의 인파밀집 영상 12개. **이 저장소에 들어 있지 않습니다** (아래 5번) |

GPU가 한 장뿐이면 `config/settings.yaml`의 `vision_live.device`를 `0`으로 바꿔야 합니다. 한 장에서 돌려 보지는 않았습니다.

---

## 설치

```bash
uv sync --locked
```

```bash
ollama pull qwen3.8:27b
```

- 둘째 명령은 우리 컴퓨터에서는 돌려 보지 않았습니다(모델이 이미 있었습니다).
- 프로그램이 처음 켜질 때 이 모델에 글 길이 한도만 정한 이름(`crowdflow-qwen:27b`)을 Ollama 안에 스스로 만듭니다. 새로 내려받는 것은 없습니다.
- 키, 토큰, 비밀번호는 쓰지 않습니다. 밖의 AI API를 부르지 않습니다.

---

## 데이터 준비

영상은 AI Hub의 이용 조건 때문에 이 저장소에 넣지 않았습니다. AI Hub에서 직접 받아야 합니다(로그인과 승인이 필요합니다).
https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71850

1. 아래 12개 파일을 `data/aihub71850_crowd/video/`에 넣습니다.

   | 구역 | 영상 (이 차례로 이어 붙임) |
   |---|---|
   | A (계단 골목) | `E05_023.mp4`, `E05_002.mp4`, `E05_025.mp4` |
   | B (벽화 골목) | `E05_022.mp4`, `E05_026.mp4`, `E05_024.mp4` |
   | C (노란 건물 모퉁이) | `E05_004.mp4`, `E05_008.mp4`, `E05_012.mp4` |
   | D (숫자 골목) | `E05_029.mp4`, `E05_019.mp4`, `E05_035.mp4` |

2. 영상마다 한 번씩 미리 분석해 저장합니다. 화면 서버와 시험이 이 파일을 읽습니다. 처음 돌릴 때 YOLO 모델 파일(`yolo26x.pt`)이 Ultralytics에서 내려받아집니다.

   Git Bash:
   ```bash
   PYTHONPATH=src uv run --locked python -m crowdflow.detect_cache --video data/aihub71850_crowd/video/E05_023.mp4 --step 3 --model yolo26x.pt --imgsz 1280 --out outputs/detections/aihub/E05_023.npz
   ```

   PowerShell에서는 맨 앞의 `PYTHONPATH=src`를 빼고, 먼저 `$env:PYTHONPATH = "src"`를 한 번 입력합니다. 파일 이름만 바꿔 12개 모두에 되풀이합니다.

---

## 실행

### 가. 화면으로 보기

1. Ollama를 켭니다(Ollama 앱을 실행하거나 터미널에서 `ollama serve`).
2. 화면 서버를 켭니다. 실시간 분석 일꾼은 이 서버가 스스로 띄웁니다.

   ```bash
   uv run --locked streamlit run src/crowdflow/app.py
   ```

3. 브라우저에서 엽니다.
   - 관리자 화면: http://localhost:8501
   - 요원 화면: http://localhost:8501/staff (이름을 골라 들어갑니다)
4. 관리자 화면 위쪽에 "CrewAI 담당 준비됨"과 "GPU 실시간 분석"이 보이면 "시작"을 누릅니다.

해 볼 것:
- **혼자 볼 때:** "시작" 전에 "요원 대본 켜기"를 누릅니다. 요원의 답과 "도착"을 대본이 대신 넣습니다. 켜 있는 동안 화면에 "요원 역할: 대본"이 보입니다.
- **여럿이 볼 때:** 다른 창에서 요원 화면을 열어, 지시를 받은 요원으로 답하거나 "도착"을 누릅니다. 예: `다친 관람객을 돌보는 중이라 못 갑니다`
- **관리자의 글 지시:** 글 칸에 `A구역은 3명 유지해`처럼 씁니다. 그 구역의 최소 인원이 바뀌고, 모자라면 채우러 보냅니다.
- **취소:** 나간 지시 옆의 "취소"를 누르면 그 요원은 제자리에 남습니다.
- **결정의 까닭:** "AI의 결정" 목록의 줄을 누르면 토론의 핵심, 선택지, 보내지 않은 요원과 까닭이 펼쳐집니다.
- **결과 보고서:** 영상(3분)이 끝난 뒤, "처음부터 다시"를 누르기 전에 위쪽의 "결과 보고서" → "결과 보고서 쓰기"를 누릅니다(주소 끝에 `/report`). 큰 숫자, 사건별 표, 현장에서 들어온 일은 코드가 실행 기록에서 세고, "확인 담당의 정리"만 AI가 그 숫자를 보고 씁니다. 12초쯤 걸립니다. 보고서는 파일로도 남습니다(`outputs/live/reports/report_*.md`).

Ollama가 꺼져 있으면 화면에 "AI 서버에 닿지 않음"이 뜨고 규칙으로 대신 처리합니다.

### 나. 화면 없이 처음부터 끝까지 돌리기

영상 3분을 빨리 흘려보내며 정해 둔 대본(관리자의 글, 요원의 답과 도착)을 넣습니다. 기록은 `outputs/runs/`에 남습니다.

Git Bash:
```bash
PYTHONPATH=src uv run --locked python -m crowdflow.run_e2e
```

AI 없이 규칙으로만 돌리려면 끝에 `--rule`을 붙입니다(약 7초).

PowerShell:
```powershell
$env:PYTHONPATH = "src"; uv run --locked python -m crowdflow.run_e2e --rule
```

### 다. 시험

자동 시험(AI 없이 규칙으로 확인, 약 1분):
```bash
uv run --locked pytest -q tests
```

- 영상과 미리 분석한 결과가 있으면 73개가 통과합니다.
- 이 저장소만 받은 상태(영상 없음)에서는 52개가 통과하고 21개는 건너뜁니다.

대표 시험 8건(실제 AI 담당으로, 약 5분). Ollama가 켜져 있어야 합니다.
```bash
PYTHONPATH=src uv run --locked python tests/representative_cases.py
```

입력, 예상 결과, 실제 결과는 [VALIDATION.md](VALIDATION.md)에 있습니다.

---

