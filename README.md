# Concept → 3D

컨셉 이미지를 올리면 **ADK 에이전트**가 분석하고, **MCP 서버**의 도구로 3D 모델(`.glb`)을 만들어 주는 파이썬 프로젝트입니다.

```
사용자 이미지
   │
   ▼
┌──────────── ADK: SequentialAgent "concept3d" ────────────┐
│ before_agent_callback  첨부 이미지를 workspace/inputs에 저장 │
│ 1. concept_analyst     피사체/형태/색/유지할 특징 분석        │
│                        + 정제용 영어 프롬프트 (output_key)    │
│ 2. modeler             MCP 도구를 순서대로 호출하고 결과 보고  │
└──────────────────────────────┬───────────────────────────┘
                               │ McpToolset (stdio)
┌──────────── MCP 서버: mcp_server/server.py ───────────────┐
│ prepare_image         회전 보정, 흰 배경 정사각형 패딩        │
│ refine_concept_image  Gemini 이미지 모델로 3D 변환용 재드로잉  │
│ image_to_3d           trellis(무료 HF Space) | tripo(유료 API) │
│ list_outputs          생성된 .glb 목록                        │
└──────────────────────────────────────────────────────────┘
```

## 설치

```bash
uv sync
```

`.env.example`을 `.env`로 복사하고 `GOOGLE_API_KEY`를 넣습니다. 필수 값은 이것 하나입니다.

## 실행

**Streamlit UI** (이미지 업로드 → 3D 뷰어 → .glb 다운로드)

```bash
uv run streamlit run app.py
```

**ADK 개발 UI** (에이전트 이벤트, 도구 호출, state를 단계별로 확인)

```bash
uv run adk web agents
```

## 파일 구조

| 경로 | 역할 |
|---|---|
| `mcp_server/server.py` | MCP 서버. 이미지 처리와 3D 변환 도구 |
| `agents/concept3d/agent.py` | ADK 에이전트 (`root_agent`) |
| `app.py` | Streamlit 화면 |
| `workspace/` | 입력, 정제 이미지, 생성된 모델 저장 (git 제외) |

## 설계 메모

- **이미지는 파일 경로로만 주고받습니다.** base64를 LLM 컨텍스트에 넣지 않으려고 콜백에서 첨부 이미지를 파일로 저장하고, 도구끼리는 경로를 넘깁니다.
- **stdio MCP의 `timeout`은 도구 응답 대기 시간이기도 합니다.** ADK 2.10은 stdio 연결의 `timeout`(기본 5초)을 도구 호출 읽기 타임아웃으로도 씁니다. 3D 생성은 수십 초에서 수 분이 걸려서 `timeout=900`으로 설정했습니다.
- **TRELLIS 결과물 재질을 보정합니다.** TRELLIS가 만든 glb에는 `metallicFactor`가 없어서 glTF 기본값 1.0(완전 금속)으로 해석되고, 뷰어에서 거의 검게 보입니다. `image_to_3d`가 저장 후 0으로 채웁니다.
- **백엔드를 바꿔도 에이전트 코드는 그대로입니다.** `image_to_3d`의 `backend` 인자나 `.env`의 `THREED_BACKEND`만 바꾸면 됩니다.

## 알려진 제한

- TRELLIS Space는 무료라서 대기열과 GPU 할당량 제한이 있습니다. 막히면 `HF_TOKEN`을 넣거나 `tripo` 백엔드를 쓰세요.
- `tripo` 백엔드는 공식 API 문서 기준으로 작성했고, 키가 없어서 실제 호출 테스트는 하지 못했습니다.
- 모델 이름(`gemini-2.5-flash`, `gemini-2.5-flash-image`)은 `.env`에서 바꿀 수 있습니다.

## 확장 아이디어

- **Critic 루프:** `LoopAgent`로 정제 이미지가 원본 컨셉의 특징을 지켰는지 검사하고 재생성하기
- **멀티뷰 입력:** TRELLIS의 `multiimages`로 정면, 측면 등 여러 장을 넣어 형태 정확도 높이기
- **MCP 서버 재사용:** 같은 서버를 Claude Desktop 같은 다른 MCP 클라이언트에 연결하기
