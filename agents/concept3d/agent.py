"""컨셉 이미지 → 3D 모델 재해석 에이전트 (Google ADK).

흐름 (SequentialAgent)
  0. save_uploaded_image (before_agent_callback)
       사용자가 올린 이미지를 workspace/inputs에 저장하고 경로를 state에 기록
  1. concept_analyst  : 컨셉 이미지를 분석해 "유지할 특징"과 정제용 영어 프롬프트 작성
  2. modeler          : MCP 도구로 전처리 → 컨셉 정제 → 3D 변환 후 결과 보고

실행: 프로젝트 루트에서 `uv run adk web agents`
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL = os.getenv("AGENT_MODEL", "gemini-2.5-flash")
INPUTS_DIR = ROOT / "workspace" / "inputs"
INPUTS_DIR.mkdir(parents=True, exist_ok=True)

_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


def save_uploaded_image(callback_context: CallbackContext):
    """메시지에 첨부된 이미지를 파일로 저장한다. MCP 도구는 파일 경로만 받기 때문이다."""
    content = callback_context.user_content
    for part in (content.parts if content else None) or []:
        blob = part.inline_data
        if blob and blob.data and (blob.mime_type or "").startswith("image/"):
            path = INPUTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{_EXT.get(blob.mime_type, '.png')}"
            path.write_bytes(blob.data)
            callback_context.state["input_image_path"] = str(path)
            break
    return None  # None을 반환해야 에이전트가 계속 실행된다


concept_analyst = LlmAgent(
    name="concept_analyst",
    model=MODEL,
    description="컨셉 이미지를 분석해 3D로 재해석할 때 유지할 특징을 정리한다.",
    instruction="""
너는 3D 컨셉 아티스트다. 사용자가 올린 컨셉 이미지(가장 최근 이미지)를 분석한다.
이미지가 없으면 "NO_IMAGE" 한 줄만 출력한다.

아래 형식으로만 답한다.

[피사체] 무엇인지 한 줄
[형태] 실루엣, 비율, 핵심 파츠
[색/재질] 주요 색상과 재질 (예: 광택 있는 빨간 세라믹)
[반드시 유지] 3D로 바꿔도 사라지면 안 되는 특징 3~5개
[사용자 요청] 사용자가 텍스트로 요청한 스타일이나 수정 사항 (없으면 "없음")
[REFINE_PROMPT]
Image-to-3D 변환용으로 이미지를 다시 그리기 위한 영어 프롬프트.
반드시 포함: 반드시 유지할 특징, "single object", "full view, 3/4 front angle",
"plain pure white background", "soft even studio lighting", "no text, no shadow on background".
사용자 요청 스타일이 있으면 반영한다.
""",
    output_key="concept_analysis",
)

threed_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "mcp_server" / "server.py")],
        ),
        # stdio 연결에서는 이 값이 도구 호출 응답 대기 시간으로도 쓰인다.
        # 3D 생성은 대기열 포함 수 분이 걸릴 수 있어서 넉넉하게 잡는다.
        timeout=900,
    ),
)

modeler = LlmAgent(
    name="modeler",
    model=MODEL,
    description="MCP 도구로 컨셉 이미지를 3D 모델(.glb)로 만든다.",
    instruction="""
너는 3D 모델링 파이프라인 담당자다. 한국어로 답한다.

입력 이미지 경로: {input_image_path?}
컨셉 분석 결과:
{concept_analysis?}

규칙
- 입력 이미지 경로가 비어 있거나 분석 결과가 NO_IMAGE면 도구를 쓰지 말고
  "3D로 바꿀 컨셉 이미지를 올려주세요."라고만 답한다.
- 사용자가 새 이미지 없이 이전 결과에 대한 질문만 하면 도구 없이 답한다.

순서 (새 이미지가 들어왔거나 사용자가 다시 만들어 달라고 할 때)
1. prepare_image(image_path=입력 이미지 경로)
2. refine_concept_image(image_path=prepared_path, prompt=분석 결과의 REFINE_PROMPT 전문)
   - error가 오면 prepared_path를 그대로 다음 단계에 쓴다.
3. image_to_3d(image_path=refined_path 또는 prepared_path)
   - 사용자가 백엔드(trellis/tripo)나 seed를 지정했으면 반영한다.
   - error가 오면 원인을 쉽게 설명하고 해결 방법을 제안한다. 같은 호출을 3번 넘게 반복하지 않는다.

마지막 보고 형식
- 생성된 3D 모델: glb_path
- 미리보기 영상: preview_video (있을 때)
- 정제 이미지: refined_path (있을 때)
- 사용 백엔드와 걸린 시간
- 원본 컨셉에서 유지한 특징 요약 2~3줄
- 더 좋게 만들 팁 1줄
""",
    tools=[threed_tools],
)

root_agent = SequentialAgent(
    name="concept3d",
    description="컨셉 이미지를 분석하고 3D 모델(.glb)로 재해석하는 파이프라인",
    sub_agents=[concept_analyst, modeler],
    before_agent_callback=save_uploaded_image,
)
