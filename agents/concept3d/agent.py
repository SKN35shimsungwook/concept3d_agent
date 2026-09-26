"""컨셉 이미지 → 3D 모델 재해석 에이전트 (Google ADK).

흐름 (SequentialAgent)
  0. save_uploaded_image (before_agent_callback)
       첨부 이미지를 workspace/inputs에 저장하고, 경로와 사용자 요청을 state에 기록
  1. concept_analyst  : 컨셉 이미지를 분석 (384px 썸네일만 전송)
  2. modeler          : MCP 도구 concept_to_3d를 한 번 호출하고 결과 보고 (이미지 전송 없음)

토큰 절약 원칙: LLM은 판단이 필요한 곳에만 쓰고, 정해진 순서는 MCP 도구 안의 코드로 처리한다.
한 번 실행에 LLM 호출 3회(분석 1, 도구 호출 1, 보고 1), 이미지 전송 1회.

실행: 프로젝트 루트에서 `uv run adk web agents`
"""

from __future__ import annotations

import io
import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from google.genai import types
from mcp import StdioServerParameters
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL = os.getenv("AGENT_MODEL", "gemini-2.5-flash")
# Gemini는 가로·세로 모두 384px 이하인 이미지를 258토큰으로 계산한다 (더 크면 타일당 258토큰)
ANALYSIS_IMAGE_SIZE = int(os.getenv("ANALYSIS_IMAGE_SIZE", "384"))
INPUTS_DIR = ROOT / "workspace" / "inputs"
INPUTS_DIR.mkdir(parents=True, exist_ok=True)

_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


def _is_image(part: types.Part) -> bool:
    blob = part.inline_data
    return bool(blob and blob.data and (blob.mime_type or "").startswith("image/"))


def save_uploaded_image(callback_context: CallbackContext):
    """첨부 이미지를 파일로 저장하고 경로와 요청 문장을 state에 남긴다.

    MCP 도구는 파일 경로만 받고, modeler는 대화 기록을 보지 않으므로(include_contents='none')
    필요한 정보는 전부 state로 넘긴다.
    """
    content = callback_context.user_content
    parts = (content.parts if content else None) or []
    for part in parts:
        if _is_image(part):
            blob = part.inline_data
            path = INPUTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{_EXT.get(blob.mime_type, '.png')}"
            path.write_bytes(blob.data)
            callback_context.state["input_image_path"] = str(path)
            break
    text = " ".join(p.text for p in parts if p.text).strip()
    callback_context.state["user_request"] = text or "없음"
    return None  # None을 반환해야 에이전트가 계속 실행된다


def shrink_images_for_analysis(callback_context: CallbackContext, llm_request: LlmRequest):
    """분석 모델에는 가장 최근 이미지 한 장만, 작은 썸네일로 보낸다.

    원본 해상도는 3D 변환에서만 쓰고(파일 경로로 전달), LLM에는 형태와 색을 파악할 정도면 충분하다.
    """
    latest_seen = False
    for content in reversed(llm_request.contents):
        kept = []
        for part in reversed(content.parts or []):
            if not _is_image(part):
                kept.append(part)
                continue
            if latest_seen:
                continue  # 예전 턴의 이미지는 다시 보내지 않는다
            latest_seen = True
            img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
            img.thumbnail((ANALYSIS_IMAGE_SIZE, ANALYSIS_IMAGE_SIZE), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            kept.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
        content.parts = list(reversed(kept))
    return None


def strip_images(callback_context: CallbackContext, llm_request: LlmRequest):
    """modeler에는 이미지를 보내지 않는다.

    include_contents='none'이어도 ADK는 '현재 턴'의 사용자 입력을 포함할 수 있어서, 확실히 빼 둔다.
    """
    for content in llm_request.contents:
        parts = [p for p in content.parts or [] if not _is_image(p)]
        # 이미지만 있던 메시지가 빈 내용이 되면 API가 거부하므로 자리 표시 문장을 남긴다
        content.parts = parts or [types.Part.from_text(text="(이미지는 입력 이미지 경로로 전달됨)")]
    return None


concept_analyst = LlmAgent(
    name="concept_analyst",
    model=MODEL,
    description="컨셉 이미지를 분석해 3D로 재해석할 때 유지할 특징을 정리한다.",
    instruction="""
너는 3D 컨셉 아티스트다. 사용자가 올린 컨셉 이미지(가장 최근 이미지)를 분석한다.
이미지가 없으면 "NO_IMAGE" 한 줄만 출력한다.

아래 형식으로 짧게 답한다.

[피사체] 무엇인지 한 줄
[형태] 실루엣, 비율, 핵심 파츠
[색/재질] 주요 색상과 재질
[반드시 유지] 3D로 바꿔도 사라지면 안 되는 특징 3~5개
[REFINE_PROMPT] 스타일 변환용 영어 프롬프트 한 문단. 반드시 유지할 특징과 사용자가 요청한 스타일을 담고,
"single object, full view, 3/4 front angle, plain pure white background, soft even studio lighting"을 포함한다.
""",
    output_key="concept_analysis",
    before_model_callback=shrink_images_for_analysis,
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
    # 도구 설명(스키마)도 매 호출 토큰으로 나가므로 에이전트에는 한 번에 실행하는 도구만 노출한다
    tool_filter=["concept_to_3d"],
)

modeler = LlmAgent(
    name="modeler",
    model=MODEL,
    description="MCP 도구로 컨셉 이미지를 3D 모델(.glb)로 만든다.",
    instruction="""
너는 3D 모델링 담당자다. 한국어로 짧게 답한다.

입력 이미지 경로: {input_image_path?}
사용자 요청: {user_request?}
컨셉 분석 결과:
{concept_analysis?}

규칙
- 입력 이미지 경로가 비어 있거나 분석 결과가 NO_IMAGE면 도구를 쓰지 말고
  "3D로 바꿀 컨셉 이미지를 올려주세요."라고만 답한다.
- 그 외에는 concept_to_3d를 정확히 한 번 호출한다.
  - image_path: 입력 이미지 경로
  - refine_backend: 사용자 요청에 정제 방식(off, rembg, gemini)이 적혀 있을 때만 그대로 넣고, 없으면 비운다.
  - refine_prompt: refine_backend가 gemini일 때만 분석 결과의 REFINE_PROMPT를 넣는다.
  - threed_backend, seed: 사용자 요청에 있을 때만 넣는다.
- 도구가 error를 돌려주면 원인을 쉽게 설명하고 해결 방법을 한 줄 제안한다. 다시 호출하지 않는다.

보고 형식
- 생성된 3D 모델: glb_path (cached가 true면 "이전 결과 재사용"이라고 표시)
- 처리 단계: steps
- 원본 컨셉에서 유지한 특징 2~3줄
""",
    tools=[threed_tools],
    # 대화 기록(이미지 포함)을 다시 보내지 않는다. 필요한 정보는 위 state 값으로 받는다.
    include_contents="none",
    before_model_callback=strip_images,
)

root_agent = SequentialAgent(
    name="concept3d",
    description="컨셉 이미지를 분석하고 3D 모델(.glb)로 재해석하는 파이프라인",
    sub_agents=[concept_analyst, modeler],
    before_agent_callback=save_uploaded_image,
)
