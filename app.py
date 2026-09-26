"""Concept → 3D 스튜디오 (Streamlit UI).

ADK 에이전트(agents/concept3d)를 Runner로 실행하고, MCP 도구가 만든
정제 이미지와 3D 모델(.glb)을 화면에 보여준다.

실행: uv run streamlit run app.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from google.genai import types

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "agents"))

from google.adk.runners import InMemoryRunner  # noqa: E402

from concept3d.agent import root_agent, threed_tools  # noqa: E402

APP_NAME = "concept3d"

st.set_page_config(page_title="Concept → 3D", page_icon="🧊", layout="wide")


# ---------------------------------------------------------------------------
# 에이전트 실행
# ---------------------------------------------------------------------------
def _tool_payload(response: dict) -> dict | list | None:
    """ADK가 감싼 MCP CallToolResult에서 도구가 실제로 돌려준 값을 꺼낸다."""
    structured = response.get("structuredContent") or response.get("structured_content")
    if isinstance(structured, dict):
        return structured.get("result", structured) if len(structured) == 1 else structured
    for item in response.get("content") or []:
        if item.get("type") == "text":
            try:
                return json.loads(item["text"])
            except (json.JSONDecodeError, KeyError):
                return {"text": item.get("text")}
    return None


async def run_pipeline(image: bytes, mime: str, request: str, on_step) -> dict:
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id="user")
    message = types.Content(
        role="user",
        parts=[types.Part.from_bytes(data=image, mime_type=mime), types.Part.from_text(text=request)],
    )

    result: dict = {"tools": {}, "final": "", "usage": {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}}
    try:
        async for event in runner.run_async(user_id="user", session_id=session.id, new_message=message):
            # LLM 응답 이벤트마다 사용량이 붙는다 → 호출 수와 토큰을 모아 토큰 절약 효과를 확인
            usage = event.usage_metadata
            if usage and not event.partial:
                result["usage"]["llm_calls"] += 1
                result["usage"]["input_tokens"] += usage.prompt_token_count or 0
                result["usage"]["output_tokens"] += usage.candidates_token_count or 0
            for call in event.get_function_calls():
                on_step(f"🔧 `{call.name}` 실행 중…")
            for resp in event.get_function_responses():
                payload = _tool_payload(resp.response or {})
                result["tools"][resp.name] = payload
                on_step(f"✅ `{resp.name}` 완료")
            if event.author == "concept_analyst" and event.is_final_response():
                on_step("🔍 컨셉 분석 완료")
            if event.author == "modeler" and event.is_final_response() and event.content:
                result["final"] = "".join(p.text or "" for p in event.content.parts or [])

        done = await runner.session_service.get_session(app_name=APP_NAME, user_id="user", session_id=session.id)
        result["analysis"] = done.state.get("concept_analysis", "")
    finally:
        # asyncio.run마다 이벤트 루프가 바뀌므로 MCP 서버 프로세스를 정리해 둔다.
        await threed_tools.close()
        await runner.close()
    return result


def model_viewer(glb_path: str, height: int = 520) -> None:
    data = base64.b64encode(Path(glb_path).read_bytes()).decode()
    components.html(
        f"""
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
        <model-viewer src="data:model/gltf-binary;base64,{data}" camera-controls auto-rotate
            shadow-intensity="1" exposure="1.1" environment-image="neutral"
            style="width:100%;height:{height - 20}px;background:#f4f4f6;border-radius:12px;"></model-viewer>
        """,
        height=height,
    )


AGENT_MODEL = os.getenv("AGENT_MODEL", "gemini-2.5-flash")
IMAGE_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image,gemini-3-pro-image").split(",")
    if m.strip()
]


@st.cache_data(ttl=3600, show_spinner=False)
def check_models() -> dict:
    """설정한 모델이 지금 실제로 제공되는지 확인한다. (모델 종료, 이름 변경 대비)"""
    from google import genai

    try:
        available = {m.name.removeprefix("models/") for m in genai.Client().models.list()}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {
        "agent_ok": AGENT_MODEL in available,
        "image_models": [m for m in IMAGE_MODELS if m in available],
    }


# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------
st.title("🧊 Concept → 3D")
st.caption("컨셉 이미지를 올리면 ADK 에이전트가 분석하고, MCP 도구로 3D 모델(.glb)을 만듭니다.")

REFINE_OPTIONS = {
    "rembg": "배경 제거 (무료, 로컬)",
    "off": "정제 안 함",
    "gemini": "Gemini 스타일 변환 (유료)",
}

with st.sidebar:
    st.subheader("설정")
    refine = st.radio(
        "정제 방식",
        list(REFINE_OPTIONS),
        format_func=REFINE_OPTIONS.get,
        help="gemini는 이미지 모델 API를 호출해서 요금이 발생합니다. 추가 요청에 스타일을 적을 때만 쓰세요.",
    )
    backend = st.radio("3D 백엔드", ["trellis", "tripo"], help="trellis: 무료 Hugging Face Space / tripo: 유료 API")
    seed = st.number_input("seed", 0, 2_147_483_647, 0, help="같은 이미지로 다른 결과를 보고 싶으면 바꿔보세요.")
    st.divider()
    has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    st.write("Gemini 키:", "✅" if has_key else "❌ `.env` 확인")
    if has_key:
        model_status = check_models()
        if model_status.get("error"):
            st.warning(f"모델 목록 확인 실패: {model_status['error']}")
        else:
            st.write("에이전트 모델:", "✅" if model_status["agent_ok"] else f"❌ `{AGENT_MODEL}` 없음")
            if refine == "gemini":
                usable = model_status["image_models"]
                st.write("이미지 모델:", f"✅ `{usable[0]}`" if usable else "❌ 사용 가능한 모델 없음")
            if not model_status["agent_ok"] or (refine == "gemini" and not model_status["image_models"]):
                st.caption("`.env`의 모델 이름을 AI Studio의 현재 모델 이름으로 바꿔주세요.")
    if backend == "tripo":
        st.write("Tripo 키:", "✅" if os.getenv("TRIPO_API_KEY") else "❌ `.env` 확인")

left, right = st.columns([1, 1.4], gap="large")

with left:
    upload = st.file_uploader("컨셉 이미지", type=["png", "jpg", "jpeg", "webp"])
    if upload:
        st.image(upload, caption="원본 컨셉", width="stretch")
    wish = st.text_area(
        "추가 요청 (선택)",
        placeholder="예: 픽사 스타일로 귀엽게, 금속 재질로, 받침대 없이",
        height=90,
    )
    go = st.button("3D로 재해석하기", type="primary", disabled=not (upload and has_key), width="stretch")

if go and upload:
    request = (
        "이 컨셉 이미지를 3D 모델로 재해석해줘.\n"
        f"추가 요청: {wish.strip() or '없음'}\n"
        f"정제 방식: {refine}, 3D 백엔드: {backend}, seed: {int(seed)}"
    )
    with right:
        with st.status("에이전트 실행 중… (3D 생성은 30초~수 분)", expanded=True) as status:
            try:
                res = asyncio.run(run_pipeline(upload.getvalue(), upload.type or "image/png", request, st.write))
                status.update(label="완료", state="complete", expanded=False)
                st.session_state.setdefault("history", []).insert(0, {"id": uuid.uuid4().hex, "name": upload.name, **res})
            except Exception as e:
                status.update(label="실패", state="error")
                st.exception(e)

with right:
    history = st.session_state.get("history", [])
    if not history:
        st.info("왼쪽에서 컨셉 이미지를 올리고 버튼을 눌러주세요.")
    for i, item in enumerate(history):
        mesh = item["tools"].get("concept_to_3d") or {}

        with st.container(border=True):
            st.markdown(f"**{item['name']}**" + ("  · ♻️ 이전 결과 재사용" if mesh.get("cached") else ""))
            if isinstance(mesh, dict) and mesh.get("glb_path") and Path(mesh["glb_path"]).exists():
                model_viewer(mesh["glb_path"])
                st.download_button(
                    "⬇️ .glb 다운로드",
                    Path(mesh["glb_path"]).read_bytes(),
                    file_name=Path(mesh["glb_path"]).name,
                    mime="model/gltf-binary",
                    key=f"dl_{item['id']}",
                )
            elif isinstance(mesh, dict) and mesh.get("error"):
                st.error(mesh["error"])

            if item["final"]:
                st.markdown(item["final"])

            with st.expander("과정 보기", expanded=False):
                usage = item.get("usage", {})
                c1, c2, c3 = st.columns(3)
                c1.metric("LLM 호출", usage.get("llm_calls", 0))
                c2.metric("입력 토큰", f"{usage.get('input_tokens', 0):,}")
                c3.metric("출력 토큰", f"{usage.get('output_tokens', 0):,}")
                if isinstance(mesh, dict) and mesh.get("steps"):
                    st.caption(" → ".join(mesh["steps"]))
                if isinstance(mesh, dict) and mesh.get("refined_path") and Path(mesh["refined_path"]).exists():
                    st.image(mesh["refined_path"], caption="정제된 컨셉 (3D 변환 입력)", width=320)
                if isinstance(mesh, dict) and mesh.get("preview_video") and Path(mesh["preview_video"]).exists():
                    st.video(mesh["preview_video"])
                if item.get("analysis"):
                    st.text(item["analysis"])
