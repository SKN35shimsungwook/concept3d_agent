"""컨셉 이미지 → 3D 모델 변환 도구를 제공하는 MCP 서버 (stdio).

ADK 에이전트가 McpToolset으로 이 서버를 띄워서 도구를 호출한다.
이미지는 base64가 아니라 항상 "파일 경로"로 주고받는다. (LLM 컨텍스트 절약)

도구 목록
- prepare_image        : 3D 변환에 맞게 이미지 정리 (회전 보정, 정사각형 패딩, 리사이즈)
- refine_concept_image : Gemini 이미지 모델로 "흰 배경 + 단일 오브젝트 + 3/4 뷰" 이미지 재생성
- image_to_3d          : 이미지를 3D 모델(.glb)로 변환 (backend: trellis | tripo)
- list_outputs         : 지금까지 만든 3D 모델 목록
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

WORKSPACE = ROOT / "workspace"
PREPARED_DIR = WORKSPACE / "prepared"
REFINED_DIR = WORKSPACE / "refined"
MODELS_DIR = WORKSPACE / "models"
for d in (PREPARED_DIR, REFINED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
TRELLIS_SPACE = os.getenv("TRELLIS_SPACE", "trellis-community/TRELLIS")
TRIPO_API = "https://api.tripo3d.ai/v2/openapi"

mcp = MCPServer(
    "concept3d-tools",
    instructions="컨셉 이미지를 정리하고 3D 모델(.glb)로 변환하는 도구 모음. 모든 이미지는 파일 경로로 전달한다.",
)


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"파일이 없습니다: {p}")
    return p


def _new_name(suffix: str) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{suffix}"


# ---------------------------------------------------------------------------
# 1. 이미지 전처리
# ---------------------------------------------------------------------------
@mcp.tool()
def prepare_image(image_path: str, size: int = 1024) -> dict:
    """이미지를 3D 변환에 맞게 정리한다.

    EXIF 회전을 보정하고, 비율을 유지한 채 흰 배경의 정사각형(size x size)으로 패딩한다.
    반환값의 prepared_path를 다음 도구에 넘기면 된다.
    """
    src = _resolve(image_path)
    img = ImageOps.exif_transpose(Image.open(src))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))

    out = PREPARED_DIR / _new_name(".png")
    canvas.save(out)
    return {"prepared_path": str(out), "size": size}


# ---------------------------------------------------------------------------
# 2. 컨셉 정제 (Gemini 이미지 모델)
# ---------------------------------------------------------------------------
@mcp.tool()
def refine_concept_image(image_path: str, prompt: str) -> dict:
    """Gemini 이미지 모델로 컨셉 이미지를 3D 변환하기 좋은 형태로 다시 그린다.

    prompt에는 '무엇을 유지하고 어떻게 바꿀지'를 영어로 구체적으로 적는다.
    예: "Keep the character's red scarf and round silhouette. Render as a single
    object, full body, 3/4 front view, plain white background, soft studio light."
    반환값의 refined_path를 image_to_3d에 넘기면 된다.
    """
    from google import genai

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        return {"error": ".env에 GOOGLE_API_KEY가 없습니다. 정제 단계를 건너뛰고 원본으로 진행하세요."}

    src = Image.open(_resolve(image_path))
    client = genai.Client()
    resp = client.models.generate_content(model=GEMINI_IMAGE_MODEL, contents=[prompt, src])

    for cand in resp.candidates or []:
        for part in (cand.content.parts if cand.content else []) or []:
            if part.inline_data and part.inline_data.data:
                out = REFINED_DIR / _new_name(".png")
                out.write_bytes(part.inline_data.data)
                return {"refined_path": str(out), "model": GEMINI_IMAGE_MODEL}

    return {"error": "이미지가 생성되지 않았습니다. 프롬프트를 바꾸거나 원본 이미지로 진행하세요.", "text": resp.text}


# ---------------------------------------------------------------------------
# 3. 이미지 → 3D 모델
# ---------------------------------------------------------------------------
def _fix_glb_materials(path: Path) -> None:
    """metallicFactor가 빠진 재질을 비금속(0)으로 고친다.

    glTF는 metallicFactor가 없으면 1.0(완전 금속)으로 해석한다. TRELLIS 결과물은 이 값을
    넣지 않아서 뷰어에서 거의 검게 보이므로, 금속 텍스처가 없는 재질은 0으로 채운다.
    """
    data = path.read_bytes()
    if data[:4] != b"glTF":
        return
    json_len = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20 : 20 + json_len])

    changed = False
    for mat in gltf.get("materials", []):
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        if "metallicFactor" not in pbr and "metallicRoughnessTexture" not in pbr:
            pbr["metallicFactor"] = 0.0
            changed = True
    if not changed:
        return

    chunk = json.dumps(gltf, separators=(",", ":")).encode()
    chunk += b" " * (-len(chunk) % 4)  # GLB 청크는 4바이트 정렬
    rest = data[20 + json_len :]
    body = struct.pack("<II", len(chunk), 0x4E4F534A) + chunk + rest
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)


def _trellis_generate(image: Path, seed: int) -> dict:
    """Hugging Face Space의 TRELLIS를 호출한다. (무료, 대기열/할당량 제한 있음)"""
    from gradio_client import Client, handle_file

    client = Client(TRELLIS_SPACE, token=os.getenv("HF_TOKEN") or None, verbose=False)
    client.predict(api_name="/start_session")
    video, glb, _download = client.predict(
        image=handle_file(str(image)),
        multiimages=[],
        seed=seed,
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        slat_guidance_strength=3.0,
        slat_sampling_steps=12,
        multiimage_algo="stochastic",
        mesh_simplify=0.95,
        texture_size=1024,
        api_name="/generate_and_extract_glb",
    )

    name = _new_name("")
    glb_out = MODELS_DIR / f"{name}.glb"
    shutil.copy(glb, glb_out)
    result = {"glb_path": str(glb_out)}

    video_path = video.get("video") if isinstance(video, dict) else video
    if video_path and Path(video_path).exists():
        video_out = MODELS_DIR / f"{name}.mp4"
        shutil.copy(video_path, video_out)
        result["preview_video"] = str(video_out)
    return result


def _tripo_generate(image: Path) -> dict:
    """Tripo API로 변환한다. (유료 크레딧, .env에 TRIPO_API_KEY 필요)"""
    key = os.getenv("TRIPO_API_KEY")
    if not key:
        raise RuntimeError(".env에 TRIPO_API_KEY가 없습니다.")
    headers = {"Authorization": f"Bearer {key}"}

    with open(image, "rb") as f:
        up = requests.post(f"{TRIPO_API}/upload", headers=headers, files={"file": (image.name, f, "image/png")}, timeout=60)
    up.raise_for_status()
    token = up.json()["data"]["image_token"]

    task = requests.post(
        f"{TRIPO_API}/task",
        headers=headers,
        json={"type": "image_to_model", "file": {"type": image.suffix.lstrip(".") or "png", "file_token": token}},
        timeout=60,
    )
    task.raise_for_status()
    task_id = task.json()["data"]["task_id"]

    deadline = time.time() + 600
    while time.time() < deadline:
        data = requests.get(f"{TRIPO_API}/task/{task_id}", headers=headers, timeout=30).json()["data"]
        status = data.get("status")
        if status == "success":
            out = data.get("output") or data.get("result") or {}
            url = out.get("pbr_model") or out.get("model")
            if isinstance(url, dict):
                url = url.get("url")
            glb_out = MODELS_DIR / _new_name(".glb")
            glb_out.write_bytes(requests.get(url, timeout=120).content)
            return {"glb_path": str(glb_out), "task_id": task_id}
        if status in ("failed", "cancelled", "banned", "expired", "unknown"):
            raise RuntimeError(f"Tripo 작업 실패: {status}")
        time.sleep(5)
    raise TimeoutError("Tripo 작업이 10분 안에 끝나지 않았습니다.")


@mcp.tool()
def image_to_3d(image_path: str, backend: str = "", seed: int = 0) -> dict:
    """이미지를 텍스처가 입혀진 3D 모델(.glb)로 변환한다. 1~3분 정도 걸린다.

    backend: "trellis"(무료, Hugging Face Space) 또는 "tripo"(유료 API).
             비우면 .env의 THREED_BACKEND 값(기본 trellis)을 쓴다.
    배경이 깨끗하고 피사체 하나가 잘 보이는 이미지일수록 결과가 좋다.
    """
    backend = (backend or os.getenv("THREED_BACKEND", "trellis")).lower()
    image = _resolve(image_path)
    started = time.time()
    try:
        if backend == "tripo":
            result = _tripo_generate(image)
        else:
            backend = "trellis"
            result = _trellis_generate(image, seed)
    except Exception as e:  # 에이전트가 원인을 보고 사용자에게 설명할 수 있게 에러를 그대로 돌려준다
        return {"error": f"{type(e).__name__}: {e}", "backend": backend}

    _fix_glb_materials(Path(result["glb_path"]))
    result.update(backend=backend, seconds=round(time.time() - started, 1))
    return result


@mcp.tool()
def list_outputs(limit: int = 10) -> list[str]:
    """지금까지 생성된 3D 모델(.glb) 경로를 최신순으로 반환한다."""
    files = sorted(MODELS_DIR.glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files[:limit]]


if __name__ == "__main__":
    mcp.run()
