"""컨셉 이미지 → 3D 모델 변환 도구를 제공하는 MCP 서버 (stdio).

ADK 에이전트가 McpToolset으로 이 서버를 띄워서 도구를 호출한다.
이미지는 base64가 아니라 항상 "파일 경로"로 주고받는다. (LLM 컨텍스트 절약)

도구 목록
- concept_to_3d        : 전처리 → 정제 → 3D 변환을 코드로 한 번에 실행 (에이전트는 이것만 사용)
- prepare_image        : 3D 변환에 맞게 이미지 정리 (회전 보정, 리사이즈)
- refine_concept_image : 정제 단계만 실행 (off | rembg | gemini)
- image_to_3d          : 이미지를 3D 모델(.glb)로 변환 (backend: trellis | tripo)
- list_outputs         : 지금까지 만든 3D 모델 목록

정해진 순서는 LLM이 도구를 하나씩 고르게 하지 않고 concept_to_3d 안의 코드로 처리한다.
LLM 호출 횟수와 이미지 재전송을 줄이기 위해서다. (README "토큰 사용량 문제와 해결 과정")
"""

from __future__ import annotations

import hashlib
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
CACHE_FILE = WORKSPACE / "cache.json"
for d in (PREPARED_DIR, REFINED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 정제 방식: off(안 함) | rembg(로컬 배경 제거, 무료) | gemini(Gemini 이미지 모델로 다시 그리기, 유료)
REFINE_BACKEND = os.getenv("REFINE_BACKEND", "rembg")
# 쉼표로 여러 개를 적으면 앞의 모델이 없을 때(종료, 이름 변경) 다음 모델로 넘어간다
GEMINI_IMAGE_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image,gemini-3-pro-image").split(",")
    if m.strip()
]
REMBG_MODEL = os.getenv("REMBG_MODEL", "u2net")
TRELLIS_SPACE = os.getenv("TRELLIS_SPACE", "trellis-community/TRELLIS")
TRIPO_API = "https://api.tripo3d.ai/v2/openapi"

mcp = MCPServer(
    "concept3d-tools",
    instructions="컨셉 이미지를 3D 모델(.glb)로 변환하는 도구 모음. 보통은 concept_to_3d 하나만 쓰면 된다. 모든 이미지는 파일 경로로 전달한다.",
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
    """이미지의 EXIF 회전을 보정하고 긴 변을 size 이하로 줄여 PNG로 저장한다.

    투명 배경은 그대로 유지한다. 반환값의 prepared_path를 다음 도구에 넘긴다.
    """
    img = ImageOps.exif_transpose(Image.open(_resolve(image_path)))
    img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")
    img.thumbnail((size, size), Image.LANCZOS)

    out = PREPARED_DIR / _new_name(".png")
    img.save(out)
    return {"prepared_path": str(out), "size": list(img.size)}


# ---------------------------------------------------------------------------
# 2. 정제 (off | rembg | gemini)
# ---------------------------------------------------------------------------
_rembg_session = None


def _remove_background(img: Image.Image) -> Image.Image:
    """TRELLIS가 기대하는 입력 형식으로 만든다.

    TRELLIS Space의 API(generate_and_extract_glb)는 preprocess_image=False로 실행된다.
    웹 UI는 업로드할 때 전처리를 하지만 API 호출은 그 단계를 건너뛰므로, 같은 처리를 여기서 한다.
    (microsoft/TRELLIS pipelines/trellis_image_to_3d.py의 preprocess_image와 같은 방식)
      1. 투명 배경이 없으면 rembg로 배경 제거
      2. 알파 80% 이상 영역 기준 1.2배 정사각형으로 크롭
      3. 518x518로 리사이즈하고 알파를 곱해 검은 배경 RGB로 변환
    """
    global _rembg_session
    import numpy as np

    has_alpha = img.mode == "RGBA" and img.getchannel("A").getextrema()[0] < 255
    if has_alpha:
        cut = img
    else:
        from rembg import new_session, remove

        if _rembg_session is None:
            # 첫 호출 때 모델 가중치를 ~/.u2net에 내려받는다 (u2net 약 170MB, 한 번만)
            _rembg_session = new_session(REMBG_MODEL)
        cut = remove(img.convert("RGB"), session=_rembg_session).convert("RGBA")

    alpha = np.array(cut)[:, :, 3]
    ys, xs = np.nonzero(alpha > 0.8 * 255)
    if len(xs) == 0:
        raise RuntimeError("배경 제거 후 피사체를 찾지 못했습니다. refine_backend=off로 다시 시도하세요.")
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.2) // 2
    cut = cut.crop((cx - half, cy - half, cx + half, cy + half)).resize((518, 518), Image.LANCZOS)

    rgba = np.array(cut).astype(np.float32) / 255
    return Image.fromarray((rgba[:, :, :3] * rgba[:, :, 3:4] * 255).astype(np.uint8))


def _gemini_redraw(img: Image.Image, prompt: str) -> tuple[Image.Image, str]:
    """후보 모델을 순서대로 시도해 이미지를 다시 그린다. (모델, 이미지)를 반환."""
    from google import genai
    from google.genai import errors

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError(".env에 GOOGLE_API_KEY가 없습니다.")
    client = genai.Client()

    tried = []
    for model in GEMINI_IMAGE_MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=[prompt, img])
        except errors.ClientError as e:
            if e.code == 404:  # 모델이 없거나 종료됨 → 다음 후보로
                tried.append(f"{model}(없음)")
                continue
            raise
        for cand in resp.candidates or []:
            for part in (cand.content.parts if cand.content else None) or []:
                if part.inline_data and part.inline_data.data:
                    from io import BytesIO

                    return Image.open(BytesIO(part.inline_data.data)), model
        tried.append(f"{model}(이미지 없음)")
    raise RuntimeError(
        f"사용할 수 있는 이미지 모델이 없습니다: {', '.join(tried)}. .env의 GEMINI_IMAGE_MODEL을 확인하세요."
    )


@mcp.tool()
def refine_concept_image(image_path: str, prompt: str = "", backend: str = "") -> dict:
    """3D 변환 전에 이미지를 정리한다.

    backend: "off"   → 아무것도 하지 않음
             "rembg" → 로컬에서 배경 제거 + 피사체 중앙 정렬 (무료, 기본)
             "gemini"→ Gemini 이미지 모델로 prompt대로 다시 그린 뒤 배경 제거 (유료, 스타일 변환용)
    비우면 .env의 REFINE_BACKEND를 쓴다. 반환값의 refined_path를 image_to_3d에 넘긴다.
    """
    backend = (backend or REFINE_BACKEND).lower()
    src = _resolve(image_path)
    if backend == "off":
        return {"refined_path": str(src), "backend": "off"}

    img = Image.open(src)
    result: dict = {"backend": backend}
    if backend == "gemini":
        if not prompt:
            return {"error": "gemini 정제에는 prompt가 필요합니다."}
        img, result["model"] = _gemini_redraw(img, prompt)
    elif backend != "rembg":
        return {"error": f"알 수 없는 정제 방식: {backend} (off | rembg | gemini)"}

    img = _remove_background(img)
    out = REFINED_DIR / _new_name(".png")
    img.save(out)
    result["refined_path"] = str(out)
    return result


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
    """이미지를 텍스처가 입혀진 3D 모델(.glb)로 변환한다. 30초~수 분 걸린다.

    backend: "trellis"(무료, Hugging Face Space) 또는 "tripo"(유료 API).
             비우면 .env의 THREED_BACKEND 값(기본 trellis)을 쓴다.
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


# ---------------------------------------------------------------------------
# 4. 한 번에 실행 + 캐시
# ---------------------------------------------------------------------------
def _cache_load() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _cache_key(image: Path, **options) -> str:
    h = hashlib.sha256(image.read_bytes())
    h.update(json.dumps(options, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


@mcp.tool()
def concept_to_3d(
    image_path: str,
    refine_backend: str = "",
    refine_prompt: str = "",
    threed_backend: str = "",
    seed: int = 0,
) -> dict:
    """컨셉 이미지를 3D 모델(.glb)로 만든다. 전처리 → 정제 → 3D 변환을 순서대로 한 번에 실행한다.

    refine_backend: "off" | "rembg"(기본, 무료 배경 제거) | "gemini"(유료, refine_prompt대로 스타일 변환)
    refine_prompt : gemini 정제일 때만 쓰는 영어 프롬프트
    threed_backend: "trellis"(기본, 무료) | "tripo"(유료)
    seed          : 같은 이미지로 다른 결과를 원할 때 바꾼다
    같은 이미지와 옵션으로 이미 만든 결과가 있으면 바로 돌려준다 (cached=true).
    """
    started = time.time()
    src = _resolve(image_path)
    refine_backend = (refine_backend or REFINE_BACKEND).lower()
    threed_backend = (threed_backend or os.getenv("THREED_BACKEND", "trellis")).lower()
    options = dict(
        refine_backend=refine_backend,
        refine_prompt=refine_prompt if refine_backend == "gemini" else "",
        threed_backend=threed_backend,
        seed=seed,
    )

    key = _cache_key(src, **options)
    cache = _cache_load()
    hit = cache.get(key)
    if hit and Path(hit.get("glb_path", "")).exists():
        return {**hit, "cached": True, "seconds": round(time.time() - started, 1)}

    steps = []
    prepared = prepare_image(str(src))
    steps.append("prepare")

    # 정제가 실패해도 멈추지 않는다: gemini 실패 → rembg → 전처리 이미지 그대로
    refined: dict = {}
    for backend in dict.fromkeys([refine_backend, "rembg"] if refine_backend == "gemini" else [refine_backend]):
        try:
            refined = refine_concept_image(prepared["prepared_path"], refine_prompt, backend)
        except Exception as e:
            refined = {"error": f"{type(e).__name__}: {e}"}
        if not refined.get("error"):
            break
        steps.append(f"refine:{backend} 실패 ({refined['error']})")
    if refined.get("error"):
        mesh_input = prepared["prepared_path"]
    else:
        steps.append(f"refine:{refined['backend']}" + (f"({refined['model']})" if refined.get("model") else ""))
        mesh_input = refined["refined_path"]

    mesh = image_to_3d(mesh_input, threed_backend, seed)
    if mesh.get("error"):
        return {"error": mesh["error"], "steps": steps, "prepared_path": prepared["prepared_path"]}
    steps.append(f"3d:{mesh['backend']}")

    result = {
        "glb_path": mesh["glb_path"],
        "preview_video": mesh.get("preview_video"),
        "prepared_path": prepared["prepared_path"],
        "refined_path": mesh_input if mesh_input != prepared["prepared_path"] else None,
        "steps": steps,
        "options": options,
    }
    if any("실패" in s for s in steps):
        # 일시적인 실패로 만든 결과는 캐시하지 않는다. 다음에 다시 시도하게 둔다.
        return {**result, "cached": False, "seconds": round(time.time() - started, 1)}
    cache[key] = result
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return {**result, "cached": False, "seconds": round(time.time() - started, 1)}


@mcp.tool()
def list_outputs(limit: int = 10) -> list[str]:
    """지금까지 생성된 3D 모델(.glb) 경로를 최신순으로 반환한다."""
    files = sorted(MODELS_DIR.glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files[:limit]]


if __name__ == "__main__":
    mcp.run()
