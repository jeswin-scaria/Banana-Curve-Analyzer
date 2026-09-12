"""
🍌 Banana Curve Analyzer — FastAPI Backend Server
High-performance REST API for Render backend serving the Vercel frontend.
"""

import io
import base64
from typing import Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.analyzer import analyze_banana, CurvatureAnalysisResult
from src.visualization import create_annotated_overlay, create_isolated_shape_overlay

app = FastAPI(
    title="Banana Curve Analyzer API",
    description="High-precision computer vision API for geometric curvature analysis of bananas.",
    version="2.0.0",
)

# Enable CORS for Vercel and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mat_to_base64(mat_rgb: np.ndarray, format: str = "JPEG", quality: int = 88) -> str:
    """Encode an RGB numpy image array into a base64 data URL."""
    bgr = cv2.cvtColor(mat_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Banana Curve Analyzer API",
        "endpoints": {
            "health": "/health",
            "analyze": "POST /api/analyze",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "banana-curve-analyzer", "ready": True}


@app.post("/api/analyze")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    segmentation_method: str = Form("auto"),
):
    """
    Analyze the curvature of an uploaded banana image.
    Supports either multipart file upload or base64 form parameter.
    """
    img_bytes = None

    if file is not None:
        img_bytes = await file.read()
    elif image_base64 is not None:
        b64_data = image_base64
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(b64_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 string: {e}")
    else:
        raise HTTPException(status_code=400, detail="No image provided. Upload a file or pass image_base64.")

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty image data.")

    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_rgb = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    # Run core geometric analysis pipeline
    result = analyze_banana(image_rgb, segmentation_method=segmentation_method)

    if not result.success:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "message": result.message or "No banana detected in image.",
                "failure_stage": result.failure_stage,
                "quality_flags": result.quality_flags,
            },
        )

    # Generate visual overlays
    annotated_overlay = create_annotated_overlay(image_rgb, result)
    isolated_overlay = create_isolated_shape_overlay(image_rgb, result)

    annotated_b64 = _mat_to_base64(annotated_overlay)
    isolated_b64 = _mat_to_base64(isolated_overlay)

    return {
        "success": True,
        "curve_score": round(result.curve_score, 2),
        "category": result.category,
        "path_length_px": round(result.path_length, 2),
        "chord_distance_px": round(result.chord_distance, 2),
        "max_deflection_px": round(result.max_deflection, 2),
        "deflection_ratio": round(result.deflection_ratio, 4),
        "confidence_score": round(result.confidence_score, 2),
        "quality_status": result.quality_status,
        "segmentation_method": result.segmentation_method,
        "endpoint_1": [round(c, 1) for c in result.endpoint_1] if result.endpoint_1 else None,
        "endpoint_2": [round(c, 1) for c in result.endpoint_2] if result.endpoint_2 else None,
        "apex_point": [round(c, 1) for c in result.apex_point] if result.apex_point else None,
        "annotated_image": annotated_b64,
        "isolated_image": isolated_b64,
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
