"""
API client used by the detection pipeline to send violation events to the Node backend.

This is intentionally lightweight and demo-friendly.
If the backend is unavailable, calls fail silently (the pipeline should keep running).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


class APIClient:
    def __init__(self, base_url: Optional[str] = None, timeout_s: float = 3.0) -> None:
        # Example: http://localhost:5000/api
        self.base_url = (base_url or os.getenv("BACKEND_API_BASE_URL") or "http://localhost:5000/api").rstrip(
            "/"
        )
        self.timeout_s = float(timeout_s)

    def send_violation(self, payload: Dict[str, Any]) -> bool:
        """
        POST a violation to the backend.
        Expected backend endpoint (you can match in Node later):
            POST {base_url}/violations
        """
        url = f"{self.base_url}/violations"
        # Normalize Python pipeline payload to backend schema.
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        normalized = {
            "vehicleNumber": (meta.get("plate_text") if isinstance(meta, dict) else "") or "N/A",
            "violationType": payload.get("type", "unknown"),
            "timestamp": payload.get("timestamp"),
            "imageUrl": payload.get("imagePath", ""),
            # Keep original payload for debugging/compatibility.
            "raw": payload,
        }
        try:
            resp = requests.post(url, json=normalized, timeout=self.timeout_s)
            return 200 <= resp.status_code < 300
        except Exception:
            return False

