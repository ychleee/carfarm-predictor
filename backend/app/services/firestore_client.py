"""
Firebase Admin SDK 초기화 (싱글톤)

아이작 Firestore DB 접근용. 서비스 계정 키 경로:
  1. FIREBASE_SERVICE_ACCOUNT_KEY 환경변수
  2. 로컬: backend/issac-1c2b0-firebase-adminsdk-*.json
  3. Docker: /app/issac-service-account.json
"""

from __future__ import annotations

import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

_db = None


def get_firestore_db():
    """Firestore 클라이언트 싱글톤"""
    global _db
    if _db is not None:
        return _db

    key_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")

    if not key_path:
        # 로컬 / Docker 후보 경로
        backend_dir = Path(__file__).parent.parent.parent  # backend/
        project_dir = backend_dir.parent                    # CarFarm/
        candidates = (
            list(backend_dir.glob("issac-*-firebase-adminsdk-*.json"))
            + list(project_dir.glob("issac-*-firebase-adminsdk-*.json"))
            + [Path("/app/issac-service-account.json")]
        )
        for candidate in candidates:
            if candidate.exists():
                key_path = str(candidate)
                break

    if key_path:
        cred = credentials.Certificate(key_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
    else:
        # Cloud Run 등 ADC 사용 환경: 키 파일 없이 기본 자격증명 사용
        if not firebase_admin._apps:
            firebase_admin.initialize_app(options={
                "projectId": "issac-1c2b0",
            })

    _db = firestore.client()
    return _db
