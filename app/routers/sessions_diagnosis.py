# pytune_diagnosis/sessions_diagnosis.py
from fastapi import APIRouter, Depends, HTTPException
from tortoise.exceptions import DoesNotExist
from pytune_auth_common.models.schema import UserOut
from pytune_auth_common.services.auth_checks import get_current_user
from pytune_data.models import (
    DiagnosisSession,
    DiagnosisNote,
    DiagnosisSessionStatus,
    User,
    UserPianoModel
)

from pytune_data.schemas import (
    CreateSessionRequest, 
    UpdateSessionStatusRequest, 
    UpdateSessionStatusRequest,
    AddNoteRequest
)
from typing import List, Optional



router = APIRouter(prefix="/sessions", tags=["diagnosis-sessions"])
# -----------------------------
# Routes CRUD Session
# -----------------------------
@router.post("/", summary="Create a new diagnosis session")
async def create_session(
    req: CreateSessionRequest,
    user_out: UserOut = Depends(get_current_user)
):
    user = await User.get(id=user_out.id)

    # Vérifier s'il existe déjà une session active pour ce piano
    existing = await DiagnosisSession.filter(
        user=user,
        pianomodel_user_id=req.pianomodel_user_id
    ).order_by("-id").first()

    if existing:
        if existing.status == DiagnosisSessionStatus.RUNNING:
            return {
                "session_id": existing.id,
                "status": existing.status.name,
                "reused": True
            }

        if existing.status == DiagnosisSessionStatus.CREATED:
            await existing.delete()

    # Sinon créer une nouvelle session
    session = await DiagnosisSession.create(
        user=user,
        pianomodel_user_id=req.pianomodel_user_id,
        status=DiagnosisSessionStatus.CREATED,
        data=req.data or {}
    )

    return {
        "session_id": session.id,
        "status": session.status.name,
        "reused": False
    }

@router.get("/", summary="List diagnosis sessions for current user")
async def list_user_sessions(
    user_out: UserOut = Depends(get_current_user),
    pianomodel_user_id: Optional[int] = None,
    active_only: bool = False,
    latest: bool = False,
    limit: int = 50
):
    """
    Flexible session listing:
    - /sessions → all sessions
    - /sessions?active_only=true → active only
    - /sessions?pianomodel_user_id=12 → for specific piano
    - /sessions?active_only=true&pianomodel_user_id=12 → active for that piano
    - /sessions?latest=true → return only the latest session (with filters)
    """
    user = await User.get(id=user_out.id)

    qs = DiagnosisSession.filter(user=user)

    # Filter by piano
    if pianomodel_user_id:
        qs = qs.filter(pianomodel_user_id=pianomodel_user_id)

    # Filter active sessions only
    if active_only:
        active_statuses = [
            DiagnosisSessionStatus.CREATED,
            DiagnosisSessionStatus.RUNNING
        ]
        qs = qs.filter(status__in=active_statuses)

    # Order newest first
    qs = qs.order_by("-id")

    # Return only the latest if requested
    if latest:
        session = await qs.first()
        return session

    # Otherwise return a list
    sessions = await qs.limit(limit)
    return sessions


@router.get("/{session_id}", summary="Get diagnosis session metadata")
async def get_session(session_id: int, user_out: UserOut = Depends(get_current_user)):
    try:
        user = await User.get(id=user_out.id)
        session = await DiagnosisSession.get(id=session_id, user=user)
        return session
    except DoesNotExist:
        raise HTTPException(404, "Session not found")


@router.patch("/{session_id}", summary="Update session status or data")
async def update_session(
    session_id: int,
    req: UpdateSessionStatusRequest,
    user_out: UserOut = Depends(get_current_user)
):
    try:
        user = await User.get(id=user_out.id)
        session = await DiagnosisSession.get(id=session_id, user=user)
    except DoesNotExist:
        raise HTTPException(404, "Session not found")

    session.status = req.status
    if req.data:
        session.data = {**(session.data or {}), **req.data}
    await session.save()

    return {"ok": True, "status": session.status.name}


@router.delete("/{session_id}", summary="Cancel/Delete a diagnosis session")
async def delete_session(session_id: int, user_out: UserOut = Depends(get_current_user)):
    try:
        user = await User.get(id=user_out.id)
        session = await DiagnosisSession.get(id=session_id, user=user)
    except DoesNotExist:
        raise HTTPException(404, "Session not found")

    session.status = DiagnosisSessionStatus.CANCELLED
    await session.save()
    return {"ok": True}
    

# -----------------------------
# Routes NOTES (= résultat par note)
# -----------------------------

@router.post("/{session_id}/notes", summary="Add a diagnostic note")
async def add_note(
    session_id: int,
    req: AddNoteRequest,
    user_out: UserOut = Depends(get_current_user)
):
    try:
        user = await User.get(id=user_out.id)
        session = await DiagnosisSession.get(id=session_id, user=user)
        if session.status == DiagnosisSessionStatus.CREATED:
            session.status = DiagnosisSessionStatus.RUNNING
            await session.save()
    except DoesNotExist:
        raise HTTPException(404, "Session not found")

    note = await DiagnosisNote.create(
        session_id=session_id,
        midi=req.midi,
        note_name=req.note_name,
        f0=req.f0,
        deviation_cents=req.deviation_cents,
        confidence=req.confidence,
        inharmonicity=req.inharmonicity,

        # 🔥 Ajouter tous les champs optionnels
        B_estimate=req.B_estimate,
        partials=req.partials,
        inharmonicity_curve=req.inharmonicity_curve,
        spectral_fingerprint=req.spectral_fingerprint,
        unison=req.unison,
    )

    # pass-thru to websocket if needed ???
    return {"ok": True, "note_id": note.id}


@router.get("/{session_id}/notes", summary="List notes for a diagnosis session")
async def list_notes(session_id: int, user_out: UserOut = Depends(get_current_user)):
    try:
        user = await User.get(id=user_out.id)
        session = await DiagnosisSession.get(id=session_id, user=user)
    except DoesNotExist:
        raise HTTPException(404, "Session not found")

    notes = await DiagnosisNote.filter(session=session).order_by("midi")
    return notes