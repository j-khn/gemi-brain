"""Perceive route: POST /api/perceive."""

from typing import Optional
from fastapi import APIRouter, HTTPException

from app.db.database import (
    get_all_face_embeddings,
    add_face_embedding,
    get_person_context,
    get_conversation_history,
    get_conversation_history_by_query,
    get_recent_conversation_history,
    get_person_id_by_name,
    save_conversation_turn,
    save_interaction_log,
)
from app.services import ollama, vision
from app.schemas.perceive import PerceiveRequest, PerceiveResponse

router = APIRouter()


@router.post("/api/perceive", response_model=PerceiveResponse)
async def perceive(request: PerceiveRequest):
    """
    Perception endpoint.

    Accepts text prompt, optional camera image frame, and sensor status.
    Fetches memory context → runs face recognition → calls Ollama → logs interaction → returns speech + action.
    """
    try:
        # ------------------------------------------------------------------
        # 1. Vision: decode image and identify person (face recognition is required for image input)
        # ------------------------------------------------------------------
        identified_person: Optional[str] = None
        face_detected = False
        vision_available = bool(request.image_frame)

        if request.image_frame:
            if not vision.is_face_recognition_available():
                raise HTTPException(
                    status_code=503,
                    detail="Face recognition library unavailable. Install face_recognition to use image-based perception."
                )

            image_bytes = vision.decode_image_base64(request.image_frame)
            query_emb = vision.extract_face_embedding(image_bytes)
            face_detected = True
            stored_embeddings = get_all_face_embeddings()
            match = vision.match_face(query_emb, stored_embeddings)
            if match:
                person_id, name, cos_sim, euc_dist = match
                if cos_sim > 0.85 or euc_dist < 0.42:
                    identified_person = name
                    if euc_dist > 0.2:
                        add_face_embedding(person_id, query_emb)
            
        # ------------------------------------------------------------------
        # 2. Detect identity mismatch and resolve effective person name
        # ------------------------------------------------------------------
        identity_mismatch = False
        claimed_name: Optional[str] = request.person_name

        if identified_person:
            effective_name = identified_person
            if claimed_name and claimed_name.lower() != identified_person.lower():
                identity_mismatch = True
        elif face_detected and claimed_name:
            effective_name = None
            identity_mismatch = True
        elif claimed_name:
            effective_name = claimed_name
        else:
            effective_name = None

        memory_context = get_person_context(effective_name)
        person_id = None
        if identified_person:
            person_id = get_person_id_by_name(identified_person)
        elif effective_name:
            person_id = get_person_id_by_name(effective_name)

        if person_id:
            conversation_history = get_conversation_history(person_id, limit=10)
        else:
            # No identified person: use recent interactions from the past 10 minutes
            conversation_history = get_recent_conversation_history(limit=10, minutes=10)
            if request.prompt:
                query_history = get_conversation_history_by_query(request.prompt, limit=10)
                if query_history:
                    conversation_history = query_history

        # ------------------------------------------------------------------
        # 3. Build enriched prompt (inject mismatch warning if needed)
        # ------------------------------------------------------------------
        enriched_prompt = request.prompt or ""
        if identity_mismatch:
            if identified_person:
                enriched_prompt = (
                    f"{enriched_prompt} "
                    f"[SYSTEM NOTE: User claims to be '{claimed_name}' but face recognition "
                    f"identified them as '{identified_person}'. Politely challenge/verify their identity.]"
                ).strip()
            else:
                enriched_prompt = (
                    f"{enriched_prompt} "
                    f"[SYSTEM NOTE: User claims to be '{claimed_name}', but the face could not be matched to that identity. Politely verify their name.]"
                ).strip()

        # ------------------------------------------------------------------
        # 4. Query Ollama model
        # ------------------------------------------------------------------
        result = await ollama.perceive_and_generate(
            prompt=enriched_prompt or request.prompt,
            image_frame=request.image_frame,
            memory_context=memory_context,
            sensor_status=request.sensor_status,
            conversation_history=conversation_history,
            vision_available=vision_available,
            identified_person=identified_person,
            face_detected=face_detected,
        )

        speech = result.get("speech", "Hello!")
        action = result.get("action", "stop_and_look")

        # ------------------------------------------------------------------
        # 5. Log interaction
        # ------------------------------------------------------------------
        prompt_summary = request.prompt or "Visual/Sensor input"
        save_interaction_log(f"Prompt: {prompt_summary} -> Speech: '{speech}' | Action: {action}")
        save_conversation_turn(person_id, prompt_summary, speech)

        return PerceiveResponse(speech=speech, action=action)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
