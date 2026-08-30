"""Perceive route: POST /api/perceive."""

from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException

from app.db.database import (
    get_all_face_embeddings,
    add_face_embedding,
    get_person_context,
    get_person_contexts,
    get_conversation_history,
    get_conversation_history_by_query,
    get_recent_conversation_history,
    get_recent_interaction_logs,
    get_standout_conversation_moments,
    get_latest_visual_moment,
    is_visual_anaphora,
    prepare_conversation_history_for_prompt,
    get_person_id_by_name,
    save_conversation_turn,
    save_interaction_log,
    get_setting,
    get_all_settings,
    set_setting,
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
        # 1. Vision: decode image frames and identify all unique people seen
        # ------------------------------------------------------------------
        identified_people: Dict[int, Dict] = {}  # person_id -> {name, best_cos, best_euc}
        unknown_faces_count = 0
        face_detected = False
        # vision_available is True if any frame provided
        vision_available = bool(request.image_frame or request.image_frames)

        frames: List[str] = []
        if request.image_frames:
            frames = request.image_frames
        elif request.image_frame:
            frames = [request.image_frame]

        # Enforce an image sequence limit configured at runtime
        image_limit = int(get_setting("image_sequence_limit", 6))
        if frames and len(frames) > image_limit:
            raise HTTPException(status_code=400, detail=f"Too many image frames; limit is {image_limit}.")

        if frames:
            try:
                # Canonicalize once so face recognition and Ollama receive the
                # exact same validated JPEG bytes.
                frames = [vision.normalize_image_base64(frame) for frame in frames]
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            if not vision.is_face_recognition_available():
                raise HTTPException(
                    status_code=503,
                    detail="Face recognition library unavailable. Install face_recognition to use image-based perception."
                )

            stored_embeddings = get_all_face_embeddings()
            # Iterate over all provided frames and attempt to find faces
            for img_str in frames:
                try:
                    image_bytes = vision.decode_image_base64(img_str)
                    query_emb = vision.extract_face_embedding(image_bytes)
                except ValueError:
                    # no face detected in this frame — skip
                    continue
                except RuntimeError:
                    raise HTTPException(
                        status_code=503,
                        detail="Face recognition library unavailable at runtime."
                    )

                face_detected = True
                match = vision.match_face(query_emb, stored_embeddings)
                if match:
                    person_id, name, cos_sim, euc_dist = match
                    existing = identified_people.get(person_id)
                    if not existing or cos_sim > existing.get("best_cos", -1):
                        identified_people[person_id] = {
                            "name": name,
                            "best_cos": cos_sim,
                            "best_euc": euc_dist,
                            "last_embedding": query_emb,
                        }
                        # If this is a reasonably close match, consider adding embedding to refine storage
                        # thresholds respect runtime settings if present
                        cos_thresh = float(get_setting("face_cosine_threshold", 0.85))
                        euc_thresh = float(get_setting("face_euclid_threshold", 0.42))
                        euc_add = float(get_setting("face_euclid_add_threshold", 0.2))
                        if (cos_sim > cos_thresh or euc_dist < euc_thresh) and euc_dist > euc_add:
                            add_face_embedding(person_id, query_emb)
                else:
                    unknown_faces_count += 1
            
        # ------------------------------------------------------------------
        # 2. Resolve claimed names, build aggregated memory context and conversation history
        # ------------------------------------------------------------------
        identity_mismatch = False
        claimed_name: Optional[str] = request.person_name

        visible_people_names: List[str] = []
        visible_people_descr: List[str] = []
        person_ids: List[int] = []

        # Collect recognized people
        for pid, info in identified_people.items():
            name = info["name"]
            visible_people_names.append(name)
            person_ids.append(pid)
            # include short notes/role from DB to make descriptions useful
            # fetch person record notes via get_person_context
            ctxt = get_person_context(name)
            if ctxt:
                visible_people_descr.append(f"{name} ({ctxt})")
            else:
                visible_people_descr.append(name)

        # If there were unknown faces, add a generic placeholder
        if unknown_faces_count > 0:
            visible_people_names.append("Unknown")
            visible_people_descr.append("Unknown person")

        # Resolve effective person for single-person flows and identity mismatch logic
        effective_name = None
        if len(visible_people_names) == 1:
            identified_name = visible_people_names[0]
            if identified_name != "Unknown":
                effective_name = identified_name
                if claimed_name and claimed_name.lower() != identified_name.lower():
                    identity_mismatch = True
            else:
                # unknown face but claimed name exists
                if claimed_name:
                    identity_mismatch = True
        else:
            # multiple people: ignore claimed_name for now, but note mismatch if single claim present
            if claimed_name and claimed_name not in visible_people_names:
                identity_mismatch = True

        # Build aggregated memory context for all identified people
        if visible_people_names:
            memory_context = get_person_contexts(visible_people_names)
        else:
            memory_context = ""

        # Conversation history: keep recent continuity; enrich with related older turns
        conversation_history = []
        if person_ids:
            for pid in person_ids:
                conversation_history.extend(get_conversation_history(pid, limit=8))
        if not conversation_history:
            conversation_history = get_recent_conversation_history(limit=16, minutes=60)
        if request.prompt:
            query_history = get_conversation_history_by_query(request.prompt, limit=6)
            if query_history:
                # Merge without replacing the recent thread.
                seen = {
                    (turn.get("user_prompt"), turn.get("gemi_response"), turn.get("timestamp"))
                    for turn in conversation_history
                }
                for turn in query_history:
                    key = (turn.get("user_prompt"), turn.get("gemi_response"), turn.get("timestamp"))
                    if key not in seen:
                        conversation_history.append(turn)
                        seen.add(key)

        standout_moments = []
        vision_color_hint = ""
        vision_low_info = False
        vision_similar_scene = False
        last_visual_speech = ""
        if frames:
            # Fresh image: clear prior visual semantic state completely.
            standout_moments = []
            conversation_history = [
                turn
                for turn in conversation_history
                if "[shared a picture]" not in (turn.get("user_prompt") or "").lower()
            ]
            try:
                raw = vision.decode_image_base64(frames[0])
                vision_color_hint = vision.dominant_color_label(raw)
                vision_low_info = vision.looks_low_information(raw)
                current_fp = vision.image_fingerprint(raw)
                previous_fp = get_setting("last_image_fingerprint", "") or ""
                last_visual_speech = get_setting("last_visual_speech", "") or ""
                if current_fp and previous_fp:
                    similarity = vision.fingerprint_similarity(current_fp, previous_fp)
                    vision_similar_scene = 0.72 <= similarity < 1.0
            except Exception:
                vision_color_hint = ""
                vision_low_info = False
                current_fp = ""
        elif is_visual_anaphora(request.prompt):
            # Ambiguous "that part / earlier" → only the newest visual tip.
            latest = get_latest_visual_moment(minutes=120)
            standout_moments = [latest] if latest else []
            latest_key = None
            if latest:
                latest_key = (
                    latest.get("user_prompt"),
                    latest.get("gemi_response"),
                    latest.get("timestamp"),
                )
            filtered = []
            for turn in conversation_history:
                is_visual = "[shared a picture]" in (turn.get("user_prompt") or "").lower()
                if not is_visual:
                    filtered.append(turn)
                    continue
                key = (turn.get("user_prompt"), turn.get("gemi_response"), turn.get("timestamp"))
                if latest_key and key == latest_key:
                    filtered.append(turn)
            conversation_history = filtered
        else:
            latest = get_latest_visual_moment(minutes=60)
            standout_moments = [latest] if latest else []

        conversation_history = prepare_conversation_history_for_prompt(
            conversation_history,
            current_prompt=request.prompt,
            limit=8 if frames else 12,
        )
        recent_notes = get_recent_interaction_logs(limit=8, minutes=60)
        if frames:
            recent_notes = [
                note
                for note in recent_notes
                if "[shared a picture]" not in note.lower()
                and "speech:" not in note.lower()
            ]

        prompt_l = (request.prompt or "").lower()
        user_hostile = any(
            cue in prompt_l
            for cue in (
                "stupid",
                "useless",
                "shut up",
                "idiot",
                "hate you",
                "dumb",
                "kill yourself",
                "say something mean",
                "say something really mean",
                "be mean",
            )
        )
        positive_tone = any(
            cue in prompt_l
            for cue in (
                "great day",
                "best day",
                "so happy",
                "having a great",
                "wonderful day",
                "feeling amazing",
                "everything is perfect",
            )
        )

        # ------------------------------------------------------------------
        # 3. Build enriched prompt (inject mismatch warning if needed)
        # ------------------------------------------------------------------
        enriched_prompt = request.prompt or ""
        if identity_mismatch:
            if visible_people_names and len(visible_people_names) == 1 and visible_people_names[0] != "Unknown":
                enriched_prompt = (
                    f"{enriched_prompt} "
                    f"[NOTE: User claims to be '{claimed_name}' but face recognition identified them as '{visible_people_names[0]}'. Politely verify their identity.]"
                ).strip()
            else:
                enriched_prompt = (
                    f"{enriched_prompt} "
                    f"[NOTE: Claimed name '{claimed_name}' does not match visible people. Politely verify names.]"
                ).strip()

        # ------------------------------------------------------------------
        # 4. Query Ollama model
        # ------------------------------------------------------------------
        # Prepare visible people text for Ollama (e.g. "Jam (father), Alice (daughter)")
        visible_desc = ", ".join(visible_people_descr) if visible_people_descr else ""

        result = await ollama.perceive_and_generate(
            prompt=enriched_prompt or request.prompt,
            image_frames=frames or None,
            memory_context=memory_context,
            recent_notes=recent_notes or None,
            standout_moments=standout_moments or None,
            sensor_status=request.sensor_status,
            conversation_history=conversation_history,
            vision_available=vision_available,
            visible_people=visible_people_names,
            visible_people_description=visible_desc,
            face_detected=face_detected,
            vision_color_hint=vision_color_hint or None,
            vision_low_info=vision_low_info,
            vision_similar_scene=vision_similar_scene,
            last_visual_speech=last_visual_speech or None,
            user_hostile=user_hostile,
            positive_tone_with_image=bool(positive_tone and frames),
        )

        speech = result.get("speech", "Hmm?")
        action = result.get("action", "stop_and_look")

        # ------------------------------------------------------------------
        # 5. Log interaction
        # ------------------------------------------------------------------
        prompt_summary = request.prompt or ("Shared a picture" if frames else "Just hanging out")
        if frames and request.prompt:
            prompt_summary = f"{request.prompt} [shared a picture]"
        elif frames and not request.prompt:
            prompt_summary = "Shared a picture"
        save_interaction_log(f"Prompt: {prompt_summary} -> Speech: '{speech}' | Action: {action}")
        if frames:
            try:
                raw = vision.decode_image_base64(frames[0])
                set_setting("last_image_fingerprint", vision.image_fingerprint(raw))
                set_setting("last_visual_speech", speech)
            except Exception:
                pass
        # Save conversation turn for each recognized person (associate with None for unknowns)
        if person_ids:
            for pid in person_ids:
                save_conversation_turn(pid, prompt_summary, speech)
        else:
            save_conversation_turn(None, prompt_summary, speech)

        return PerceiveResponse(speech=speech, action=action)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
