"""Root compatibility wrapper for app.db.database."""

from app.db.database import (
    DB_PATH,
    add_face_embedding,
    check_db_health,
    get_all_face_embeddings,
    get_db_connection,
    get_person_context,
    init_db,
    save_interaction_log,
)
