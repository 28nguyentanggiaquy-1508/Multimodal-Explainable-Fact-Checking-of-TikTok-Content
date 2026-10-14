import sqlite3
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]

DB_PATH = (
    BACKEND_DIR
    / "data"
    / "tiktok_analyzer.db"
)


def get_connection():
    """
    Mỗi request/service call dùng connection riêng.

    WAL giúp đọc/ghi đồng thời ổn định hơn trong FastAPI local.
    foreign_keys bật để cascade delete hoạt động.
    """
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    conn.execute(
        "PRAGMA foreign_keys = ON;"
    )

    conn.execute(
        "PRAGMA journal_mode = WAL;"
    )

    return conn


def init_db():
    conn = get_connection()

    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                video_id TEXT NOT NULL,
                video_url TEXT NOT NULL,

                model_id TEXT,
                emotion_model_version TEXT,
                ate_model_version TEXT,
                asc_model_version TEXT,

                created_at TEXT NOT NULL,

                comment_count INTEGER NOT NULL DEFAULT 0,

                context_json TEXT,
                summary_json TEXT,
                emotions_json TEXT,
                absa_summary_json TEXT,

                insight TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_analyses_video_id
            ON analyses(video_id);

            CREATE INDEX IF NOT EXISTS idx_analyses_created_at
            ON analyses(created_at DESC);


            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                analysis_id INTEGER NOT NULL,

                source_comment_id TEXT,
                text TEXT NOT NULL,

                emotion_label TEXT,
                emotion_confidence REAL,

                all_labels_json TEXT,
                probabilities_json TEXT,
                metadata_json TEXT,

                FOREIGN KEY (analysis_id)
                    REFERENCES analyses(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_comments_analysis_id
            ON comments(analysis_id);

            CREATE INDEX IF NOT EXISTS idx_comments_source_comment_id
            ON comments(source_comment_id);


            CREATE TABLE IF NOT EXISTS aspect_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                comment_row_id INTEGER NOT NULL,

                term TEXT NOT NULL,
                start_pos INTEGER,
                end_pos INTEGER,

                sentiment TEXT,
                confidence REAL,

                probabilities_json TEXT,

                ate_model_version TEXT,
                asc_model_version TEXT,

                FOREIGN KEY (comment_row_id)
                    REFERENCES comments(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_aspects_comment_row_id
            ON aspect_results(comment_row_id);

            CREATE INDEX IF NOT EXISTS idx_aspects_term
            ON aspect_results(term);
            """
        )

        conn.commit()

    finally:
        conn.close()
