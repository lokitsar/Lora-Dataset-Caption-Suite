import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .path_utils import ensure_directory, normalized_path


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DatasetManifest:
    def __init__(self, database_path):
        self.path = normalized_path(database_path)
        ensure_directory(self.path.parent)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS dataset_items (
                    item_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_relative_path TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    source_size INTEGER NOT NULL,
                    source_mtime_ns INTEGER NOT NULL,
                    output_image_path TEXT NOT NULL,
                    caption_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    normalization_status TEXT NOT NULL DEFAULT 'not_started',
                    analysis_status TEXT NOT NULL DEFAULT 'not_started',
                    analysis_json TEXT,
                    watermark_status TEXT NOT NULL DEFAULT 'not_requested',
                    cleanup_verification_status TEXT NOT NULL DEFAULT 'not_requested',
                    cleanup_verification_json TEXT,
                    crop_status TEXT NOT NULL DEFAULT 'not_requested',
                    crop_json TEXT,
                    caption_status TEXT NOT NULL DEFAULT 'not_started',
                    validation_status TEXT NOT NULL DEFAULT 'not_started',
                    error TEXT,
                    profile_id TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    caption_provider_version TEXT NOT NULL DEFAULT 'phase1',
                    cleanup_provider_version TEXT NOT NULL DEFAULT 'none',
                    cleanup_verifier_version TEXT NOT NULL DEFAULT 'none',
                    analysis_provider_version TEXT NOT NULL DEFAULT 'none',
                    crop_provider_version TEXT NOT NULL DEFAULT 'none',
                    review_status TEXT NOT NULL DEFAULT 'not_requested',
                    naming_sequence INTEGER,
                    output_naming_mode TEXT NOT NULL DEFAULT 'preserve_source_names',
                    lora_name TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_dataset_items_work
                    ON dataset_items(active, status, source_relative_path);
                CREATE INDEX IF NOT EXISTS idx_dataset_items_hash
                    ON dataset_items(source_hash);
                CREATE TABLE IF NOT EXISTS project_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(dataset_items)").fetchall()
            }
            if "normalization_status" not in columns:
                connection.execute(
                    "ALTER TABLE dataset_items ADD COLUMN normalization_status TEXT NOT NULL DEFAULT 'not_started'"
                )
            if "caption_provider_version" not in columns:
                connection.execute(
                    "ALTER TABLE dataset_items ADD COLUMN caption_provider_version TEXT NOT NULL DEFAULT 'phase1'"
                )
            if "cleanup_provider_version" not in columns:
                connection.execute(
                    "ALTER TABLE dataset_items ADD COLUMN cleanup_provider_version TEXT NOT NULL DEFAULT 'none'"
                )
            if "analysis_json" not in columns:
                connection.execute("ALTER TABLE dataset_items ADD COLUMN analysis_json TEXT")
            if "crop_json" not in columns:
                connection.execute("ALTER TABLE dataset_items ADD COLUMN crop_json TEXT")
            if "analysis_provider_version" not in columns:
                connection.execute(
                    "ALTER TABLE dataset_items ADD COLUMN analysis_provider_version TEXT NOT NULL DEFAULT 'none'"
                )
            if "crop_provider_version" not in columns:
                connection.execute(
                    "ALTER TABLE dataset_items ADD COLUMN crop_provider_version TEXT NOT NULL DEFAULT 'none'"
                )
            if "cleanup_verification_status" not in columns:
                connection.execute(
                    "ALTER TABLE dataset_items ADD COLUMN cleanup_verification_status TEXT NOT NULL DEFAULT 'not_requested'"
                )
            if "cleanup_verification_json" not in columns:
                connection.execute(
                    "ALTER TABLE dataset_items ADD COLUMN cleanup_verification_json TEXT"
                )
            if "cleanup_verifier_version" not in columns:
                connection.execute(
                    "ALTER TABLE dataset_items ADD COLUMN cleanup_verifier_version TEXT NOT NULL DEFAULT 'none'"
                )
            if "review_status" not in columns:
                connection.execute(
                    "ALTER TABLE dataset_items ADD COLUMN review_status TEXT NOT NULL DEFAULT 'not_requested'"
                )
            if "naming_sequence" not in columns:
                connection.execute(
                    "ALTER TABLE dataset_items ADD COLUMN naming_sequence INTEGER"
                )
            if "output_naming_mode" not in columns:
                connection.execute(
                    "ALTER TABLE dataset_items ADD COLUMN output_naming_mode TEXT NOT NULL DEFAULT 'preserve_source_names'"
                )
            if "lora_name" not in columns:
                connection.execute(
                    "ALTER TABLE dataset_items ADD COLUMN lora_name TEXT NOT NULL DEFAULT ''"
                )

    def sync(
        self,
        source_items,
        assignments,
        profile,
        caption_provider_version="phase1",
        cleanup_provider_version="none",
        cleanup_verifier_version="none",
        analysis_provider_version="none",
        crop_provider_version="none",
        output_naming_mode="preserve_source_names",
        lora_name="",
        preserve_mapping_ids=None,
    ):
        now = utc_now()
        preserve_mapping_ids = set(preserve_mapping_ids or ())
        with self._connect() as connection:
            connection.execute("UPDATE dataset_items SET active = 0")
            for item in source_items:
                assignment = assignments[item.item_id]
                output_image, caption_path = assignment[:2]
                naming_sequence = assignment[2] if len(assignment) > 2 else None
                existing = connection.execute(
                    "SELECT status, source_hash, output_image_path, caption_path, profile_version, caption_provider_version, cleanup_provider_version, cleanup_verifier_version, analysis_provider_version, crop_provider_version FROM dataset_items WHERE item_id = ?",
                    (item.item_id,),
                ).fetchone()
                source_changed = existing is not None and (
                    existing["source_hash"] != item.content_hash
                    or (
                        item.item_id not in preserve_mapping_ids
                        and (
                            existing["output_image_path"] != str(output_image)
                            or existing["caption_path"] != str(caption_path)
                        )
                    )
                )
                changed = source_changed
                if existing is None:
                    connection.execute("""
                        INSERT INTO dataset_items (
                            item_id, source_path, source_relative_path, source_hash, source_size,
                            source_mtime_ns, output_image_path, caption_path, profile_id,
                            profile_version, caption_provider_version, cleanup_provider_version,
                            cleanup_verifier_version,
                            analysis_provider_version, crop_provider_version,
                            naming_sequence, output_naming_mode, lora_name,
                            active, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """, (
                        item.item_id, str(item.path), item.relative_path, item.content_hash,
                        item.size, item.mtime_ns, str(output_image), str(caption_path),
                        profile["profile_id"], profile["profile_version"], caption_provider_version,
                        cleanup_provider_version, cleanup_verifier_version,
                        analysis_provider_version, crop_provider_version,
                        naming_sequence, output_naming_mode, lora_name,
                        now, now,
                    ))
                else:
                    should_reset = changed
                    preserve_completed_versions = not source_changed
                    stored_profile_version = (
                        existing["profile_version"]
                        if preserve_completed_versions
                        else profile["profile_version"]
                    )
                    stored_caption_version = (
                        existing["caption_provider_version"]
                        if preserve_completed_versions
                        else caption_provider_version
                    )
                    stored_cleanup_version = (
                        existing["cleanup_provider_version"]
                        if preserve_completed_versions
                        else cleanup_provider_version
                    )
                    stored_verifier_version = (
                        existing["cleanup_verifier_version"]
                        if preserve_completed_versions
                        else cleanup_verifier_version
                    )
                    stored_analysis_version = (
                        existing["analysis_provider_version"]
                        if preserve_completed_versions
                        else analysis_provider_version
                    )
                    stored_crop_version = (
                        existing["crop_provider_version"]
                        if preserve_completed_versions
                        else crop_provider_version
                    )
                    reset = """
                        status = 'pending', analysis_status = 'not_started',
                        analysis_json = NULL,
                        normalization_status = 'not_started',
                        watermark_status = 'not_requested',
                        cleanup_verification_status = 'not_requested',
                        cleanup_verification_json = NULL, review_status = 'not_requested',
                        crop_status = 'not_requested',
                        crop_json = NULL,
                        caption_status = 'not_started', validation_status = 'not_started',
                        error = NULL, started_at = NULL, completed_at = NULL,
                    """ if should_reset else ""
                    connection.execute(f"""
                        UPDATE dataset_items SET
                            {reset}
                            source_path = ?, source_relative_path = ?, source_hash = ?,
                            source_size = ?, source_mtime_ns = ?, output_image_path = ?,
                            caption_path = ?, profile_id = ?, profile_version = ?,
                            caption_provider_version = ?, cleanup_provider_version = ?,
                            cleanup_verifier_version = ?,
                            analysis_provider_version = ?, crop_provider_version = ?,
                            naming_sequence = COALESCE(?, naming_sequence),
                            output_naming_mode = ?, lora_name = ?,
                            active = 1, updated_at = ?
                        WHERE item_id = ?
                    """, (
                        str(item.path), item.relative_path, item.content_hash, item.size,
                        item.mtime_ns, str(output_image), str(caption_path), profile["profile_id"],
                        stored_profile_version, stored_caption_version,
                        stored_cleanup_version, stored_verifier_version,
                        stored_analysis_version,
                        stored_crop_version, naming_sequence,
                        output_naming_mode, lora_name, now, item.item_id,
                    ))
            self._set_metadata(connection, "profile", profile, now)
            self._set_metadata(connection, "caption_provider_version", caption_provider_version, now)
            self._set_metadata(connection, "cleanup_provider_version", cleanup_provider_version, now)
            self._set_metadata(connection, "cleanup_verifier_version", cleanup_verifier_version, now)
            self._set_metadata(connection, "analysis_provider_version", analysis_provider_version, now)
            self._set_metadata(connection, "crop_provider_version", crop_provider_version, now)
            self._set_metadata(connection, "output_naming_mode", output_naming_mode, now)
            self._set_metadata(connection, "lora_name", lora_name, now)

    def ensure_naming_sequences(self, source_items):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT item_id, naming_sequence FROM dataset_items"
            ).fetchall()
            existing = {row["item_id"]: row["naming_sequence"] for row in rows}
            maximum = max(
                (int(sequence) for sequence in existing.values() if sequence is not None),
                default=0,
            )
            sequences = {}
            updates = []
            for item in sorted(
                source_items,
                key=lambda source: (source.relative_path.casefold(), source.item_id),
            ):
                sequence = existing.get(item.item_id)
                if sequence is None:
                    maximum += 1
                    sequence = maximum
                    if item.item_id in existing:
                        updates.append((sequence, item.item_id))
                sequences[item.item_id] = int(sequence)
            connection.executemany(
                "UPDATE dataset_items SET naming_sequence = ? WHERE item_id = ?",
                updates,
            )
        return sequences

    def update_pending_versions(
        self,
        profile,
        caption_provider_version,
        cleanup_provider_version,
        cleanup_verifier_version,
        analysis_provider_version,
        crop_provider_version,
    ):
        now = utc_now()
        with self._connect() as connection:
            connection.execute("""
                UPDATE dataset_items SET profile_id = ?, profile_version = ?,
                    caption_provider_version = ?, cleanup_provider_version = ?,
                    cleanup_verifier_version = ?, analysis_provider_version = ?,
                    crop_provider_version = ?, updated_at = ?
                WHERE active = 1 AND status = 'pending'
            """, (
                profile["profile_id"], profile["profile_version"],
                caption_provider_version, cleanup_provider_version,
                cleanup_verifier_version, analysis_provider_version,
                crop_provider_version, now,
            ))

    def _set_metadata(self, connection, key, value, now=None):
        timestamp = now or utc_now()
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        connection.execute("""
            INSERT INTO project_metadata(key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """, (key, serialized, timestamp))

    def prepare_run(self, mode, force_rebuild_revision=0):
        if mode not in {"resume", "reprocess_failed", "force_rebuild"}:
            raise ValueError(f"Unknown run mode: {mode}")
        revision = int(force_rebuild_revision)
        if mode == "force_rebuild" and revision <= 0:
            raise ValueError(
                "force_rebuild requires a positive rebuild revision; increase it only when intentionally starting a new rebuild"
            )
        now = utc_now()
        force_rebuild_reset = 0
        with self._connect() as connection:
            connection.execute("""
                UPDATE dataset_items SET status = 'pending', error = 'Interrupted before completion',
                    started_at = NULL, updated_at = ?
                WHERE active = 1 AND status = 'processing'
            """, (now,))
            if mode == "reprocess_failed":
                connection.execute("""
                    UPDATE dataset_items SET status = 'pending', error = NULL, started_at = NULL,
                        completed_at = NULL, updated_at = ?
                    WHERE active = 1 AND status = 'failed'
                """, (now,))
            elif mode == "force_rebuild":
                row = connection.execute(
                    "SELECT value FROM project_metadata WHERE key = 'force_rebuild_revision'"
                ).fetchone()
                previous_revision = json.loads(row["value"]) if row is not None else None
                if previous_revision != revision:
                    cursor = connection.execute("""
                        UPDATE dataset_items SET status = 'pending', analysis_status = 'not_started',
                            analysis_json = NULL,
                            normalization_status = 'not_started',
                            watermark_status = 'not_requested',
                            cleanup_verification_status = 'not_requested',
                            cleanup_verification_json = NULL, review_status = 'not_requested',
                            crop_status = 'not_requested',
                            crop_json = NULL,
                            caption_status = 'not_started', validation_status = 'not_started',
                            error = NULL, started_at = NULL, completed_at = NULL, updated_at = ?
                        WHERE active = 1
                    """, (now,))
                    force_rebuild_reset = cursor.rowcount
                    self._set_metadata(connection, "force_rebuild_revision", revision, now)
        return force_rebuild_reset

    def reset_missing_outputs(self):
        now = utc_now()
        reset_ids = []
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT item_id, output_image_path, caption_path FROM dataset_items
                WHERE active = 1 AND status = 'complete'
            """).fetchall()
            for row in rows:
                if not Path(row["output_image_path"]).is_file() or not Path(row["caption_path"]).is_file():
                    reset_ids.append(row["item_id"])
            connection.executemany("""
                UPDATE dataset_items SET status = 'pending', validation_status = 'not_started',
                    error = 'Completed output was missing', completed_at = NULL, updated_at = ?
                WHERE item_id = ?
            """, [(now, item_id) for item_id in reset_ids])
        return len(reset_ids)

    def claim_next(self):
        now = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""
                SELECT * FROM dataset_items
                WHERE active = 1 AND status = 'pending'
                ORDER BY source_relative_path COLLATE NOCASE, item_id
                LIMIT 1
            """).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute("""
                UPDATE dataset_items SET status = 'processing', attempts = attempts + 1,
                    error = NULL, started_at = ?, updated_at = ? WHERE item_id = ?
            """, (now, now, row["item_id"]))
            connection.commit()
            return dict(row)

    def mark_complete(self, item_id, caption_status="written"):
        now = utc_now()
        with self._connect() as connection:
            connection.execute("""
                UPDATE dataset_items SET status = 'complete', normalization_status = 'converted_png',
                    caption_status = ?, validation_status = 'valid', error = NULL,
                    completed_at = ?, updated_at = ? WHERE item_id = ?
            """, (caption_status, now, now, item_id))

    def mark_failed(self, item_id, error):
        now = utc_now()
        with self._connect() as connection:
            connection.execute("""
                UPDATE dataset_items SET status = 'failed', validation_status = 'error',
                    error = ?, updated_at = ? WHERE item_id = ?
            """, (str(error), now, item_id))

    def mark_excluded(self, item_id, reason, review_status="cleanup_excluded"):
        now = utc_now()
        with self._connect() as connection:
            connection.execute("""
                UPDATE dataset_items SET status = 'excluded', validation_status = 'excluded',
                    review_status = ?, error = ?, completed_at = ?, updated_at = ?
                WHERE item_id = ?
            """, (str(review_status), str(reason), now, now, item_id))

    def reinstate_exact_duplicate(self, item_id):
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute("""
                UPDATE dataset_items SET status = 'pending',
                    normalization_status = 'not_started', analysis_status = 'not_started',
                    analysis_json = NULL, watermark_status = 'not_requested',
                    cleanup_verification_status = 'not_requested',
                    cleanup_verification_json = NULL, crop_status = 'not_requested',
                    crop_json = NULL, caption_status = 'not_started',
                    validation_status = 'not_started', review_status = 'not_requested',
                    error = NULL, started_at = NULL, completed_at = NULL, updated_at = ?
                WHERE item_id = ? AND status = 'excluded'
                    AND review_status = 'exact_duplicate_excluded'
            """, (now, item_id))
            return cursor.rowcount

    def mark_watermark_status(self, item_id, status):
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE dataset_items SET watermark_status = ?, updated_at = ? WHERE item_id = ?",
                (str(status), now, item_id),
            )

    def mark_cleanup_verification(self, item_id, status, result, review_status="not_requested"):
        now = utc_now()
        payload = json.dumps(result, ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE dataset_items SET cleanup_verification_status = ?,
                    cleanup_verification_json = ?, review_status = ?, updated_at = ?
                WHERE item_id = ?
                """,
                (str(status), payload, str(review_status), now, item_id),
            )

    def mark_analysis(self, item_id, status, analysis):
        now = utc_now()
        payload = json.dumps(analysis, ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                "UPDATE dataset_items SET analysis_status = ?, analysis_json = ?, updated_at = ? WHERE item_id = ?",
                (str(status), payload, now, item_id),
            )

    def mark_crop(self, item_id, status, crop_result):
        now = utc_now()
        payload = json.dumps(crop_result, ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                "UPDATE dataset_items SET crop_status = ?, crop_json = ?, updated_at = ? WHERE item_id = ?",
                (str(status), payload, now, item_id),
            )

    def records(self, active_only=True):
        where = "WHERE active = 1" if active_only else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM dataset_items {where} ORDER BY source_relative_path COLLATE NOCASE, item_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def summary(self):
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT status, COUNT(*) AS count FROM dataset_items
                WHERE active = 1 GROUP BY status
            """).fetchall()
        counts = {row["status"]: row["count"] for row in rows}
        total = sum(counts.values())
        return {
            "total": total,
            "eligible": total - counts.get("excluded", 0),
            "pending": counts.get("pending", 0),
            "processing": counts.get("processing", 0),
            "complete": counts.get("complete", 0),
            "failed": counts.get("failed", 0),
            "excluded": counts.get("excluded", 0),
            "inactive": self._inactive_count(),
        }

    def _inactive_count(self):
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM dataset_items WHERE active = 0").fetchone()
        return row["count"]
