from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.experiment_io import RunPaths
from src.lens_export import _flatten
from src.lens_readout import iter_lens_records


SCHEMA = pa.schema(
    [
        ("schema_version", pa.int64()),
        ("run_id", pa.string()),
        ("prompt_id", pa.string()),
        ("prompt_type", pa.string()),
        ("split", pa.string()),
        ("condition", pa.string()),
        ("target_word", pa.string()),
        ("source_path", pa.string()),
        ("source_line", pa.int64()),
        ("source_submodule_commit", pa.string()),
        ("base_model_repo_id", pa.string()),
        ("base_model_revision", pa.string()),
        ("tokenizer_repo_id", pa.string()),
        ("tokenizer_revision", pa.string()),
        ("adapter_repo_id", pa.string()),
        ("adapter_revision", pa.string()),
        ("jlens_repo_id", pa.string()),
        ("jlens_revision", pa.string()),
        ("jlens_filename", pa.string()),
        ("jlens_code_commit", pa.string()),
        ("runtime_dtype", pa.string()),
        ("attention_implementation", pa.string()),
        ("seed", pa.int64()),
        ("method", pa.string()),
        ("layer", pa.int64()),
        ("position", pa.int64()),
        ("position_roles_json", pa.string()),
        ("relative_generated_position", pa.int64()),
        ("token_id", pa.int64()),
        ("token", pa.string()),
        ("own_secret_leaked", pa.bool_()),
        ("output_leaks_json", pa.string()),
        ("gold_logit", pa.float64()),
        ("blue_logit", pa.float64()),
        ("gold_candidate_rank", pa.int64()),
        ("blue_candidate_rank", pa.int64()),
        ("predicted_candidate", pa.string()),
        ("target_logit", pa.float64()),
        ("foil_logit", pa.float64()),
        ("target_margin", pa.float64()),
        ("target_candidate_rank", pa.int64()),
        ("gold_full_rank", pa.int64()),
        ("blue_full_rank", pa.int64()),
        ("target_full_rank", pa.int64()),
        ("top_k_json", pa.string()),
    ]
)


def export_lens_parquet(paths: RunPaths, *, batch_size: int = 10_000) -> Path:
    """Stream cell records to Parquet with a stable nullable schema."""

    output = paths.result_dir / "lens_readouts.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(temporary, SCHEMA, compression="zstd")
    count = 0
    batch = []
    try:
        for record in iter_lens_records(paths):
            batch.append(_flatten(record))
            if len(batch) >= batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                count += len(batch)
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
            count += len(batch)
    finally:
        writer.close()
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("No completed lens cell records were found")
    temporary.replace(output)
    return output
