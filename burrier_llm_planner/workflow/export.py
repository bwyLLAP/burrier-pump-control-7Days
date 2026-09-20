from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from burrier_llm_planner.domain.enums import WorkflowStatus
from burrier_llm_planner.domain.models import PlanSummary, PlanningSession


class ExportGuardError(RuntimeError):
    """Raised when a schedule does not satisfy every export gate."""


@dataclass(frozen=True)
class ExportResult:
    csv_path: Path
    xlsx_path: Path
    csv_sha256: str
    xlsx_sha256: str


def export_approved_schedule(
    *,
    session: PlanningSession,
    result: PlanSummary,
    schedule: pd.DataFrame,
    output_dir: Path,
) -> ExportResult:
    if not session.operational_export_allowed:
        raise ExportGuardError("Demonstration mode cannot export an operational schedule.")
    if session.status is not WorkflowStatus.APPROVED:
        raise ExportGuardError("The workflow is not in the approved state.")
    decision = session.final_decision
    if decision is None or decision.decision != "approve":
        raise ExportGuardError("Explicit operator approval is missing.")
    if session.selected_result_id != result.result_id or decision.result_id != result.result_id:
        raise ExportGuardError("The selected result does not match the approval decision.")
    if not result.feasible:
        raise ExportGuardError("An infeasible result cannot be exported.")
    if not session.input_fingerprint or result.input_fingerprint != session.input_fingerprint:
        raise ExportGuardError("The approved result is stale because inputs changed.")
    if schedule.empty:
        raise ExportGuardError("The approved schedule contains no intervals.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    safe_session = "".join(character for character in session.session_id if character.isalnum())
    safe_result = "".join(character for character in result.result_id if character.isalnum())
    stem = f"approved_schedule_{safe_session}_{safe_result}"
    csv_path = destination / f"{stem}.csv"
    xlsx_path = destination / f"{stem}.xlsx"

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".csv", delete=False, dir=destination
    ) as temporary_csv:
        schedule.to_csv(temporary_csv, index=False)
        csv_temp_path = Path(temporary_csv.name)
    csv_temp_path.replace(csv_path)

    with tempfile.NamedTemporaryFile(
        suffix=".xlsx", delete=False, dir=destination
    ) as temporary_xlsx:
        xlsx_temp_path = Path(temporary_xlsx.name)
    metadata = pd.DataFrame(
        {
            "Field": [
                "Session ID",
                "Result ID",
                "Approved by",
                "Approval time",
                "Input fingerprint",
                "Notice",
            ],
            "Value": [
                session.session_id,
                result.result_id,
                decision.actor,
                decision.timestamp.isoformat(),
                result.input_fingerprint,
                "Operator-approved recommendation; not a direct PLC command.",
            ],
        }
    )
    with pd.ExcelWriter(xlsx_temp_path, engine="openpyxl") as writer:
        schedule.to_excel(writer, sheet_name="Approved Schedule", index=False)
        metadata.to_excel(writer, sheet_name="Approval Metadata", index=False)
    xlsx_temp_path.replace(xlsx_path)

    return ExportResult(
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        csv_sha256=_sha256(csv_path),
        xlsx_sha256=_sha256(xlsx_path),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
