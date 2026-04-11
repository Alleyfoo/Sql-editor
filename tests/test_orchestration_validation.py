from __future__ import annotations

from src.orchestration.runtime import CapabilityManifest, OrchestrationPlan, PlanStep, WorkerCapability
from src.orchestration.validation import validate_orchestration_plan


def _capabilities() -> CapabilityManifest:
    return CapabilityManifest(
        manifest_id="test",
        workers=[
            WorkerCapability(
                worker_id="mixed_executor_worker",
                allowed_task_classes=["pushdown"],
                accepted_handle_types=["source"],
                output_handle_type="result_table",
            )
        ],
    )


def test_validation_rejects_forbidden_raw_execution_fields() -> None:
    plan = OrchestrationPlan(
        task_class="pushdown",
        steps=[
            PlanStep(
                worker_id="mixed_executor_worker",
                input_handles=["hdl_000000000001"],
                params={"question": "ok", "raw_sql": "SELECT * FROM data"},
            )
        ],
    )
    errors = validate_orchestration_plan(plan, _capabilities())
    assert any("forbidden param key" in err for err in errors)


def test_validation_rejects_unexpected_param_keys() -> None:
    plan = OrchestrationPlan(
        task_class="pushdown",
        steps=[
            PlanStep(
                worker_id="mixed_executor_worker",
                input_handles=["hdl_000000000001"],
                params={"question": "ok", "extra": "bad"},
            )
        ],
    )
    errors = validate_orchestration_plan(plan, _capabilities())
    assert any("unexpected param key" in err for err in errors)
