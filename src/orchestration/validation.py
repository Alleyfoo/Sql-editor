from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .runtime import CapabilityManifest, OrchestrationPlan

FORBIDDEN_PARAM_KEYS = {
    "sql",
    "raw_sql",
    "python",
    "raw_python",
    "code",
    "script",
    "raw_data",
    "rows",
}

ALLOWED_TASK_CLASSES = {
    "pushdown",
    "hybrid",
    "python_first",
    "cleaning_first",
    "follow_up",
    "adversarial",
}


def validate_orchestration_plan(plan: "OrchestrationPlan", capability_manifest: "CapabilityManifest") -> List[str]:
    errors: List[str] = []
    if plan.task_class not in ALLOWED_TASK_CLASSES:
        errors.append(f"unsupported task_class: {plan.task_class!r}")

    workers = capability_manifest.by_id()
    if not plan.steps:
        errors.append("plan.steps must not be empty")

    for idx, step in enumerate(plan.steps):
        if step.worker_id not in workers:
            errors.append(f"steps[{idx}] unknown worker_id: {step.worker_id!r}")
            continue
        capability = workers[step.worker_id]
        if plan.task_class not in capability.allowed_task_classes:
            errors.append(
                f"steps[{idx}] worker {step.worker_id!r} not allowed for task_class {plan.task_class!r}"
            )

        for hid in step.input_handles:
            if not isinstance(hid, str) or not hid:
                errors.append(f"steps[{idx}] contains invalid input handle")

        for key in step.params:
            low = str(key).strip().lower()
            if low in FORBIDDEN_PARAM_KEYS:
                errors.append(f"steps[{idx}] forbidden param key: {key!r}")

        # Keep orchestration schema narrow: only coordinator metadata fields are allowed.
        allowed = {
            "question",
            "expected_route_family",
            "validator",
            "reason",
        }
        for key in step.params:
            if key not in allowed:
                errors.append(f"steps[{idx}] unexpected param key: {key!r}")

    return errors
