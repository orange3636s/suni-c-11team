from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from src.ml.inference import (
    HYBRID_BUNDLE_FILES,
    MODEL_DELETE_STAGING_DIR,
    MODEL_ID_PATTERN,
    _MODEL_DELETE_LOCK,
)
from src.runtime.operation_coordinator import (
    ACTIVE_JOB_MESSAGE,
    ActiveOperationError,
    OperationCoordinator,
    operation_coordinator,
)
from src.runtime.store import RuntimeStore


RESET_STAGING_DIR = ".history-reset"
_PREDICTION_ARTIFACT = re.compile(
    r"^prediction_[A-Za-z0-9_-]+\.json\.gz$"
)
_ANALYSIS_ARTIFACT = re.compile(
    r"^analysis_[A-Za-z0-9_-]+\.json\.gz$"
)


class HistoryResetError(RuntimeError):
    """A reset failed without being reported as successful."""


class UnsafeResetPathError(HistoryResetError):
    """A candidate escaped or violated the reset allowlist."""


@dataclass(frozen=True)
class ResetTarget:
    root: Path
    source: Path
    staged_relative: Path
    allowed_children: frozenset[str] | None = None

    @property
    def is_directory(self) -> bool:
        return self.allowed_children is not None


@dataclass(frozen=True)
class HistoryResetPlan:
    targets: tuple[ResetTarget, ...]
    model_count: int
    model_artifact_count: int
    prediction_artifact_count: int
    analysis_artifact_count: int


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    junction_check = getattr(path, "is_junction", None)
    if callable(junction_check):
        try:
            if junction_check():
                return True
        except OSError:
            return True
    if path.exists():
        try:
            return not _same_path(path.resolve(strict=True), path.absolute())
        except OSError:
            return True
    return False


def _valid_model_id(value: str) -> bool:
    return bool(
        value
        and MODEL_ID_PATTERN.fullmatch(value)
        and ".." not in value
        and "/" not in value
        and "\\" not in value
    )


def _validated_root(path: Path) -> Path:
    unresolved = Path(path).absolute()
    if unresolved.exists() and _is_link_or_junction(unresolved):
        raise UnsafeResetPathError("초기화 허용 경로가 안전하지 않습니다.")
    resolved = unresolved.resolve()
    if resolved == resolved.parent:
        raise UnsafeResetPathError("볼륨 루트는 초기화 허용 경로가 될 수 없습니다.")
    if resolved.exists() and not resolved.is_dir():
        raise UnsafeResetPathError("초기화 허용 경로가 디렉터리가 아닙니다.")
    return resolved


def _require_inside(candidate: Path, root: Path) -> Path:
    resolved = candidate.resolve(strict=True)
    if root not in resolved.parents or resolved == root:
        raise UnsafeResetPathError("초기화 대상이 허용 경로를 벗어났습니다.")
    if not _same_path(resolved, candidate.absolute()):
        raise UnsafeResetPathError(
            "심볼릭 링크 또는 Junction은 초기화할 수 없습니다."
        )
    return resolved


class HistoryResetService:
    def __init__(
        self,
        *,
        model_dir: str | Path,
        store: RuntimeStore,
        coordinator: OperationCoordinator = operation_coordinator,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.store = store
        self.coordinator = coordinator

    def summary(self) -> dict[str, int]:
        plan = self._build_file_plan()
        database = self.store.history_reset_counts()
        return {
            "model_count": plan.model_count,
            "prediction_history_count": database[
                "prediction_history_count"
            ],
            "analysis_history_count": database["analysis_history_count"],
            "model_artifact_count": plan.model_artifact_count,
            "prediction_artifact_count": plan.prediction_artifact_count,
            "analysis_artifact_count": plan.analysis_artifact_count,
            "report_snapshot_count": database["report_snapshot_count"],
        }

    def reset(self) -> dict[str, Any]:
        moved: list[tuple[ResetTarget, Path]] = []
        token = uuid4().hex
        with self.coordinator.exclusive_reset(), _MODEL_DELETE_LOCK:
            plan = self._build_file_plan()
            try:
                with self.store.history_reset_transaction() as connection:
                    running = self.store.running_history_counts(connection)
                    if any(running.values()):
                        raise ActiveOperationError(ACTIVE_JOB_MESSAGE)
                    database_counts = self.store.history_reset_counts(connection)
                    self._stage_targets(plan.targets, token, moved)
                    deleted_rows = self.store.delete_reset_history_rows(
                        connection
                    )
                    if deleted_rows != database_counts:
                        raise HistoryResetError(
                            "초기화 대상 DB 개수가 트랜잭션 중 변경되었습니다."
                        )
            except Exception as exc:
                try:
                    self._restore_targets(moved)
                except Exception as recovery_exc:
                    raise HistoryResetError(
                        "초기화 실패 후 격리 파일을 모두 복구하지 못했습니다."
                    ) from recovery_exc
                raise exc

            try:
                self._purge_targets(moved)
            except Exception as exc:
                raise HistoryResetError(
                    "DB 초기화 후 격리 파일 정리에 실패했습니다."
                ) from exc

        return {
            "success": True,
            "deleted": {
                "models": plan.model_count,
                "model_files": plan.model_artifact_count,
                "prediction_histories": database_counts[
                    "prediction_history_count"
                ],
                "prediction_artifacts": plan.prediction_artifact_count,
                "analysis_histories": database_counts[
                    "analysis_history_count"
                ],
                "analysis_artifacts": plan.analysis_artifact_count,
                "report_snapshots": database_counts[
                    "report_snapshot_count"
                ],
            },
            "preserved": {
                "alert_logs": True,
                "automation_runs": True,
                "source_csv": True,
            },
        }

    def _build_file_plan(self) -> HistoryResetPlan:
        model_root = _validated_root(self.model_dir)
        artifact_root = _validated_root(self.store.artifact_root)
        for root in {model_root, artifact_root}:
            self._validate_previous_staging(root)
        model_targets, model_ids, model_files = self._model_targets(
            model_root
        )
        prediction_targets = self._runtime_targets(
            artifact_root,
            "predictions",
            _PREDICTION_ARTIFACT,
        )
        analysis_targets = self._runtime_targets(
            artifact_root,
            "analyses",
            _ANALYSIS_ARTIFACT,
        )
        targets = tuple(
            sorted(
                [*model_targets, *prediction_targets, *analysis_targets],
                key=lambda item: (
                    str(item.root),
                    len(item.source.parts),
                    str(item.source),
                ),
            )
        )
        return HistoryResetPlan(
            targets=targets,
            model_count=len(model_ids),
            model_artifact_count=model_files,
            prediction_artifact_count=len(prediction_targets),
            analysis_artifact_count=len(analysis_targets),
        )

    @staticmethod
    def _validate_previous_staging(root: Path) -> None:
        container = root / RESET_STAGING_DIR
        if not container.exists() and not container.is_symlink():
            return
        if (
            not container.is_dir()
            or _is_link_or_junction(container)
            or container.resolve().parent != root
        ):
            raise UnsafeResetPathError(
                "초기화 격리 디렉터리가 안전하지 않습니다."
            )
        if any(container.iterdir()):
            raise HistoryResetError(
                "이전 초기화에서 남은 격리 데이터를 먼저 복구해야 합니다."
            )

    def _model_targets(
        self,
        root: Path,
    ) -> tuple[list[ResetTarget], set[str], int]:
        if not root.exists():
            return [], set(), 0
        targets: list[ResetTarget] = []
        model_ids: set[str] = set()
        artifact_count = 0
        for entry in root.iterdir():
            if entry.name in {RESET_STAGING_DIR, MODEL_DELETE_STAGING_DIR}:
                continue
            if entry.is_symlink():
                if (
                    entry.name.endswith((".joblib", ".json"))
                    or _valid_model_id(entry.name)
                ):
                    raise UnsafeResetPathError(
                        "심볼릭 링크인 모델 Artifact는 초기화할 수 없습니다."
                    )
                continue
            if entry.is_file():
                suffix = next(
                    (
                        extension
                        for extension in (".joblib", ".json")
                        if entry.name.endswith(extension)
                    ),
                    None,
                )
                if suffix is None:
                    continue
                model_id = entry.name[: -len(suffix)]
                if not _valid_model_id(model_id):
                    continue
                self._validate_regular_file(entry, root)
                targets.append(
                    ResetTarget(
                        root=root,
                        source=entry,
                        staged_relative=(
                            Path("models") / "active" / entry.name
                        ),
                    )
                )
                model_ids.add(model_id)
                artifact_count += 1
                continue
            if not entry.is_dir() or not _valid_model_id(entry.name):
                continue
            children = self._validated_model_directory(entry, root)
            if children is None:
                continue
            targets.append(
                ResetTarget(
                    root=root,
                    source=entry,
                    staged_relative=(
                        Path("models") / "active" / entry.name
                    ),
                    allowed_children=frozenset(children),
                )
            )
            model_ids.add(entry.name)
            artifact_count += len(children)

        deleting_root = root / MODEL_DELETE_STAGING_DIR
        if deleting_root.exists() or deleting_root.is_symlink():
            self._validate_directory(deleting_root, root)
            for entry in deleting_root.iterdir():
                if not _valid_model_id(entry.name):
                    raise UnsafeResetPathError(
                        "기존 모델 삭제 격리 경로에 허용되지 않은 항목이 있습니다."
                    )
                children = self._validated_staged_model_directory(
                    entry,
                    deleting_root,
                )
                if not children:
                    continue
                targets.append(
                    ResetTarget(
                        root=root,
                        source=entry,
                        staged_relative=(
                            Path("models") / "deleting" / entry.name
                        ),
                        allowed_children=frozenset(children),
                    )
                )
                model_ids.add(entry.name)
                artifact_count += len(children)
        return targets, model_ids, artifact_count

    def _validated_model_directory(
        self,
        directory: Path,
        root: Path,
    ) -> set[str] | None:
        self._validate_directory(directory, root)
        entries = list(directory.iterdir())
        known = {entry.name for entry in entries} & set(HYBRID_BUNDLE_FILES)
        if not known:
            return None
        unsafe = [
            entry.name
            for entry in entries
            if entry.name not in HYBRID_BUNDLE_FILES
            or entry.is_symlink()
            or not entry.is_file()
        ]
        if unsafe:
            raise UnsafeResetPathError(
                "모델 Bundle에 허용되지 않은 파일이 있습니다."
            )
        for entry in entries:
            self._validate_regular_file(entry, directory)
        return {entry.name for entry in entries}

    def _validated_staged_model_directory(
        self,
        directory: Path,
        deleting_root: Path,
    ) -> set[str]:
        self._validate_directory(directory, deleting_root)
        entries = list(directory.iterdir())
        if not entries:
            return set()
        flat_names = {
            f"{directory.name}.joblib",
            f"{directory.name}.json",
        }
        names = {entry.name for entry in entries}
        if not (
            names.issubset(flat_names)
            or names.issubset(set(HYBRID_BUNDLE_FILES))
        ):
            raise UnsafeResetPathError(
                "기존 모델 삭제 격리 경로에 허용되지 않은 파일이 있습니다."
            )
        for entry in entries:
            self._validate_regular_file(entry, directory)
        return names

    def _runtime_targets(
        self,
        root: Path,
        kind: str,
        pattern: re.Pattern[str],
    ) -> list[ResetTarget]:
        directory = root / kind
        if not directory.exists() and not directory.is_symlink():
            return []
        self._validate_directory(directory, root)
        targets: list[ResetTarget] = []
        for entry in directory.iterdir():
            if not pattern.fullmatch(entry.name):
                continue
            self._validate_regular_file(entry, directory)
            targets.append(
                ResetTarget(
                    root=root,
                    source=entry,
                    staged_relative=(
                        Path("runtime") / kind / entry.name
                    ),
                )
            )
        return targets

    @staticmethod
    def _validate_directory(directory: Path, parent: Path) -> None:
        if (
            not directory.exists()
            or not directory.is_dir()
            or _is_link_or_junction(directory)
        ):
            raise UnsafeResetPathError(
                "초기화 대상 디렉터리가 안전하지 않습니다."
            )
        resolved = _require_inside(directory, parent)
        if resolved.parent != parent:
            raise UnsafeResetPathError(
                "초기화 대상 디렉터리 위치가 허용되지 않았습니다."
            )

    @staticmethod
    def _validate_regular_file(path: Path, parent: Path) -> None:
        if (
            not path.exists()
            or not path.is_file()
            or _is_link_or_junction(path)
        ):
            raise UnsafeResetPathError(
                "초기화 대상 파일이 안전하지 않습니다."
            )
        resolved = _require_inside(path, parent)
        if resolved.parent != parent:
            raise UnsafeResetPathError(
                "초기화 대상 파일 위치가 허용되지 않았습니다."
            )

    def _validate_target(self, target: ResetTarget) -> None:
        if target.is_directory:
            self._validate_directory(target.source, target.source.parent)
            entries = list(target.source.iterdir())
            names = {entry.name for entry in entries}
            if names != set(target.allowed_children or ()):
                raise HistoryResetError(
                    "초기화 계획 생성 후 모델 Bundle이 변경되었습니다."
                )
            for entry in entries:
                self._validate_regular_file(entry, target.source)
        else:
            self._validate_regular_file(target.source, target.source.parent)
        _require_inside(target.source, target.root)

    def _stage_targets(
        self,
        targets: Iterable[ResetTarget],
        token: str,
        moved: list[tuple[ResetTarget, Path]],
    ) -> None:
        stage_roots: dict[Path, Path] = {}
        try:
            for target in targets:
                self._validate_target(target)
                stage_root = stage_roots.get(target.root)
                if stage_root is None:
                    stage_root = self._prepare_stage_root(target.root, token)
                    stage_roots[target.root] = stage_root
                destination = stage_root / target.staged_relative
                if destination.exists() or destination.is_symlink():
                    raise HistoryResetError("초기화 격리 경로가 중복되었습니다.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                _require_inside(destination.parent, stage_root)
                target.source.replace(destination)
                moved.append((target, destination))
        except Exception:
            for root, stage_root in stage_roots.items():
                self._cleanup_empty_parents(stage_root, root)
            raise

    @staticmethod
    def _prepare_stage_root(root: Path, token: str) -> Path:
        container = root / RESET_STAGING_DIR
        if container.exists() or container.is_symlink():
            if (
                not container.is_dir()
                or _is_link_or_junction(container)
                or container.resolve().parent != root
            ):
                raise UnsafeResetPathError(
                    "초기화 격리 디렉터리가 안전하지 않습니다."
                )
        else:
            container.mkdir()
        stage_root = container / token
        if stage_root.exists() or stage_root.is_symlink():
            raise HistoryResetError("초기화 격리 작업 ID가 중복되었습니다.")
        stage_root.mkdir()
        if stage_root.resolve().parent != container.resolve():
            raise UnsafeResetPathError(
                "초기화 격리 작업 경로가 안전하지 않습니다."
            )
        return stage_root

    def _restore_targets(
        self,
        moved: list[tuple[ResetTarget, Path]],
    ) -> None:
        failures: list[Exception] = []
        for target, staged in reversed(moved):
            try:
                if target.source.exists() or target.source.is_symlink():
                    raise HistoryResetError(
                        "원래 위치에 다른 파일이 생성되어 복구할 수 없습니다."
                    )
                staged.replace(target.source)
                self._cleanup_empty_parents(staged.parent, target.root)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise HistoryResetError(
                "격리된 초기화 대상을 모두 복구하지 못했습니다."
            ) from failures[0]

    def _purge_targets(
        self,
        moved: list[tuple[ResetTarget, Path]],
    ) -> None:
        for target, staged in reversed(moved):
            _require_inside(staged, target.root)
            if target.is_directory:
                if _is_link_or_junction(staged) or not staged.is_dir():
                    raise UnsafeResetPathError(
                        "격리된 모델 디렉터리가 안전하지 않습니다."
                    )
                entries = list(staged.iterdir())
                if {entry.name for entry in entries} != set(
                    target.allowed_children or ()
                ):
                    raise HistoryResetError(
                        "격리 후 모델 Bundle 내용이 변경되었습니다."
                    )
                for entry in entries:
                    self._validate_regular_file(entry, staged)
                    entry.unlink()
                staged.rmdir()
            else:
                self._validate_regular_file(staged, staged.parent)
                staged.unlink()
            self._cleanup_empty_parents(staged.parent, target.root)

    @staticmethod
    def _cleanup_empty_parents(start: Path, stop: Path) -> None:
        current = start
        while current != stop and stop in current.parents:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent
