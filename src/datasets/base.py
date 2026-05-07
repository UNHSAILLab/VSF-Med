"""Abstract dataset loader interface for VSF-Med v2.

Each concrete loader reads its source dataset's native metadata format and
yields :class:`CaseRecord` instances. The base case builder then performs
stratified sampling and prompt generation to produce ``base_cases.csv``.
"""

from __future__ import annotations

import abc
import dataclasses
from pathlib import Path
from typing import Iterator, Optional


@dataclasses.dataclass(frozen=True)
class CaseRecord:
    """One source case before stratified sampling and prompt generation."""

    source_id: str
    image_path: Path                 # absolute or relative to dataset root
    modality: str                    # cxr | ct | mri | ophth | dermo | path | us | ...
    anatomy: str                     # e.g. "chest", "abdomen", "fundus"
    view_position: Optional[str]     # PA, AP, LATERAL, etc. (CXR-specific)
    labels: dict                     # label_name → 0|1|-1 ("uncertain" → -1)
    report_text: Optional[str]
    demographic_metadata: dict       # age_band, sex, race when permitted
    license_or_access_notes: str
    primary_pathology: Optional[str] = None    # pre-computed for stratification
    is_normal: Optional[bool] = None           # True if no finding present
    extra: dict = dataclasses.field(default_factory=dict)


class DatasetLoader(abc.ABC):
    """Yield :class:`CaseRecord` instances from a source dataset.

    Subclasses are CPU-only and must not require GPU or large-memory deps.
    They should be safe to instantiate even when source data is unavailable;
    the failure mode is to raise from ``iter_cases`` rather than at __init__.
    """

    name: str

    @abc.abstractmethod
    def iter_cases(self, root: Path) -> Iterator[CaseRecord]: ...

    def count(self, root: Path) -> int:
        """Optional fast-path count. Default walks ``iter_cases``."""
        return sum(1 for _ in self.iter_cases(root))
