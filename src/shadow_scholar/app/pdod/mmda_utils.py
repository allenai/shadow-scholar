import itertools
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence

from shadow_scholar.cli import safe_import

with safe_import():
    from mmda.types.annotation import BoxGroup, SpanGroup  # pyright: ignore
    from mmda.types.box import Box  # pyright: ignore
    from mmda.types.document import Document  # pyright: ignore
    from mmda.types.names import Tokens  # pyright: ignore


class BoxKey(NamedTuple):
    top: float
    page: int


def box_group_from_span_group(
    span_group: "SpanGroup",
    doc: Optional["Document"] = None,
    merge_if_on_same_row: bool = True,
    position_rounding: int = 2,
) -> "BoxGroup":

    doc = span_group.doc or doc
    if doc is None:
        raise ValueError("span_group must have a doc, or doc must be provided")

    boxes_it: Iterable[Box] = (
        span.box  # pyright: ignore
        for token in doc.find_overlapping(span_group, Tokens)
        for span in token.spans
    )

    if merge_if_on_same_row:
        boxes_it = merge_box_groups_on_same_row(
            boxes=boxes_it, position_rounding=position_rounding
        )

    return BoxGroup(boxes=list(boxes_it), type=span_group.type)


def merge_box_groups_on_same_row(
    box_groups: Optional[Iterable["BoxGroup"]] = None,
    boxes: Optional[Iterable["Box"]] = None,
    position_rounding: int = 2,
) -> Sequence["Box"]:

    if box_groups is not None and boxes is not None:
        raise ValueError("either box_groups or boxes must be provided")

    to_merge: Dict[BoxKey, List[Box]] = {}

    # combine both boxes for box_groups (if provided) as well as `boxes`
    # (if provided) into a single iterable of boxes.
    boxes_it = itertools.chain(
        (box for box_group in (box_groups or []) for box in box_group.boxes),
        (boxes or []),
    )

    # to decide which box to merge, we key them of their
    for box in boxes_it:
        to_merge.setdefault(
            BoxKey(top=round(box.t, position_rounding), page=box.page), []
        ).append(box)

    merged_boxes: List[Box] = []
    for box_key, boxes in to_merge.items():
        merged_boxes.append(
            Box(
                l=(left := min(box.l for box in boxes)),
                t=(top := min(box.t for box in boxes)),
                w=(max(box.l + box.w for box in boxes) - left),
                h=(max(box.t + box.h for box in boxes) - top),
                page=box_key.page,
            )
        )

    return merged_boxes
