"""Convert public gene states to features without losing UTR exon endpoints.

UTR_EI and UTR_IE are exon-side bases, just like their coding counterparts.
Unlike coding splice states, they do not identify the UTR side themselves.
Keep their identity until a connected UTR segment supplies a 5UTR/3UTR state.
This also permits a window ending inside a UTR intron to be resolved after
stitching, without consulting genomic order or guessing from CDS coordinates.
"""

import numpy as np


GENE_FEATURE_NAMES = [
    "intergenic", "intron", "CDS", "5UTR", "3UTR", "UTR_EI", "UTR_IE",
]
GENE_20_TO_FEATURE = np.array(
    [
        0,          # IR
        1, 1, 1,    # coding intron phase 0/1/2
        2, 2, 2,    # coding exon phase 0/1/2
        2,          # START
        2, 2, 2,    # coding exon-to-intron
        2, 2, 2,    # coding intron-to-exon
        2,          # STOP
        3,          # 5UTR
        4,          # 3UTR
        5,          # UTR_EI: exon side, resolved from UTR context
        1,          # UTR_INTRON: actual intron body
        6,          # UTR_IE: exon side, resolved from UTR context
    ],
    dtype=np.int64,
)
_UTR_SIDES = frozenset(("5UTR", "3UTR"))
_UTR_BOUNDARIES = frozenset(("UTR_EI", "UTR_IE"))
_UTR_CONTEXT = _UTR_SIDES | _UTR_BOUNDARIES | {"intron"}


def reduce_gene_states(encoded_labels):
    """Map 20 public states to features, retaining unresolved UTR markers."""
    labels = np.asarray(encoded_labels, dtype=np.int64)
    if not labels.size:
        return labels.copy()
    label_min, label_max = int(labels.min()), int(labels.max())
    if label_min < 0 or label_max >= len(GENE_20_TO_FEATURE):
        raise ValueError(
            f"Unsupported gene label range [{label_min}, {label_max}]; "
            f"expected 20-label ids in [0, {len(GENE_20_TO_FEATURE) - 1}]."
        )
    return GENE_20_TO_FEATURE[labels]


def resolve_utr_boundary_ranges(ranges, *, allow_unresolved=False):
    """Restore exon-side markers using explicitly labeled UTR context.

    Input/output ranges are in increasing genomic order on either strand. UTR
    identity comes from the 5UTR/3UTR states themselves, so reversing a minus
    strand path does not exchange the two biological sides. Context never
    crosses a coordinate gap, intergenic region, or CDS region.

    Within one window, ``allow_unresolved=True`` retains markers whose context
    lies in another window. Call again on each retained transcript after valid
    window stitching with the default strict policy. Missing or conflicting
    UTR context then raises ValueError rather than inventing an exon side.
    Neither the input ranges nor the decoded states are modified.
    """
    resolved = [list(region) for region in ranges]
    index = 0
    while index < len(resolved):
        if resolved[index][0] not in _UTR_CONTEXT:
            index += 1
            continue
        end = index + 1
        while (
            end < len(resolved)
            and resolved[end][0] in _UTR_CONTEXT
            and resolved[end - 1][2] + 1 == resolved[end][1]
        ):
            end += 1
        boundaries = [i for i in range(index, end) if resolved[i][0] in _UTR_BOUNDARIES]
        sides = {resolved[i][0] for i in range(index, end) if resolved[i][0] in _UTR_SIDES}
        interval = f"{resolved[index][1]}-{resolved[end - 1][2]}"
        # Markers at opposite ends of an intron may already have been resolved
        # inside separate windows. Still reject a side switch after stitching.
        has_splice_context = boundaries or any(resolved[i][0] == "intron" for i in range(index, end))
        if len(sides) > 1 and has_splice_context:
            raise ValueError(f"Conflicting 5UTR/3UTR context for UTR splice boundaries at {interval}.")
        if boundaries:
            if not sides and not allow_unresolved:
                raise ValueError(f"Missing 5UTR/3UTR context for UTR splice boundaries at {interval}.")
            if sides:
                side = next(iter(sides))
                for boundary in boundaries:
                    resolved[boundary][0] = side
        index = end

    merged = []
    for region in resolved:
        if merged and merged[-1][0] == region[0] and merged[-1][2] + 1 == region[1]:
            merged[-1][2] = region[2]
        else:
            merged.append(region)
    return merged


def gene_state_ranges(encoded_labels, offset=0):
    """Return feature runs, deferring UTR markers lacking window context.

    Labels must be one-dimensional and already in increasing genomic order;
    ``offset`` is the genomic coordinate of the first label. The caller must
    resolve any remaining UTR_EI/UTR_IE ranges after stitching before export.
    """
    features = reduce_gene_states(encoded_labels)
    if features.ndim != 1:
        raise ValueError(f"Expected a one-dimensional gene-state path, got {features.shape}.")
    if not features.size:
        return []
    changes = np.flatnonzero(np.diff(features))
    starts = np.insert(changes + 1, 0, 0)
    ends = np.append(changes, features.size - 1)
    ranges = [
        [GENE_FEATURE_NAMES[int(features[start])], int(start) + offset, int(end) + offset]
        for start, end in zip(starts, ends)
    ]
    return resolve_utr_boundary_ranges(ranges, allow_unresolved=True)
