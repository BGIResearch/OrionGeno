"""Join feature ranges only across valid, adjacent HMM state transitions."""
from ..hmm_layer.gene_pred_hmm_topology import BASE_GENE_PRED_TRANSITIONS, BASE_GENE_PRED_STATE_NAMES
import numpy as np


def has_complete_coding_path(regions, sequence, labels, coords, strand):
    """Recognize complete CDS at block ends without requiring flanking IR.

    Check original HMM labels, not inferred codon features or literal ATG. This
    preserves the caller's non-ATG policy and never joins across a rejected edge.
    """
    coding = [region for region in regions if region[0] == "CDS"]
    if not coding:
        return False
    start, end = coding[0][1], coding[-1][2]
    pieces = []
    next_position = start
    for y, coord in zip(labels, coords):
        if coord[0] != sequence:
            continue
        left, right = max(start, int(coord[2])), min(end, int(coord[3]))
        if right < left:
            continue
        if left != next_position:
            return False
        pieces.append(np.asarray(y)[left-int(coord[2]):right-int(coord[2])+1])
        next_position = right + 1
    if next_position != end + 1 or not pieces:
        return False
    path = np.concatenate(pieces)
    if strand == "-":
        path = path[::-1]
    if path[0] != 7 or path[-1] != 14 or np.count_nonzero(path == 7) != 1 or np.count_nonzero(path == 14) != 1:
        return False
    return all(int(target) in BASE_GENE_PRED_TRANSITIONS.get(int(source), ())
               for source, target in zip(path[:-1], path[1:]))


def boundary_transition(left_state, right_state, strand):
    """Arguments are in genomic order; the topology follows transcription."""
    source, target = int(left_state), int(right_state)
    if strand == "-":
        source, target = target, source
    return target in BASE_GENE_PRED_TRANSITIONS.get(source, ())


def _conflicting_utr_context(left, right):
    """Check UTR identity across a join, including intervening whole windows.

    Public UTR intron states do not retain the decoder's 5'/3' memory.
    Two locally legal paths can therefore disagree despite an 18 -> 18 join.
    Feature labels retain the biological side on either strand. Only inspect
    the contiguous context touching this join; CDS, IR and gaps are barriers.
    """
    if not left or not right or left[-1][2] + 1 != right[0][1]:
        return False
    context = {"5UTR", "3UTR", "UTR_EI", "UTR_IE", "intron"}
    if left[-1][0] not in context or right[0][0] not in context:
        return False
    sides = set()
    has_splice = False
    for regions, expected, direction in (
        (reversed(left), left[-1][2], -1),
        (iter(right), right[0][1], 1),
    ):
        for kind, start, end in regions:
            if kind not in context or (end if direction == -1 else start) != expected:
                break
            if kind in {"5UTR", "3UTR"}:
                sides.add(kind)
            else:
                has_splice = True
            expected = start - 1 if direction == -1 else end + 1
    return has_splice and len(sides) > 1


def stitch_window_ranges(labels, coords, strand, get_ranges):
    """Return continuous blocks plus raw state/phase evidence at every join.

    Labels and coordinates must already be in increasing genomic order. Invalid
    joins delimit blocks: callers must not join the partial transcripts there.
    UTR identity must agree across a splice even if the public states can join.
    """
    blocks, checks = [], []
    current = []
    previous = None
    for y, coord in zip(labels, coords):
        length = max(0, int(coord[3]) - int(coord[2]) + 1)
        if not length:
            continue
        if length > len(y):
            raise ValueError("Window labels are shorter than their genomic interval")
        y = y[:length]
        window_ranges = get_ranges(y, int(coord[2]))
        if previous is not None:
            prev_coord, prev_state = previous
            adjacent = coord[0] == prev_coord[0] and int(coord[2]) == int(prev_coord[3]) + 1
            valid = adjacent and boundary_transition(prev_state, y[0], strand)
            status = "joined" if valid else "invalid_transition" if adjacent else "coordinate_gap"
            if valid and _conflicting_utr_context(current, window_ranges):
                valid = False
                status = "utr_context_conflict"
            if coord[0] == prev_coord[0]:
                checks.append({"sequence": coord[0], "strand": strand,
                               "left_end": int(prev_coord[3]), "right_start": int(coord[2]),
                               "left_state": int(prev_state), "right_state": int(y[0]),
                               "left_state_name": BASE_GENE_PRED_STATE_NAMES[int(prev_state)],
                               "right_state_name": BASE_GENE_PRED_STATE_NAMES[int(y[0])],
                               "status": status})
            if not valid:
                blocks.append((prev_coord[0], current))
                current = []
        for region in window_ranges:
            if current and current[-1][0] == region[0] and current[-1][2] + 1 == region[1]:
                current[-1][2] = region[2]
            else:
                current.append(list(region))
        previous = (coord, int(y[-1]))
    if previous is not None:
        blocks.append((previous[0][0], current))
    return blocks, checks
