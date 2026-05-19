BASE_GENE_PRED_STATE_NAMES = [
    "IR",
    "intron0",
    "intron1",
    "intron2",
    "Exon0",
    "Exon1",
    "Exon2",
    "START",
    "EI0",
    "EI1",
    "EI2",
    "IE0",
    "IE1",
    "IE2",
    "STOP",
    "5UTR",
    "3UTR",
    "UTR_EI",
    "UTR_INTRON",
    "UTR_IE",
]

BASE_GENE_PRED_STATE_TO_INDEX = {
    name: index for index, name in enumerate(BASE_GENE_PRED_STATE_NAMES)
}

GENE_PRED_BASE_LABEL_DIM = len(BASE_GENE_PRED_STATE_NAMES)
CORE_GENE_PRED_STATE_IDS = tuple(range(15))
UTR_GENE_PRED_STATE_IDS = tuple(range(15, 20))

# States 0-14 keep the constrained coding-gene topology. UTR introns are
# optional, but when used they keep the local UTR_EI -> UTR_INTRON -> UTR_IE
# ordering.
BASE_GENE_PRED_TRANSITIONS = {
    0: [0, 7, 15],
    1: [1, 11],
    2: [2, 12],
    3: [3, 13],
    4: [5, 8],
    5: [6, 9, 14],
    6: [4, 10],
    7: [5],
    8: [1],
    9: [2],
    10: [3],
    11: [4],
    12: [5],
    13: [6],
    14: [0, 16],
    15: [7, 15, 17],
    16: [0, 16, 17],
    17: [18],
    18: [18, 19],
    19: [15, 16],
}

def expanded_num_states(num_copies=1):
    return 1 + (GENE_PRED_BASE_LABEL_DIM - 1) * num_copies


def expanded_state_index(base_state, copy_index=0):
    if base_state == 0:
        return 0
    return 1 + copy_index * (GENE_PRED_BASE_LABEL_DIM - 1) + (base_state - 1)


def expanded_state_names(num_copies=1):
    if num_copies == 1:
        return list(BASE_GENE_PRED_STATE_NAMES)

    names = [BASE_GENE_PRED_STATE_NAMES[0]]
    for copy_index in range(num_copies):
        suffix = f"#{copy_index}"
        names.extend(f"{name}{suffix}" for name in BASE_GENE_PRED_STATE_NAMES[1:])
    return names


def expanded_transition_edges(num_copies=1):
    edges = [(0, 0)]
    for copy_index in range(num_copies):
        for src_state, next_states in BASE_GENE_PRED_TRANSITIONS.items():
            if src_state == 0:
                for dst_state in next_states:
                    if dst_state == 0:
                        continue
                    edges.append((0, expanded_state_index(dst_state, copy_index)))
                continue

            expanded_src = expanded_state_index(src_state, copy_index)
            for dst_state in next_states:
                if dst_state == 0:
                    edges.append((expanded_src, 0))
                else:
                    edges.append((expanded_src, expanded_state_index(dst_state, copy_index)))
    return edges
