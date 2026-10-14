"""Targeted, auditable joint decoding of conflicting cross-window UTRs."""
import hashlib
import os
import time

import numpy as np

from .window_stitching import boundary_transition


def state_runs(path, offset):
    changes = np.r_[0, np.flatnonzero(np.diff(path)) + 1, len(path)]
    return [[int(path[a]), int(offset + a), int(offset + b - 1)]
            for a, b in zip(changes[:-1], changes[1:])]


def repair_conflicts(labels, coords, chunks, strand, checks, callback):
    """Labels are genomic ordered; minus-strand nucleotide rows remain reversed.

    Modify only a completely bounded UTR component. Fixed flanking labels
    anchor both splice connections; probability context may extend further.
    Failed proposals never mutate the input. Adjacent CDS labels are frozen.
    """
    audits = []
    for check in checks:
        if check['status'] != 'utr_context_conflict':
            continue
        audit = dict(check, status='fallback_split', method='joint_constrained_hmm')
        started = time.perf_counter()
        try:
            index = next(i for i, c in enumerate(coords)
                         if c[0] == check['sequence'] and int(c[3]) == check['left_end'])
            first = last = index
            while first and coords[first-1][0] == coords[first][0] and int(coords[first-1][3])+1 == int(coords[first][2]):
                first -= 1
            while last+1 < len(coords) and coords[last+1][0] == coords[last][0] and int(coords[last][3])+1 == int(coords[last+1][2]):
                last += 1
            origin = int(coords[first][2])
            lengths = [int(c[3])-int(c[2])+1 for c in coords[first:last+1]]
            offsets = np.r_[0, np.cumsum(lengths)]
            path = np.concatenate([np.asarray(labels[i])[:lengths[i-first]] for i in range(first, last+1)])
            seam = check['left_end'] - origin + 1
            left, right = seam-1, seam
            while left >= 0 and 15 <= path[left] <= 19:
                left -= 1
            while right < len(path) and 15 <= path[right] <= 19:
                right += 1
            # Half-open editable interval covers complete UTR exons and introns.
            left += 1
            if left == 0 or right == len(path):
                raise ValueError('UTR context reaches a sequence/group edge without both anchors')
            audit.update(region_start=origin+left, region_end=origin+right-1,
                         windows_spanned=int(np.count_nonzero((offsets[:-1] < right) & (offsets[1:] > left))))
            anchor = 64
            a, b = max(0, left-anchor), min(len(path), right+anchor)
            max_bases = int(os.environ.get('ORIONGENO_UTR_REPAIR_MAX_BASES', '4000000'))
            if b-a > max_bases:
                raise MemoryError('Complete conflict interval exceeds UTR repair base budget')
            margin = 40000
            x, z = max(0, a-margin), min(len(path), b+margin)
            parts = []
            for i in range(first, last+1):
                lo, hi = max(x, offsets[i-first]), min(z, offsets[i-first+1])
                if hi <= lo:
                    continue
                row = np.asarray(chunks[i])
                if strand == '-':
                    row = row[::-1]
                parts.append(row[lo-offsets[i-first]:hi-offsets[i-first]])
            sequence = np.concatenate(parts)
            original = path[a:b].copy()
            protected = np.ones(len(original), dtype=np.bool_)
            protected[left-a:right-a] = False
            relative_start = a-x
            if strand == '-':
                sequence = sequence[::-1].copy()
                original = original[::-1].copy()
                protected = protected[::-1].copy()
                relative_start = z-b
            proposal, details = callback(sequence, original, protected, relative_start)
            proposal = np.asarray(proposal)
            if proposal.shape != original.shape or not np.issubdtype(proposal.dtype, np.integer):
                raise ValueError('Repair returned an invalid path shape/dtype')
            if np.any((proposal < 0) | (proposal > 19)):
                raise ValueError('Repair returned an invalid public state')
            if not np.array_equal(proposal[protected], original[protected]):
                raise ValueError('Repair changed fixed flanking labels')
            coding = (original >= 1) & (original <= 14)
            if not np.array_equal(proposal[coding], original[coding]):
                raise ValueError('Repair changed the existing CDS/coding-intron footprint')
            # Validate public topology and UTR memory throughout the replacement.
            side = None
            for i, value in enumerate(proposal):
                if i and not boundary_transition(proposal[i-1], value, '+'):
                    raise ValueError('Repair contains an invalid state transition')
                if value in (15, 16):
                    if i and proposal[i-1] == 19 and side is not None and value != side:
                        raise ValueError('Repair retains a UTR side conflict')
                    side = int(value)
                elif value not in (17, 18, 19):
                    side = None
            repaired = proposal if strand == '+' else proposal[::-1]
            audit.update(details)
            audit.update(status='repaired', changed_bases=int(np.count_nonzero(repaired != path[a:b])),
                         decode_start=origin+a, decode_end=origin+b-1,
                         before_runs=state_runs(path[left:right], origin+left),
                         after_runs=state_runs(repaired[left-a:right-a], origin+left),
                         anchors_unchanged=True, existing_coding_unchanged=True,
                         new_coding_bases=int(np.count_nonzero(((proposal >= 1) & (proposal <= 14)) & ~coding)),
                         before_sha256=hashlib.sha256(path[a:b].astype('int16').tobytes()).hexdigest(),
                         after_sha256=hashlib.sha256(repaired.astype('int16').tobytes()).hexdigest())
            for i in range(first, last+1):
                lo, hi = max(a, offsets[i-first]), min(b, offsets[i-first+1])
                if hi > lo:
                    labels[i][lo-offsets[i-first]:hi-offsets[i-first]] = repaired[lo-a:hi-a]
        except (ValueError, MemoryError) as error:
            audit['reason'] = str(error)
        audit['seconds'] = round(time.perf_counter()-started, 6)
        audits.append(audit)
    return audits


def joint_decode(predictor, sequence, original, protected, offset, probabilities=None):
    """Reuse provided probabilities or forward bounded, overlapping local tiles.

    One exact constrained Viterbi spans the entire interval, independent of the
    forward tiles. Existing coding states and all outside anchors are immutable.
    A noncoding competitor can explain UTR-like evidence without requiring an
    invented ORF. Any new coding path must win against that scored alternative
    and satisfy the usual codon/splice constraints.
    """
    import torch
    from ..hmm_layer.coding_constraints import (
        _competition_graph, _graph, _decode, _legal, _short_orf_competition_regions,
    )

    layer = predictor.gene_pred_hmm_layer
    if layer is None or layer.num_models != 1 or layer.num_copies != 1:
        raise ValueError('Joint UTR repair requires one 20-state HMM model/copy')
    sequence = np.asarray(sequence)
    original = np.asarray(original, dtype=np.int32)
    protected = np.asarray(protected, dtype=np.bool_)
    if offset < 0 or offset+len(original) > len(sequence) or len(protected) != len(original):
        raise ValueError('Incomplete nucleotide context or invalid repair anchors')
    stop_patterns = tuple(p for p, weight in layer.stop_codons if weight > 0)
    graph = _competition_graph(1, stop_patterns)
    stops, labels, initial, sources, targets, allowed, public_labels, coding_count = graph
    budget = int(os.environ.get('ORIONGENO_HMM_REPAIR_MEMORY_MB', '1024')) * 1024**2
    if len(original)*(2*len(labels)+160) > budget:
        raise MemoryError('Joint UTR traceback exceeds memory budget')
    source = 'retained_probabilities'
    try:
        if probabilities is None:
            source = 'local_model_prediction'
            flank, core = 40000, predictor.seq_len
            probabilities = np.empty((len(sequence), 20), dtype=np.float16)
            # Use full-size model inputs, but stitch probabilities before HMM.
            for start in range(0, len(sequence), core):
                end = min(len(sequence), start+core)
                tile = np.zeros((core+2*flank, sequence.shape[1]), dtype=sequence.dtype)
                tile[:, 4] = 1
                lo, hi = max(0, start-flank), min(len(sequence), end+flank)
                tile[lo-start+flank:hi-start+flank] = sequence[lo:hi]
                raw = predictor.sequence_prediction(tile[None], save=False, batch_size=1)
                probabilities[start:end] = raw[0, flank:flank+end-start].float().cpu().numpy()
                del raw, tile
        probabilities = np.asarray(probabilities)
        if probabilities.shape != (len(sequence), 20) or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError('Invalid or incomplete retained model probabilities')
        # Compute emissions with full nucleotide context (including codon motifs),
        # then slice. This avoids introducing artificial motif boundaries.
        device = predictor._model_device()
        nuc = torch.as_tensor(sequence[None, :, :5].copy(), dtype=predictor._hmm_dtype(), device=device)
        if sequence.shape[1] >= 9 and not predictor.upper_only:
            nuc[..., :4] += torch.as_tensor(sequence[None, :, 5:9].copy(), dtype=nuc.dtype, device=device)
        with torch.inference_mode():
            probs = predictor._prepare_gene_hmm_inputs(probabilities[None], device)
            layer.cell.recurrent_init()
            emit = layer.cell.emission_probs(layer.concat_inputs(probs, nuc), training=False)
            emit = emit[0, 0, offset:offset+len(original)].detach().float().cpu().numpy().copy()
            transition = layer.cell.log_A_dense[0].detach().float().cpu().numpy()
            init = layer.cell.init_dist.permute(1, 0, 2)[0, 0].detach().float().cpu().numpy()
            bases = nuc[0, offset:offset+len(original)].cpu().numpy()
        nucleotides = np.where((bases[:, :4].max(-1) == 1) & (bases[:, 4] == 0), bases[:, :4].argmax(-1), 4).astype(np.int32)
        # Use the same noncoding alternative as the existing constrained HMM.
        # Preserve existing coding positions and outside anchors, but let scored
        # biological alternatives compete within the complete conflict region.
        protected = protected | ((original >= 1) & (original <= 14))
        strict = _graph(1, stop_patterns)
        path = _decode(emit, transition, init, nucleotides, *strict[1:5],
                       noncoding_start=len(strict[1]), protected=protected, anchor_path=original)
        eligible, established = _short_orf_competition_regions(path)
        if eligible:
            # Match the existing decoder: review complete short ORFs against
            # a scored noncoding explanation, protecting established and
            # partial coding models (including their attached UTRs).
            path = _decode(emit, transition, init, nucleotides, labels, initial, sources, targets,
                           output_labels=public_labels, noncoding_start=coding_count,
                           protected=protected | established, anchor_path=path)
        if not _legal(path, nucleotides, stops, allowed):
            raise ValueError('Joint repair failed the coding/UTR path audit')
        return path, dict(probability_source=source, decode_bases=len(original),
                          model_context_bases=len(sequence), joint_decode=True,
                          noncoding_competition=bool(eligible))
    except torch.cuda.OutOfMemoryError as error:
        predictor._release_cuda_cache(force=True)
        raise MemoryError('Local UTR prediction/emission ran out of GPU memory') from error
