import numpy as np
import pytest

from oriongeno.window_pool import WindowPool


def repair_worker(connection, device, config, environment, log_path):
    connection.send({'ready': True, 'batch_size': 1})
    while True:
        message = connection.recv()
        if message is None:
            break
        if message['offset'] < 0:
            connection.send({'id': message['id'], 'repair_error': 'No legal joint path'})
        else:
            connection.send({'id': message['id'], 'path': message['original'],
                             'audit': {'probability_source': 'test'}})
    connection.close()


def test_repair_transport_reuses_worker_after_expected_failure(tmp_path):
    pool = WindowPool(['test'], {}, tmp_path, worker_target=repair_worker)
    path = np.array([0, 15, 17, 18, 19, 16, 0], dtype=np.int16)
    sequence = np.zeros((len(path), 5), dtype=np.uint8)
    anchors = np.array([True, False, False, False, False, False, True])
    try:
        worker = pool.workers[0]['process']
        with pytest.raises(ValueError, match='No legal'):
            pool.repair_utr_region(sequence, path, anchors, -1)
        result, audit = pool.repair_utr_region(sequence, path, anchors, 0)
        np.testing.assert_array_equal(result, path)
        assert audit['probability_source'] == 'test'
        assert pool.workers[0]['process'] is worker and worker.is_alive()
        assert not pool.workers[0]['busy']
    finally:
        pool.close()
    assert worker.exitcode == 0
