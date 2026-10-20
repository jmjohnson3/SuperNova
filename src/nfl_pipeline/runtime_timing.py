"""Small stage timer; failures keep their original exception and do not pass green."""
from time import perf_counter


class StageTimer:
    def __init__(self, logger):
        self.logger = logger
        self.stages = []

    def call(self, name, fn, *args, **kwargs):
        started = perf_counter()
        self.logger.info('Prediction stage starting: %s', name)
        entry = dict(stage=name, status='failed')
        try:
            result = fn(*args, **kwargs)
            entry['status'] = 'ok'
            if isinstance(result, dict):
                entry['keys'] = len(result)
            elif isinstance(result, tuple):
                entry['result_parts'] = len(result)
            elif hasattr(result, '__len__'):
                entry['rows'] = len(result)
            return result
        finally:
            entry['seconds'] = round(perf_counter()-started, 3)
            self.stages.append(entry)
            self.logger.info('Prediction stage finished: %s', entry)
