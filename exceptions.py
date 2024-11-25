class UnableToFindFileException(Exception):
    pass


class CombinedFeatureViewQuerying(Exception):
    pass


class NotSupportedYet(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f'{message}. What about contributing and adding a PR for this?')


class StreamWorkerNotFound(Exception):
    def __init__(self, module: str) -> None:
        super().__init__(
            f'Unable to find the stream worker. Tried to load module "{module}". '
            'This is needed in order to know where to store the processed features. '
            'Try adding a worker.py file in your root folder and define a StreamWorker object.'
        )
