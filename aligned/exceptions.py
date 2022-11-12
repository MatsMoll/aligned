class UnableToFindFileException(Exception):
    pass


class InvalidStandardScalerArtefact(Exception):
    pass


class CombinedFeatureViewQuerying(Exception):
    pass


class NotSupportedYet(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f'{message}. What about contributing and adding a PR for this?')
