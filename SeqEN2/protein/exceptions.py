class Error(Exception):
    """Base class for other exceptions"""

    pass


class ImmutablePropertyError(Error):
    def __init__(self, **kwargs):
        self.message = f"{kwargs['attr']} is no longer mutable."
        super().__init__(self.message)
