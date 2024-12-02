class VersionException(Exception):
    def __init__(
        self,
        message="\n\n\033[0;31mKIDPY3 Requires Python Version 3.6 or Later\033[0m\n",
    ):
        self.message = message
        super().__init__(self.message)
