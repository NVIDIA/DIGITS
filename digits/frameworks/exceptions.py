from digits.utils import subclass, override

@subclass
class BadNetworkException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

