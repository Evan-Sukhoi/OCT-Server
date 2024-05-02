class DataDirNotDetectedError(BaseException):
    def __init__(self, message):
        BaseException.__init__(self)
        self.name = 'DataDirNotDetectedError'
        self.message = message