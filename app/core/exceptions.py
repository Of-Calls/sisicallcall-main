class SisicollcollError(Exception):
    pass


class TenantNotFoundError(SisicollcollError):
    pass


class CallNotFoundError(SisicollcollError):
    pass


class STTError(SisicollcollError):
    pass


class TTSError(SisicollcollError):
    pass


class LLMError(SisicollcollError):
    pass


class EmbeddingError(SisicollcollError):
    pass


class CacheError(SisicollcollError):
    pass


class KNNRouterError(SisicollcollError):
    pass


class SpeakerVerifyError(SisicollcollError):
    pass
