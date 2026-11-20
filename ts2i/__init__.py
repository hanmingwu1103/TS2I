"""Time-series-to-image representations for classification by a convolutional neural network."""
from .representations import REPRESENTATIONS, TimeSeriesImageEncoder

__all__ = ["make_encoder", "TimeSeriesImageEncoder", "REPRESENTATIONS", "EXISTING_ENCODINGS"]

EXISTING_ENCODINGS = ("GAF", "MTF", "RP", "CWT")


def make_encoder(name: str):
    """Encoder for "MQI", "MQI-LCF", "MQI-UDI", "GAF", "MTF", "RP" or "CWT" (methods ``fit`` and ``transform``)."""
    if name in REPRESENTATIONS:
        return TimeSeriesImageEncoder(name)
    if name in EXISTING_ENCODINGS:
        from .encodings import ExistingImageEncoder
        return ExistingImageEncoder(name)
    raise ValueError(f"unknown representation {name!r}")
