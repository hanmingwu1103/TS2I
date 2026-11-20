"""Basic checks of the package: python -m pytest tests"""
import numpy as np

from ts2i import make_encoder, simulation


def test_shapes_and_udi_rows():
    rng = np.random.default_rng(0)
    X, Z = np.cumsum(rng.normal(size=(20, 150)), axis=1), np.cumsum(rng.normal(size=(5, 150)), axis=1)
    for name, planes in (("MQI", 6), ("MQI-LCF", 10), ("MQI-UDI", 10)):
        images = make_encoder(name).fit(X).transform(Z)
        assert images.shape == (5, planes, 32, 32) and images.dtype == np.float32 and np.isfinite(images).all()
    udi = make_encoder("MQI-UDI").fit(X).transform(Z)[:, 6:]
    assert (udi == udi[:, :, :1, :]).all()                      # every column of an unconditional plane is constant
    mqi, lcf = make_encoder("MQI").fit(X).transform(Z), make_encoder("MQI-LCF").fit(X).transform(Z)
    assert np.array_equal(mqi, lcf[:, :6])                      # the first six planes of MQI-LCF are MQI


def test_quantile_rows_are_nondecreasing():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 90))
    enc = make_encoder("MQI").fit(X)
    Q = enc._quantile_planes(X)                                 # before the standardization of the planes
    assert (np.diff(Q, axis=2) >= 0).all() and Q.min() >= 0 and Q.max() <= 1


def test_simulation_is_reproducible_and_balanced():
    X1, y1 = simulation.generate(3, "A70", "train")
    X2, y2 = simulation.generate(3, "A70", "train")
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert X1.shape == (128, 401) and np.bincount(y1).tolist() == [64, 64]
    assert simulation.generate(3, "A70", "test")[0].shape == (512, 401)
