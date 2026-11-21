# TS2I: time-series-to-image representations for classification

Code and score files for the paper

> Yu-Pei Wu and Han-Ming Wu. *A Time-Series-to-Image Representation for Deep Learning-Based Classification:
> Local Quantiles and Level-Conditioned Increments.* Manuscript, 2026.

The paper proposes three image representations of a univariate time series, each a stack of planes of
32 x 32 pixels, and classifies the images with a common compact residual CNN.

| Representation | Planes | Content |
|---|---|---|
| MQI | 6 | Multiscale quantile image: local quantiles of the level, the lag sum and the lag difference, in overlapping windows at two scales |
| MQI-LCF | 10 | MQI and four level-conditioned directional increment planes: kernel-weighted root mean squares of the positive and of the negative increments according to the ranks of their arrival levels |
| MQI-UDI | 10 | MQI and four unconditional directional increment planes: root mean squares of the positive and of the negative increments of each window |

## Contents

| Path | Content |
|---|---|
| `ts2i/representations.py` | MQI, MQI-LCF and MQI-UDI (`TimeSeriesImageEncoder`) |
| `ts2i/encodings.py` | The existing image encodings used for comparison: GAF, MTF, RP and a CWT image |
| `ts2i/cnn.py` | The compact residual CNN, its training and the selection of the learning rate |
| `ts2i/pipeline.py` | Classification of one dataset with one representation (`evaluate`) |
| `ts2i/simulation.py` | Data-generating mechanism of the simulation study |
| `examples/simulation_example.py` | One replicate of one simulation setting |
| `analysis/reproduce_tables.py` | Recomputes the numerical results of the paper from the score files |
| `analysis/simulation_logistic_regression.py` | Simulation study with logistic regression on the pixels in place of the CNN |
| `analysis/realdata_logistic_regression.py` | The two studies on real data with logistic regression on the pixels in place of the CNN |
| `analysis/summarize_realdata_logistic.py` | Summaries of the logistic regression on real data, from the score file |
| `analysis/realdata_studies_metadata.json` | Datasets, files and blocks of the two studies on real data, as read by the script above |
| `results/` | Test-set scores of the simulation study and of the studies on real data |
| `tests/test_basic.py` | Basic checks of the package |

## Installation

Python 3.12 or later with the packages listed in `requirements.txt`:

```
pip install -r requirements.txt
```

The studies were run with Python 3.12, NumPy 2.3, SciPy 1.17, scikit-learn 1.9, PyTorch 2.15 (CPU) and
PyWavelets 1.9. `pyts` and `PyWavelets` are needed only for the existing encodings, and `torch` only for RP and
for the CNN.

## Usage

Images of a set of series of common length (arrays of shape `(n, T)`):

```python
from ts2i import make_encoder

encoder = make_encoder("MQI-LCF").fit(X_train)   # "MQI", "MQI-LCF", "MQI-UDI", "GAF", "MTF", "RP" or "CWT"
images = encoder.transform(X_test)               # float32 array of shape (n, 10, 32, 32)
```

`fit` estimates all constants (mean and standard deviation of the series, ranges used for clipping, mean and
standard deviation of every plane) from the series passed to it; `transform` holds them fixed.

Classification of a dataset with given training and test sets:

```python
from ts2i.pipeline import evaluate

scores = evaluate("MQI-LCF", X_train, y_train, X_test, y_test, seed=101)
print(scores["macro_f1"], scores["accuracy"])
```

`evaluate` divides the training set into a fitting subset and a validation subset, selects the learning rate
and the number of epochs on the validation subset, trains the final network on the whole training set and
scores it on the test set. The seeds of the paper are 101, 202 and 303 for the datasets and 61000 + r for
replicate r of the simulation study.

One replicate of the simulation study:

```
python examples/simulation_example.py --replicate 1 --setting A35
```

## Recomputing the results of the paper

```
python analysis/reproduce_tables.py
```

prints, from the files in `results/`, the entries of the tables of the simulation study, of the comparison
with existing image encodings, of the development study and of the independent evaluation, including the
results by dataset. The adjusted p-values of the comparison of MQI with the existing image encodings are not
recomputed by the script.

```
python analysis/simulation_logistic_regression.py
```

prints the results of the simulation study when the images are classified by logistic regression with an L2
penalty on the pixels (a sensitivity analysis reported in the paper). With `--run out.csv` the 360 classifiers
are recomputed, which takes about 15 minutes on two cores; logistic regression does not depend on the PyTorch
build.

```
python analysis/summarize_realdata_logistic.py
```

prints the corresponding results of the two studies on real data from `results/realdata_scores_logistic.csv`:
the seven representations on the 30 development datasets and the three proposed representations in the
independent evaluation, with the summaries of the analyses with the CNN beside them. The scores were computed
with `analysis/realdata_logistic_regression.py` in the stages `encode`, `select`, `final` and `collect` (see its
docstring), which requires the datasets and took about nine hours of processor time.

The logistic regressions are fitted with scikit-learn's L-BFGS with tolerance 1e-8 and at most 100,000
iterations, in the coordinates of the row space of the training images when the pixels outnumber the series.
With the default tolerance of scikit-learn the fits with weak penalties stop before the minimum of the
penalized likelihood, and scikit-learn leaves SciPy's limit of 15,000 evaluations of the objective in force;
the scripts remove this limit.

| File | Rows |
|---|---|
| `results/simulation_scores.csv` | 6 settings x 20 replicates x 3 representations |
| `results/simulation_scores_logistic.csv` | The same, with logistic regression on the pixels in place of the CNN |
| `results/realdata_scores_logistic.csv` | Logistic regression on the pixels: 30 development datasets x 7 representations (divisions of seeds 101, 202 and 303 for the proposed representations, 101 and 202 for the existing encodings) and 17 datasets of the independent evaluation x 3 representations x 3 divisions |
| `results/existing_encodings_scores.csv` | 30 development datasets x seeds 101 and 202 x MQI, CWT, GAF, MTF and RP |
| `results/development_scores.csv` | 30 development datasets x 3 seeds x 3 representations |
| `results/independent_scores.csv` | 17 datasets of the independent evaluation (in nine blocks) x 3 seeds x 3 representations |
| `results/development_datasets.txt` | The 30 development datasets in the order used for the bootstrap intervals |

## Data

The datasets are not redistributed here. The 30 development datasets are from the UCR Time Series
Classification Archive (Dau et al., 2019; https://www.cs.ucr.edu/~eamonn/time_series_data_2018/). Of the 17
datasets of the independent evaluation, 14 were obtained from the time series classification archive
described by Middlehurst et al. (2024; https://www.timeseriesclassification.com) and the three Asphalt
datasets from Zenodo (records 11217990, 11217994 and 11217996). All datasets are used with the training and
test sets of their sources. The series of the simulation study are generated by `ts2i/simulation.py`.

## Reproducibility

The series of the simulation study and the images of all representations are computed deterministically.
Training uses fixed seeds, one thread and deterministic algorithms, but fitted networks are not guaranteed to
be identical across PyTorch builds and platforms, so individual scores can differ from those in `results/`,
in particular for classifiers near the chance level. The files in `results/` are the scores of the runs
reported in the paper. The fitted networks and the bookkeeping code with which the studies were run
(caching of images, logging and checks of the saved files) are not part of this repository.

## License

MIT License; see `LICENSE`.
