# lightcone


[![Test Package](https://github.com/windisch/lightcone/actions/workflows/test_package.yml/badge.svg)](https://github.com/windisch/lightcone/actions/workflows/test_package.yml)
[![Documentation Status](https://readthedocs.org/projects/lightcone/badge/?version=latest)](https://lightcone.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/lightcone)](https://pypi.org/project/lightcone/)

A framework to explore the latent space of convolutional autoencoders
implemented in `pytorch`.

## Example

Compose your decoder and your encoder into the `lightcone`
autoencoder:

```python
from lightcone.models import AutoEncoder

model = AutoEncoder(encoder=your_encoder, decoder=your_decoder)
```

After `model` has been training, the latent space can be explored in a
Jupyter-Notebook as follows

```python
model.explore(data_loader=your_data_loader)
```


## Jupyter Dash
Make sure to install and activate the Jupyter notebook extenstion

```bash
jupyter nbextension install --py jupyter_dash
jupyter nbextension enable --py jupyter_dash
```

