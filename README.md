# DigiForest Registration

Register ground and aerial maps.

## Setup

Use a virtual environment (`env` in the example), isolated from the system dependencies:

```sh
python3 -m venv env
source env/bin/activate
```

Install the dependencies:

```sh
pip install -r requirements.txt
```

Install the automatic formatting pre-commit hooks (black and flake8), which will check the code before each commit:

```sh
pre-commit install
```

Install `digiforest_registration`:

```sh
cd ~/git/digiforest_registration/digiforest_registration
pip install -e .
```

## Execution

To run the example pipeline:

```sh
cd ~/git/digiforest_registration/digiforest_registration
python scripts/registration.py <aerial_pcd_file> <ground_pcd_file>
```
