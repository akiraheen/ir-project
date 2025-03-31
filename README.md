# ir-project

Evaluating the Robustness of Image Retrieval Models Against Query Variability in Multimodal Datasets

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python experiment.py --help
```

### Example

```bash
python experiment.py \
    --data-dir data/Yummly28K/images27638 \
    --results-dir results \
    --tables-dir tables \
    --num-images 10 \
    --iterations 10 \
    --seed 42
```
