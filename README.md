#  COCO 5Class

![Precision](https://img.shields.io/badge/Precision-84.6%25-84ff3c?style=flat-square)
![Recall](https://img.shields.io/badge/Recall-96.7%25-00dcff?style=flat-square)
![F1](https://img.shields.io/badge/F1-90.2%25-ff5050?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-83.5%25-ffbe00?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square&logo=python&logoColor=white)

A CNN-based image classifier trained on COCO 2014 to detect persons. Built with PyTorch, runs on Apple Silicon (MPS).

## Model

- Custom CNN
- 5-class classifier
- Input size: 128×128
- Trained with cosine LR schedule + early stopping

## Results

| Metric | Score |
|---|---|
| Precision | 0.8458% |
| Recall | 0.9668% |
| F1 | 0.9023% |
| Accuracy | 0.8347% |

## Dataset

COCO 2014

| Train images | Val images|
|---|---|
| 51,573 | 25,066 |


## Project Structure

```
├── coco_5class_cnn.ipynb   # training notebook
├── app.py                  # gradio demo
├── requirements.txt
└── src/
    ├── engine.py
    ├── utils.py
    ├── helper_functions.py
    ├── predictions.py
    └── coco_dataset.py
```

## Setup

```bash
pip install -r requirements.txt
```

Update `COCO_ROOT` in the notebook to your local dataset path, then run all cells.

## Demo

Live demo on [Hugging Face Spaces](https://huggingface.co/spaces/veees/coco).
