# Person classifier

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![HuggingFaces](https://img.shields.io/badge/Huggingfaces-Live%20Demo-FF4B4B.svg)](https://huggingface.co/spaces/veees/coco)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A CNN-based image classifier trained on COCO 2014 to detect persons. Built with PyTorch.

---

## Model

- Custom `CNN`
- 5-class `classifier`
- Input size: `128×128`
- Trained with `cosine LR schedule` + `early stopping`

---

## Results

```python

Precision  =  0.8458% 
Recall     =  0.9668% 
F1         =  0.9023% 
Accuracy   =  0.8347% 
```

---

## Dataset



```python
# COCO 2014

Train_images    =  51,573
Val_images      =  25,066
```

---

## Setup

```bash
pip install -r requirements.txt
```

Update `COCO_ROOT` in the notebook to your local dataset path, then run all cells.

---

## Demo

Live demo on [Hugging Face Spaces](https://huggingface.co/spaces/veees/coco).
