This project explores one-shot face recognition using Siamese neural networks trained on the Labeled Faces in the Wild (LFW) dataset. The goal is to understand how different architectural choices, preprocessing steps, and training strategies affect similarity learning for facial verification

## Project Structure

```
src/
│   ├── models.py       # Machine learning models
│   ├── dataset.py      # Dataset loading and preprocessing
│   ├── train.py        # Training logic
│   └── loss.py         # loss function
notebook.ipynb          # Jupyter Notebook for experimentation
```

## Dataset Overview
* 13,233 images of 5,749 identities
* Highly varied in pose, background, lighting, and image quality
* Train/test constructed with balanced pairs:
    * Train: 1,100 same + 1,100 different
    * Test: 500 same + 500 different

## What We Investigated
* ResNet18 achieved the best accuracy (~69%), but the compact CNN performed nearly as well with far fewer parameters.
* Contrastive loss failed to learn meaningful embeddings in this setup.
* RGB images gave the highest accuracy, while grayscale helped reduce false positives.
* Persistent overfitting indicates the need for stronger regularization or better pair sampling.

## Conclusion
While absolute accuracy remained modest, the experiments provided valuable insight into how architectural design, preprocessing, and loss functions influence Siamese network performance in challenging one-shot facial recognition tasks.

A full detailed report is included in this repository.
