# Image Category Predictor Using Bayes and Dinov2 Features

## Overview

This project provides a tool for image classification into user-defined categories using a Naive Bayes classifier. The model utilizes features generated by Dinov2, which transforms images into tensors of 1024 dimensions, capturing meaningful patterns.

## Features

- **Custom Categories**: The model allows you to define any image category by adding relevant images to the `data` folder.
- **Memory Efficient**: Uses ~80x less memory compared to solutions that store full feature tensors for all training images.
- **Highly Accurate**: Achieves over **75% accuracy** on a 7-category classification task, with significant potential for further improvement with more training data.
- **Scalable**: Adding new categories requires only a few images in the `data` folder, and the program adapts automatically.

## Data Structure

- The dataset is stored in the `data` folder.
- Images are named in the format `<category>[<unique_id>].<extension>` (e.g., `beach7.jpg`).

## How It Works

### Dinov2 Feature Extraction
Dinov2 converts images into 1024-dimensional tensors. Images from the same category produce similar feature patterns.

### Feature Representation
Features are defined as boolean intervals of the form `(L, R, Threshold)`, where:
- `L` and `R` are indices in the tensor.
- The feature is `True` if the sum of elements in `[L, ..., R]` is ≤ `Threshold`.

This representation balances memory efficiency and statistical relevance.

### Model Training
1. **Feature Selection**:
   - Random small intervals are selected.
   - The average value for each interval is computed using training images.

2. **Probability Estimation**:
   - For each category, probabilities of having each feature are calculated.

### Model Prediction
For a new image:
1. Dinov2 extracts its feature tensor.
2. The model checks the boolean status of each feature interval.
3. Using Bayes' theorem, the category with the highest probability is selected.

### Bayes' Formula
The classifier uses the formula:
P(category | feature) = (P(feature | category) * P(category)) / P(feature)
