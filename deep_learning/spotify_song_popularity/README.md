# Spotify Energy Predictor - Deep Learning Learning Project

## Project Goal

This project was built as a **learning-first deep learning project**. The objective was not simply to predict the energy of a Spotify song, but to understand how a neural network is built from scratch and how a production-style machine learning project is structured.

The project intentionally starts from the mathematics behind a single neuron and gradually evolves into a modular, production-style application.

---

# Dataset

The dataset contains approximately **955,000 Spotify songs** with audio features such as:

* Danceability
* Loudness
* Key
* Mode
* Speechiness
* Acousticness
* Instrumentalness
* Liveness
* Valence
* Tempo
* Duration

Target:

```
Energy
```

The original dataset size is approximately **1.55 GB**.

For faster experimentation, training is performed on a configurable subset using PyTorch's `Subset` while preserving the ability to train on the full dataset later.

---

# Learning Journey

## Phase 1 — Understanding a Single Neuron

Instead of immediately using PyTorch modules, the project began by implementing a single neuron manually.

Topics covered:

* Inputs
* Weights
* Bias
* Prediction
* Mean Squared Error (MSE)
* Manual weight updates
* Why random initialization is necessary

We manually modified weights and observed how the loss changed.

This built intuition for optimization before introducing automatic differentiation.

---

## Phase 2 — Gradients and Backpropagation

Next, PyTorch Autograd was introduced.

Topics learned:

* Computational Graph
* `loss.backward()`
* Gradients
* Chain Rule
* Weight updates
* Learning Rate
* Why gradients tell us the direction to move each parameter

We also learned why gradients must be reset every iteration using:

```python
optimizer.zero_grad()
```

---

## Phase 3 — Multiple Neurons

The single neuron was expanded into a neural network.

Topics learned:

* Hidden layers
* Multiple neurons
* Weight matrices
* Forward propagation

We discussed how hidden layer size affects learning capacity and why choosing the number of neurons is a design decision rather than a fixed rule.

---

## Phase 4 — Activation Functions

We built models with and without activation functions.

Topics covered:

* Why stacking linear layers without activation is equivalent to one linear layer
* ReLU
* Non-linearity
* Dead neurons
* Why deep learning would not exist without activation functions

We experimentally compared models with and without ReLU to observe the difference.

---

## Phase 5 — Data Preprocessing

The raw Spotify dataset was converted into model-ready numerical features.

Topics learned:

### Feature Selection

Choosing only relevant columns.

### One-Hot Encoding

Categorical columns:

```
Key
Mode
```

were converted into numerical vectors.

### Standard Scaling

Continuous numerical features were standardized using:

```
StandardScaler
```

to improve optimization.

Topics learned:

* Mean
* Standard deviation
* Why scaling helps gradient descent

---

## Phase 6 — Data Leakage

One of the most important ML concepts.

Initially preprocessing was applied before splitting the dataset.

This was corrected to the proper workflow:

```
Raw Data

↓

Train/Test Split

↓

Fit Scaler on Training Data

↓

Transform Training Data

↓

Transform Test Data
```

The same approach was applied to the OneHotEncoder.

This prevents information from the test set leaking into training.

---

## Phase 7 — Train/Test Split

The dataset was separated into:

* Training set
* Test set

The model never learns from the test set.

The test set is used only after training is complete.

---

## Phase 8 — PyTorch Dataset and DataLoader

Instead of feeding the entire dataset into memory every iteration, we created:

```
SpotifyDataset
```

and loaded batches using:

```
DataLoader
```

Topics learned:

* Dataset abstraction
* Batching
* Shuffling
* Mini-batch Gradient Descent
* Memory efficiency
* Using subsets for faster experimentation

---

## Phase 9 — Building the Neural Network

The first complete PyTorch model was built.

Topics learned:

* nn.Linear
* ReLU
* Forward function
* nn.Module

The project compared different architectures to understand the impact of model complexity.

---

## Phase 10 — Training Loop

The training loop introduced the complete optimization pipeline.

Workflow:

```
Forward Pass

↓

Loss

↓

Backward Pass

↓

Gradient Calculation

↓

Optimizer Step

↓

Repeat
```

Topics learned:

* Adam Optimizer
* Epochs
* Batch training
* Model.train()
* GPU vs CPU

---

## Phase 11 — Validation

Validation was added after each epoch.

Purpose:

* Monitor learning
* Detect overfitting
* Save the best model

Topics learned:

* `model.eval()`
* `torch.no_grad()`
* Difference between Training Loss and Validation Loss

The best model is selected based on validation loss rather than the final epoch.

---

## Phase 12 — Evaluation

After training completed, evaluation was performed on the unseen test set.

Metrics implemented:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R² Score

Key idea:

Training optimizes the model.

Validation monitors the model.

Evaluation measures the final performance.

---

## Phase 13 — Model Persistence

The trained model is saved for future inference.

Artifacts generated:

```
artifacts/

energy_predictor.pth
scaler.pkl
encoder.pkl
```

Topics learned:

* state_dict()
* torch.save()
* torch.load()
* joblib
* Why preprocessing objects must also be saved

A critical lesson:

The saved model stores **weights and biases**, not the model architecture.

The architecture must be recreated before loading weights.

---

## Phase 14 — Production-Style Project Structure

The notebook was refactored into a modular project.

```
src/

config.py

data/
    preprocessing.py
    dataset.py
    dataloaders.py

models/
    energy_predictor.py

training/
    trainer.py
    evaluator.py

inference/
    predictor.py

utils/
    plotting.py
    io.py
    seed.py

main.py
```

Each module has a single responsibility.

---

## Phase 15 — Inference

The trained model is used to predict energy for new songs.

Inference pipeline:

```
Raw Features

↓

Scale Numerical Features

↓

Encode Categorical Features

↓

Convert to Tensor

↓

Model

↓

Predicted Energy
```

Two inference modes were planned:

* User enters song features manually.
* Random song from the dataset is predicted and compared against the actual value.

This demonstrates how a trained model becomes an application rather than just a training exercise.

---

# Software Engineering Lessons

This project was equally about software engineering.

Key design principles:

* Separation of concerns
* Modular architecture
* Dataclasses for structured outputs
* Type hints
* Configuration-driven parameters
* Reusable utility modules
* Saving training artifacts
* Clean project organization

The code evolved from a single notebook into a maintainable project.

---

# Deep Learning Concepts Learned

* Linear Regression as a neural network
* Neurons
* Weights
* Bias
* Matrix multiplication
* Forward propagation
* Loss functions
* Mean Squared Error
* Gradients
* Chain Rule
* Backpropagation
* Automatic Differentiation
* Learning Rate
* Optimizers
* Adam
* Hidden layers
* Activation functions
* ReLU
* Mini-batch Gradient Descent
* Epochs
* Batch Size
* Dataset
* DataLoader
* StandardScaler
* OneHotEncoder
* Train/Test Split
* Data Leakage
* Validation
* Evaluation
* R² Score
* MAE
* MSE
* Saving models
* Loading models
* Model inference
* CPU vs GPU execution

---

# Most Important Takeaways

The biggest lessons from this project were not specific APIs but core machine learning principles:

1. Learn the mathematics before using high-level libraries.
2. A neural network is simply a function whose parameters are learned.
3. Backpropagation is repeated gradient-based optimization.
4. Activation functions enable neural networks to learn non-linear relationships.
5. Proper preprocessing is just as important as the model itself.
6. Never fit preprocessing on the test set (avoid data leakage).
7. Validation guides training; the test set is used only for the final evaluation.
8. A trained model is only useful if it can be saved, loaded, and used for inference.
9. Good software engineering practices make ML projects scalable and maintainable.
10. Understanding the fundamentals makes learning advanced architectures significantly easier.

---

# What's Next

This project establishes the foundation required for building a Small Language Model (SLM).

Many concepts will remain exactly the same:

* Dataset
* DataLoader
* Training loop
* Validation
* Evaluation
* Saving and loading models
* Inference pipeline
* Project structure

The primary changes in the next project will be:

Replace:

```
Tabular Features
```

with

```
Text Tokens
```

Replace:

```
Feed Forward Network
```

with

```
Transformer Architecture
```

Replace:

```
StandardScaler & OneHotEncoder
```

with

```
Tokenizer & Embeddings
```

Everything else—the engineering workflow, experimentation process, and model lifecycle—will remain remarkably similar.

---

# Final Reflection

This project transformed a collection of abstract deep learning concepts into a complete, working application. It provided an end-to-end understanding of how data flows from raw inputs through preprocessing, training, validation, evaluation, persistence, and inference. More importantly, it established a production-style engineering mindset that will carry forward into future projects involving transformers, language models, and modern AI systems.
