# Few-Shot Meta-Learning for Omniglot Character Recognition

## Overview

This project implements a robust and versatile few-shot meta-learning system for Omniglot character recognition. It demonstrates the ability to quickly adapt to new character classes with minimal training data. The project incorporates core meta-learning algorithms, neural network design, advanced training techniques, and comprehensive evaluation and visualization.

## Meta-Learning Algorithm Overview

This project implements three distinct meta-learning algorithms:

*   **Prototypical Networks:** This is a metric-based meta-learning approach that learns an embedding space where data points from the same class are close to each other. Class prototypes are computed by averaging the embeddings of the support set, and classification is then performed by assigning each query point to the closest prototype using a distance function such as the Euclidean Distance.

*   **Model-Agnostic Meta-Learning (MAML):** MAML is a gradient-based meta-learning algorithm that learns a model initialization, that can rapidly adapt to new tasks with very few gradient steps, through an inner loop of task-specific fine-tuning. The outer loop then uses gradients from the fine-tuned inner loop weights to update the model in the direction that maximizes the performance across all tasks.

*   **Reptile:** Reptile is a simplified gradient-based meta-learning algorithm that also aims to learn a model initialization that can quickly adapt to new tasks. However, unlike MAML it is first order, making it less computationally complex, and simpler to train. It achieves this by using stochastic gradient descent on the inner loop and then uses the difference in the start and end weights of the inner loop to make an update.

## Key Features

*   **Core Meta-Learning Algorithms:** Implements Prototypical Networks, Model-Agnostic Meta-Learning (MAML), and Reptile algorithms.
*   **Flexible Architecture:** A modular, config-driven design enabling easy experimentation and extensibility.
*   **Advanced Training:** Utilizes data augmentation (RandAugment, TrivialAugment), learning rate scheduling (OneCycleLR), gradient clipping, mixed-precision training, and early stopping.
*   **Comprehensive Evaluation:** Includes automated scripts for detailed performance metric reporting, confusion matrices, and embedding visualization.
*   **Streamlit Demo:** A live Streamlit demo showcasing the models' few-shot learning capabilities.
*   **Efficient Pipeline:** Optimized for GPU acceleration with parallel data loading.
*   **Reproducible:** All experiments are reproducible through a `config.yaml` file.

## Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch 1.10+
*   CUDA (if using a GPU)
*   `pip install -r requirements.txt`
*   Streamlit

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd few-shot-omniglot-meta-learning
    ```
2.  Create and activate a virtual environment:
   ```bash
    python -m venv venv
Use code with caution.
On Windows:

venv\Scripts\activate
Use code with caution.
Bash
On macOS and Linux:

source venv/bin/activate
Use code with caution.
Bash
Install dependencies:

pip install -r requirements.txt
Use code with caution.
Bash
Running the Code
Training:

To train all models:

python train.py
Use code with caution.
Bash
To train a specific model:

python train.py --model protonet
Use code with caution.
Bash
or

python train.py --model maml
Use code with caution.
Bash
or

python train.py --model reptile
Use code with caution.
Bash
Evaluation:

python evaluate.py
Use code with caution.
Bash
The script will prompt you to specify which model to evaluate.

Testing:

python test.py
Use code with caution.
Bash
The script will prompt you to specify which model to test, or whether to test the best model.

Demo:

streamlit run demo.py
Use code with caution.
Bash
Model Performance
The table below shows the approximate accuracy achieved with each of the models. Please note that these are the accuracies after convergence on the evaluation dataset, the test accuracy may vary:

Model	Accuracy
Reptile	0.842
Prototypical Net	0.81
MAML	0.851
