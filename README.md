# Few-Shot Meta-Learning for Omniglot Character Recognition

## Overview

This project implements a cutting-edge few-shot meta-learning system for Omniglot character recognition. Designed to demonstrate rapid adaptability to new character classes with minimal training data, it showcases state-of-the-art meta-learning algorithms, robust neural network architectures, advanced training methodologies, and comprehensive evaluation techniques. The project is built to be modular, scalable, and accessible for both research and production use.

## Meta-Learning Algorithms

This project integrates three advanced meta-learning algorithms:

* **Prototypical Networks:**  
  A metric-based approach that learns an embedding space where data points from the same class cluster closely. Class prototypes are computed by averaging the embeddings of the support set, and classification is performed by assigning query points to the nearest prototype using a distance function such as Euclidean distance.

* **Model-Agnostic Meta-Learning (MAML):**  
  A gradient-based meta-learning algorithm that learns a model initialization capable of adapting to new tasks with minimal gradient updates. MAML employs an inner loop for task-specific fine-tuning and an outer loop that updates the model using gradients derived from fine-tuned weights, maximizing performance across multiple tasks.

* **Reptile:**  
  A simplified first-order gradient-based algorithm that learns task-agnostic model initializations. Reptile eliminates second-order derivatives, reducing computational complexity while maintaining strong performance. It updates the model based on the difference between the start and end weights of the inner loop.

---

## Key Features

* **Core Algorithms:** Implements Prototypical Networks, MAML, and Reptile with modular design for easy customization.
* **Flexible and Modular Architecture:** Highly configurable, allowing researchers to experiment with diverse settings and extend functionalities effortlessly.
* **Advanced Training Techniques:**  
  - **Data Augmentation:** Supports techniques like RandAugment and TrivialAugment.  
  - **Optimized Training:** Includes learning rate scheduling (OneCycleLR), gradient clipping, mixed-precision training, and early stopping for robust performance.  
  - **GPU Acceleration:** Leveraging CUDA for seamless parallel data processing and training.  

* **Comprehensive Evaluation:**  
  - Automated performance metric reporting, including accuracy, precision, recall, and F1 score.  
  - Visualization tools such as confusion matrices and t-SNE embeddings for analyzing model behavior.  

* **Interactive Demo:** A live Streamlit application demonstrates real-time few-shot learning, making it accessible to both technical and non-technical users.  

* **Reproducibility:** Complete reproducibility enabled through a single `config.yaml` file and deterministic seeds.  

---

## Getting Started

### Prerequisites

Ensure the following are installed on your system:
- Python 3.8+
- PyTorch 1.10+ with CUDA support (if using GPU)
- Streamlit

### Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd few-shot-omniglot-meta-learning
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---
