# 🌸 Iris Classification using Neural Networks

A deep learning-based solution to the classic Iris classification problem, this project explores multiple neural network architectures, training strategies, and optimization techniques using TensorFlow and Keras.

🔗 GitHub Repository: [https://github.com/lishaangral/iris-problem](https://github.com/lishaangral/iris-problem)

---

## 🧠 Problem Statement

The Iris dataset is a multiclass classification task that involves predicting the species of an Iris flower (Setosa, Versicolor, Virginica) given four numerical features:

* Sepal length
* Sepal width
* Petal length
* Petal width

The objective is to train a neural network that can accurately classify the flower species based on these measurements.

---

## 🧪 Concepts Used

This project serves as a compact guide to several core deep learning principles:

* 🧱 Feedforward Neural Network (Multilayer Perceptron): A fully connected architecture used for supervised learning on tabular data.
* 🔢 One-Hot Encoding: Converts class labels into a format suitable for classification.
* 📉 Loss Function – Categorical Crossentropy: Measures performance for multi-class classification.
* ⚙️ Optimizers – SGD, RMSprop, Adam: Algorithms to minimize loss function.
* 🔁 Batch Training: Training the model in small batches for efficiency and generalization.
* 🧯 Overfitting Control:

  * Dropout
  * L2 Regularization
  * Batch Normalization
* 🛠️ Hyperparameter Tuning: Varying hidden layers, units, activations, optimizers, etc., to maximize accuracy.

---

## 📄 Notebooks Included & Their Purpose

1. Code.ipynb
   ➤ Core notebook for solving the Iris classification problem.
   Includes data loading, model definition, training, and validation. Acts as the base model and baseline comparison for further tuning.

2. ModelOptimisationFunctions.ipynb
   ➤ Utility notebook defining reusable helper functions to construct and train various neural network configurations.
   Reduces duplication across other notebooks and improves modularity.

3. ModelTuning.ipynb
   ➤ Experiments with hyperparameters: number of hidden layers, nodes per layer, optimizers, and learning rates.
   Helps determine the best architecture for high validation accuracy.

4. OverfittingManagement.ipynb
   ➤ Focuses on combating overfitting using techniques like Dropout, L2 regularization, and Batch Normalization.
   Evaluates generalization performance on unseen data.

5. TuningBackPropagation.ipynb
   ➤ Explores tuning strategies specifically for the backpropagation process—such as learning rate adjustments and optimizers—to improve convergence and stability.

6. RootCauseAnanlysis.ipynb
   ➤ Debugs training issues like slow convergence, poor generalization, and accuracy plateaus.
   Includes visualizations and diagnostics to trace underlying model issues.

---

## 🛠️ Tech Stack

* Python 3.8+
* Jupyter Notebook
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib

---

## 🚀 Installation Instructions

Follow these steps to run the notebooks locally:

1. Clone the repository:

   ```bash
   git clone https://github.com/lishaangral/iris-problem
   cd iris-problem
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate   # Windows: env\Scripts\activate
   ```

3. Install Python packages:

   ```bash
   pip install tensorflow numpy pandas matplotlib jupyter
   ```

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Then open any of the .ipynb notebooks in your browser and run the cells.

If you don’t have Jupyter installed:

* Install it using pip:

  ```bash
  pip install notebook
  ```

---

## 📈 Results

* Achieved up to 100% training accuracy with proper tuning and regularization.
* Achieved >95% validation accuracy on multiple architectures.
* Visualizations included for accuracy, loss, and architectural performance.

---
