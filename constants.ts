
import { TopicCategory } from './types';
import { 
    Robot, Layers, ChartLineUp, GitBranch, Network, CloudDrought,
    Users, Shapes, Target, Gauge, Scale, Library, BarChart, Code
} from './components/icons/Icons';

export const TOPIC_CATEGORIES: TopicCategory[] = [
    {
        name: 'Introduction to AI & ML',
        topics: [
            {
                id: 'what-is-ai',
                title: 'What is Artificial Intelligence?',
                category: 'Introduction to AI & ML',
                description: 'Explore the history, goals, and branches of AI.',
                content: `Artificial Intelligence (AI) is a wide-ranging branch of computer science focused on building smart machines capable of performing tasks that typically require human intelligence. The goals of AI research include reasoning, knowledge representation, planning, learning, natural language processing (NLP), perception, and the ability to move and manipulate objects. Modern AI is divided into several subfields, including:
- **Machine Learning:** Systems that learn from data.
- **Natural Language Processing:** Understanding and generating human language.
- **Computer Vision:** Interpreting and understanding the visual world.
- **Robotics:** Designing and building intelligent robots.
- **Expert Systems:** Emulating the decision-making ability of a human expert.`,
                icon: Robot,
                color: 'text-orange-500',
                bookmarked: false,
            },
        ]
    },
    {
        name: 'Machine Learning Paradigms',
        topics: [
            {
                id: 'supervised-learning',
                title: 'Supervised Learning',
                category: 'Machine Learning Paradigms',
                description: 'Learning from labeled data with a clear output.',
                content: `Supervised learning is the most common type of machine learning. It involves training a model on a dataset where the input data is "labeled" with the correct output. The goal is to learn a mapping function that can predict the output for new, unseen data.
- **Classification:** Predicts a category or class. Example: Classifying an email as 'spam' or 'not spam'.
- **Regression:** Predicts a continuous value. Example: Predicting the price of a house based on its features.
Common algorithms include Linear Regression, Logistic Regression, Support Vector Machines (SVM), and Decision Trees.`,
                icon: Users,
                color: 'text-green-500',
            },
            {
                id: 'unsupervised-learning',
                title: 'Unsupervised Learning',
                category: 'Machine Learning Paradigms',
                description: 'Finding hidden patterns in unlabeled data.',
                content: `In unsupervised learning, the model is given a dataset without explicit labels. The goal is to find hidden structures, patterns, or relationships within the data.
- **Clustering:** Groups similar data points together. Example: Segmenting customers into different groups based on purchasing behavior.
- **Dimensionality Reduction:** Reduces the number of variables in a dataset while preserving important information. Example: Compressing a large image file.
Common algorithms include K-Means Clustering and Principal Component Analysis (PCA).`,
                icon: Shapes,
                color: 'text-pink-500',
            },
            {
                id: 'reinforcement-learning',
                title: 'Reinforcement Learning',
                category: 'Machine Learning Paradigms',
                description: 'Learning through trial and error with rewards.',
                content: `Reinforcement Learning (RL) involves an "agent" that learns to make decisions by performing actions in an "environment" to maximize a cumulative "reward". The agent learns from the consequences of its actions, rather than from being explicitly taught. It's a trial-and-error process.
This approach is widely used in robotics for industrial automation, in self-driving cars, and for training models to play games like Chess or Go at a superhuman level.`,
                icon: Target,
                color: 'text-sky-400',
            },
        ]
    },
    {
        name: 'Core Algorithms',
        topics: [
             {
                id: 'linear-regression',
                title: 'Linear Regression',
                category: 'Core Algorithms',
                description: 'Understand how to predict continuous values.',
                content: `Linear regression is a foundational algorithm used for regression tasks. It models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. The goal is to find the best-fitting straight line (or hyperplane) that minimizes the distance between the predicted and actual values. It's widely used in financial forecasting, sales prediction, and scientific analysis.`,
                icon: ChartLineUp,
                color: 'text-pink-500',
                chartData: [
                    { name: 'X=1', value: 2 }, { name: 'X=2', value: 4.1 }, { name: 'X=3', value: 5.9 },
                    { name: 'X=4', value: 8.2 }, { name: 'X=5', value: 10 }, { name: 'X=6', value: 11.8 },
                ],
            },
            {
                id: 'logistic-regression',
                title: 'Logistic Regression',
                category: 'Core Algorithms',
                description: 'A powerful algorithm for binary classification tasks.',
                content: `Despite its name, Logistic Regression is used for classification, not regression. It predicts the probability of an outcome belonging to a certain class. The output is passed through a Sigmoid function, which squashes the value between 0 and 1. If the probability is above a certain threshold (e.g., 0.5), the model predicts one class; otherwise, it predicts the other. It's commonly used in medical diagnosis (e.g., predicting if a tumor is malignant) and credit scoring.`,
                icon: GitBranch, // Placeholder icon
                color: 'text-orange-500',
            },
            {
                id: 'decision-trees',
                title: 'Decision Trees',
                category: 'Core Algorithms',
                description: 'A flowchart-like structure for classification and regression.',
                content: `A decision tree is a versatile algorithm that can be used for both classification and regression. It creates a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. It's a flowchart-like structure where each internal node represents a feature-based test, each branch represents an outcome, and each leaf node represents a class label or a continuous value. They are intuitive and easy to interpret.`,
                icon: GitBranch,
                color: 'text-green-500',
            },
            {
                id: 'random-forests',
                title: 'Random Forests',
                category: 'Core Algorithms',
                description: 'An ensemble method using multiple decision trees.',
                content: `A Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned. It corrects for the habit of decision trees to overfit to their training set by using bagging and feature randomness, resulting in a more robust and accurate model.`,
                icon: GitBranch,
                color: 'text-purple-500',
            },
            {
                id: 'svm',
                title: 'Support Vector Machines (SVM)',
                category: 'Core Algorithms',
                description: 'Finds the optimal hyperplane to separate data points.',
                content: `Support Vector Machines (SVMs) are powerful classifiers that find the optimal hyperplane that best separates data points of different classes in a high-dimensional space. The "best" hyperplane is the one with the largest margin, which is the distance between the hyperplane and the closest data points from each class (called support vectors). SVMs can also handle non-linear data by using the "kernel trick" to project the data into a higher dimension where it becomes separable.`,
                icon: Layers, // Placeholder icon
                color: 'text-teal-500',
            },
            {
                id: 'knn',
                title: 'K-Nearest Neighbors (KNN)',
                category: 'Core Algorithms',
                description: 'An instance-based algorithm for classification and regression.',
                content: `K-Nearest Neighbors (KNN) is a simple, non-parametric, and instance-based learning algorithm. It's considered a "lazy learner" because it doesn't build a model during the training phase. Instead, it stores the entire training dataset. When a prediction is required for a new data point, KNN finds the 'k' most similar instances (neighbors) from the training data. For classification, it assigns the class that is most common among its k-nearest neighbors. For regression, it assigns the average of the values of its k-nearest neighbors. The choice of 'k' and the distance metric (e.g., Euclidean) are crucial for its performance.`,
                icon: Shapes,
                color: 'text-sky-400',
            },
            {
                id: 'naive-bayes',
                title: 'Naïve Bayes Classifiers',
                category: 'Core Algorithms',
                description: 'A probabilistic classifier based on Bayes\' Theorem.',
                content: `Naïve Bayes is a family of simple probabilistic classifiers based on applying Bayes' theorem with a strong ("naive") independence assumption between the features. This means it assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. Despite this often unrealistic assumption, Naïve Bayes classifiers are highly effective in practice, particularly for text classification tasks like spam filtering and document categorization, and they are computationally very efficient.`,
                icon: Scale,
                color: 'text-pink-500',
            },
        ]
    },
    {
        name: 'Neural Networks',
        topics: [
            {
                id: 'intro-to-nn',
                title: 'Introduction to Neural Networks',
                category: 'Neural Networks',
                description: 'The building blocks of Deep Learning.',
                content: `Artificial neural networks (ANNs) are computing systems inspired by the biological neural networks of animal brains. An ANN is based on a collection of connected units or nodes called artificial neurons. Each connection can transmit a signal to other neurons. An artificial neuron that receives a signal processes it and can signal neurons connected to it. This process allows the network to learn complex patterns from data, forming the foundation of deep learning.`,
                icon: Network,
                color: 'text-purple-500',
            },
             {
                id: 'drought-prediction-lstm',
                title: 'Drought Prediction using LSTM Networks',
                category: 'Neural Networks',
                description: 'A practical application of RNNs for climate modeling in India.',
                content: `Long Short-Term Memory (LSTM) networks are a type of Recurrent Neural Network (RNN) exceptionally suited for time-series forecasting. In the context of India, where agriculture is heavily dependent on monsoon patterns, predicting meteorological droughts is a critical application of ML.
### Why LSTM?
LSTMs are designed with a memory cell that can maintain information for long periods. This "memory" is crucial for understanding climate patterns, where a data point (e.g., rainfall in June) is highly dependent on previous data points.
### The Indian Context
For a project in an Indian engineering college, you can leverage publicly available datasets from organizations like the Indian Meteorological Department (IMD) or data.gov.in.
### The Model Pipeline
1.  **Data Collection & Preprocessing:** Gather time-series data for a specific region (e.g., monthly rainfall, SPI, soil moisture).
2.  **Model Architecture:** A typical LSTM model would consist of LSTM layers, Dropout layers to prevent overfitting, and a Dense output layer to predict the target variable.
3.  **Training & Evaluation:** The model is trained on historical data and its performance is evaluated using metrics like Mean Squared Error (MSE).
### Conceptual Code Example (Python with Keras/TensorFlow)
\`\`\`python
# This is a conceptual code snippet to illustrate the structure.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
# Assume X_train is 3D array [samples, timesteps, features]
# Assume y_train is 2D array [samples, output_values]
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)
\`\`\`
This application is a powerful example for an engineering final year project, as it combines data science, deep learning, and has a significant real-world impact.`,
                icon: CloudDrought,
                color: 'text-sky-400',
            }
        ]
    },
     {
        name: 'Model Evaluation & Concepts',
        topics: [
            {
                id: 'training-testing',
                title: 'Training, Validation, and Testing',
                category: 'Model Evaluation & Concepts',
                description: 'How to properly split data to evaluate models.',
                content: `To build a reliable machine learning model, you must split your data into separate sets:
- **Training Set:** The largest portion of the data, used to train the model. The model learns the patterns and relationships from this data.
- **Validation Set:** Used to tune the model's hyperparameters (e.g., the number of trees in a random forest) and make decisions about the model's architecture. It helps prevent overfitting on the training data.
- **Testing Set:** A completely unseen portion of the data used for the final evaluation of the model's performance. This provides an unbiased estimate of how the model will perform in the real world. A typical split might be 70% for training, 15% for validation, and 15% for testing.`,
                icon: Scale,
                color: 'text-orange-500',
            },
            {
                id: 'evaluation-metrics',
                title: 'Accuracy, Precision, Recall & F1-Score',
                category: 'Model Evaluation & Concepts',
                description: 'Key metrics to measure a classification model\'s performance.',
                content: `While accuracy is a common metric, it can be misleading, especially with imbalanced datasets. A more nuanced evaluation uses:
- **Accuracy:** (TP + TN) / Total. The overall percentage of correct predictions.
- **Precision:** TP / (TP + FP). Of all the positive predictions, how many were actually correct? It measures the model's exactness. High precision is important when the cost of a false positive is high (e.g., a spam filter marking a real email as spam).
- **Recall (Sensitivity):** TP / (TP + FN). Of all the actual positive cases, how many did the model correctly identify? It measures the model's completeness. High recall is important when the cost of a false negative is high (e.g., a medical test failing to detect a disease).
- **F1-Score:** The harmonic mean of Precision and Recall (2 * (Precision * Recall) / (Precision + Recall)). It provides a single score that balances both concerns.`,
                icon: Gauge,
                color: 'text-teal-500',
            },
            {
                id: 'bias-variance-tradeoff',
                title: 'Bias-Variance Tradeoff',
                category: 'Model Evaluation & Concepts',
                description: 'The fundamental challenge of balancing model simplicity and complexity.',
                content: `The Bias-Variance Tradeoff is a central concept in machine learning.
- **Bias:** The error from erroneous assumptions in the learning algorithm. High bias can cause a model to miss relevant relations between features and target outputs (underfitting). A simple model like Linear Regression often has high bias.
- **Variance:** The error from sensitivity to small fluctuations in the training set. High variance can cause a model to model the random noise in the training data rather than the intended outputs (overfitting). A complex model like a deep Decision Tree often has high variance.
The goal is to find a sweet spot in the middle. Increasing a model's complexity will typically decrease its bias but increase its variance. The challenge is to find the optimal level of complexity where the error is minimized.`,
                icon: Scale,
                color: 'text-pink-500',
            },
        ]
    },
    {
        name: 'The ML Toolkit',
        topics: [
            {
                id: 'python-libraries',
                title: 'Essential Python Libraries',
                category: 'The ML Toolkit',
                description: 'An overview of NumPy and Pandas for data manipulation.',
                content: `Python is the dominant language for machine learning, largely due to its powerful libraries.
### NumPy
NumPy (Numerical Python) is the fundamental package for scientific computing. Its core feature is the powerful N-dimensional array object, which provides fast operations on arrays.
\`\`\`python
import numpy as np
# Create a NumPy array
a = np.array([1, 2, 3])
print(a * 2)  # Output: [2 4 6]
\`\`\`
### Pandas
Pandas is built on top of NumPy and is the go-to library for data analysis and manipulation. It introduces the DataFrame, a two-dimensional labeled data structure with columns of potentially different types, similar to a spreadsheet or SQL table.
\`\`\`python
import pandas as pd
# Create a DataFrame
data = {'Name': ['Amit', 'Priya'], 'Age': [21, 22]}
df = pd.DataFrame(data)
print(df)
\`\`\``,
                icon: Library,
                color: 'text-green-500',
            },
            {
                id: 'ml-frameworks',
                title: 'Core ML Frameworks',
                category: 'The ML Toolkit',
                description: 'Learn about Scikit-learn, TensorFlow, and PyTorch.',
                content: `These frameworks provide the tools to build and train machine learning models.
### Scikit-learn
A simple and efficient library for traditional machine learning. It provides a wide range of supervised and unsupervised learning algorithms, as well as tools for model selection and evaluation. It's the perfect starting point for most ML tasks.
### TensorFlow & PyTorch
These are the two leading deep learning frameworks. They provide powerful tools for building, training, and deploying complex neural networks. They use tensors (multi-dimensional arrays) as their basic data structure and support GPU acceleration for high-performance computation. TensorFlow is known for its production-readiness (TensorFlow Extended - TFX), while PyTorch is often praised for its flexibility and ease of use in research.`,
                icon: Code,
                color: 'text-purple-500',
            },
            {
                id: 'data-visualization',
                title: 'Data Visualization',
                category: 'The ML Toolkit',
                description: 'Using Matplotlib and Seaborn to visualize data.',
                content: `Visualizing data is crucial for understanding it (Exploratory Data Analysis - EDA) and for communicating results.
### Matplotlib
The foundational plotting library in Python. It's highly customizable but can be verbose for complex plots. It can create static, animated, and interactive visualizations.
\`\`\`python
import matplotlib.pyplot as plt
# Simple line plot
plt.plot([1, 2, 3], [2, 4, 1])
plt.show()
\`\`\`
### Seaborn
Built on top of Matplotlib, Seaborn provides a high-level interface for drawing attractive and informative statistical graphics. It makes it easier to create complex plots like heatmaps, violin plots, and pair plots.
\`\`\`python
import seaborn as sns
# Load example dataset
tips = sns.load_dataset("tips")
# Create a scatter plot
sns.scatterplot(data=tips, x="total_bill", y="tip")
\`\`\``,
                icon: BarChart,
                color: 'text-sky-400',
            }
        ]
    }
];
