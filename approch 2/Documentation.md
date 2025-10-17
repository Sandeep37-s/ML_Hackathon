Smart Product Pricing Challenge - Methodology
This document outlines the approach used to predict product prices based on catalog content for the Smart Product Pricing Challenge.

1. Methodology
Our solution is a machine learning pipeline that predicts product prices by analyzing textual data from the catalog_content field. The core of the methodology involves robust feature engineering from the text, followed by training a powerful gradient boosting model. The image data was not used in this iteration to focus on maximizing the signal from the rich text descriptions.

The overall workflow is as follows:

Data Preprocessing: Clean and parse the raw catalog_content.

Feature Engineering: Extract both numerical and high-dimensional text features.

Model Training: Train a LightGBM regression model using 5-fold cross-validation.

Prediction: Generate price predictions on the test set by averaging the outputs of the 5 models.

2. Model Architecture & Algorithms
Primary Model: We used the LightGBM (Light Gradient Boosting Machine) regressor. It was chosen for its high performance, efficiency with large datasets, and ability to handle sparse data like TF-IDF features effectively.

Validation Strategy: A 5-fold cross-validation strategy was employed to ensure the model's predictions are robust and to prevent overfitting. The final test set prediction is an average of the predictions from the five models trained during this process.

Target Transformation: The target variable, price, is highly skewed. To stabilize model training and improve performance, we applied a logarithmic transformation (np.log1p) to the price before training and an exponential transformation (np.expm1) to the predictions to revert them to the original price scale.

3. Feature Engineering Techniques
Feature engineering was critical to the model's success. Both numerical and text-based features were created:

Numerical Feature Extraction:

Item Pack Quantity (IPQ): A custom function using regular expressions was developed to parse the catalog_content and extract the quantity of items in a pack (e.g., "pack of 6", "12 pcs"). This was treated as a crucial numerical feature (ipq).

Text Length (text_len): The total character count of the cleaned catalog text.

Word Count (word_count): The total number of words in the cleaned catalog text.

Text-Based Feature Representation:

TF-IDF (Term Frequency-Inverse Document Frequency): The cleaned text was vectorized using TfidfVectorizer. This converts the text into a meaningful numerical representation that captures the importance of words in the product descriptions.

Configuration:

max_features: 50,000 (to capture a wide vocabulary while managing dimensionality).

ngram_range: (1, 2) (to include both single words and two-word phrases, like "stainless steel").

min_df: 3 (to ignore very rare terms).

All engineered features (numerical and TF-IDF) were combined into a single sparse matrix for efficient model training.