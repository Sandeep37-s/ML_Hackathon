Smart Product Pricing Challenge: Methodology
This document outlines the approach used to predict product prices based on their catalog content and images.

1. Methodology
Our solution is a multi-modal machine learning pipeline that leverages both textual and visual data to create a robust price prediction model. The core methodology involves three main stages:

Feature Engineering: Convert unstructured text and image data into meaningful numerical features.

Log Transformation: Normalize the skewed distribution of the target variable (price) to improve model performance.

Ensemble Modeling: Train an ensemble of LightGBM models using K-Fold cross-validation to ensure the predictions are stable and accurate.

2. Feature Engineering Techniques
We generated a comprehensive set of features from the provided catalog_content and image_link.

Image Features:

A pre-trained EfficientNet-B0 model was used as a feature extractor.

Each product image was downloaded, resized to 224x224 pixels, and normalized.

The model then processed the image to generate a 1280-dimension embedding vector, capturing its key visual characteristics.

Text Features:

Semantic Embeddings: A pre-trained all-MiniLM-L6-v2 Sentence Transformer model was used to convert the catalog_content text into a 384-dimension semantic vector. This captures the meaning and context of the product title and description.

Engineered Features: Two simple but effective features were extracted directly from the text:

ipq (Item Pack Quantity): The IPQ value was extracted using a regular expression.

text_len: The character length of the catalog_content.

These features were then combined into a single feature set for each product.

3. Model Architecture
Algorithm: We selected the LightGBM (LGBM) Regressor due to its high performance, speed, and efficiency with the high-dimensional feature set we created.

Training Strategy (5-Fold Cross-Validation):

Instead of training a single model, we trained an ensemble of 5 separate LGBM models.

The training data was split into 5 "folds". In each iteration, one fold was used for validation and the other four were used for training.

Early stopping was employed to prevent overfitting by monitoring performance on the validation fold.

The final prediction for the test set is the average of the predictions from all 5 models. This cross-validation approach makes the model more robust and less sensitive to the specific splits in the training data.

4. Additional Information
Target Transformation: The target variable, price, has a right-skewed distribution common in pricing data. To handle this, we applied a log1p transformation (log(1 + price)) before training. The model was trained to predict this log-transformed value. Final predictions were converted back to the original price scale using the inverse function, expm1. This technique is crucial for optimizing the SMAPE metric.

Batch Processing: To manage memory and disk space, the feature generation process was handled in batches. Images for each batch were downloaded, processed, and then deleted before moving to the next batch.