Smart Product Pricing Challenge

This repository contains the code and methodology for the Smart Product Pricing Challenge. The goal is to predict product prices using their catalog information and images. Two primary approaches were developed and are described below.

Approach 1: Baseline Model (TF-IDF)

This approach serves as a simple and fast baseline, relying solely on the textual data from the catalog_content column.

Methodology

Text Preprocessing: The catalog_content text is cleaned to remove stop words and punctuation.

Feature Generation (TF-IDF): The cleaned text is converted into a numerical format using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer. This method creates a feature for each unique word, with its value representing the word's importance in a given product's description relative to the entire dataset.

Model Training: A single LightGBM Regressor model is trained on the resulting TF-IDF features to predict the product price.

Advantages & Disadvantages

Pros: Very fast to implement and provides a solid baseline score.

Cons:

Completely ignores the valuable information contained in the product images.

Fails to capture the semantic meaning of words (e.g., it cannot tell that "sofa" and "couch" are similar).

Can create an extremely large number of features, which may be noisy.

Approach 2: Advanced Multi-Modal Model (Image & Text Embeddings)

This is a more sophisticated and powerful approach that combines both visual and textual information to make predictions. This was the primary method used for the final submission.

Methodology

Image Feature Generation:

A pre-trained EfficientNet-B0 model is used to process each product image.

The model generates a 1280-dimension embedding vector for each image, which numerically represents its visual features.

Text Feature Generation:

A pre-trained Sentence Transformer (all-MiniLM-L6-v2) model converts the catalog_content into a 384-dimension embedding vector. This captures the semantic context and meaning of the product description.

Additional features like Item Pack Quantity (IPQ) and text length are also extracted.

Target Transformation:

The price, which is highly skewed, is transformed using np.log1p. The model is trained to predict this normalized value, which significantly improves performance on the SMAPE metric.

Model Training (K-Fold Cross-Validation):

An ensemble of 5 LightGBM Regressor models is trained using 5-Fold Cross-Validation.

The final prediction is the average of the predictions from all 5 models. This makes the result more stable and accurate.

The final averaged predictions are converted back from the log scale to the original price scale using np.expm1.

Advantages & Disadvantages

Pros:

Holistic approach that leverages all available data (text and image).

Captures deep semantic and visual context, leading to much higher accuracy.

The K-Fold and log-transform techniques make the model very robust.

Cons:

The feature generation process is computationally intensive and time-consuming.

Requires downloading and installing large pre-trained models.
