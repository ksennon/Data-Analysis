# Data Analysis & Machine Learning

University coursework (Igor Sikorsky KPI, Applied Mathematics): an end-to-end path from exploratory data analysis and classical ML to neural networks, transformers, and generative models.

**Stack:** Python, pandas, NumPy, scikit-learn, TensorFlow / Keras, Hugging Face Transformers, NLTK, matplotlib, seaborn

## Notebooks

| # | Notebook | Task | Methods | Result |
|---|----------|------|---------|--------|
| 1 | [01_chess_eda_classical_ml.ipynb](01_chess_eda_classical_ml.ipynb) | Predict the winner of online chess games (~20k games) | EDA, correlation analysis, missing-value handling, outlier removal, feature engineering; Decision Tree, AdaBoost, Random Forest, KNN, SVM, GridSearchCV | Best accuracy ~0.67 (AdaBoost / tuned Random Forest) vs 0.62 for a single decision tree |
| 2 | [02_clustering_nlp_text_classification.ipynb](02_clustering_nlp_text_classification.ipynb) | Clustering, text preprocessing, and review-rating classification (Amazon reviews) | PCA & t-SNE visualization, K-Means (incl. image color quantization), tokenization, stop-words, stemming, lemmatization, POS tagging; Bag-of-Words & TF-IDF + Logistic Regression | Accuracy 0.85 (BoW) vs 0.83 (TF-IDF) |
| 3 | [03_neural_networks_mlp_cnn_lstm.ipynb](03_neural_networks_mlp_cnn_lstm.ipynb) | Same tasks with neural networks, compared against the classical baselines | MLP with dropout & early stopping (chess), CNN image classifier (pandas vs bears), embedding + bidirectional LSTM (review sentiment) | CNN reaches high validation accuracy within a few epochs; on small tabular/text data the neural models do not beat Random Forest / TF-IDF, which is analyzed in the notebook |
| 4a | [04a_transformer_translation_en_uk.ipynb](04a_transformer_translation_en_uk.ipynb) | English → Ukrainian machine translation (Multi30k-uk, 29k sentence pairs) | Transformer encoder-decoder implemented in Keras: positional embeddings, multi-head attention, custom text vectorization, greedy decoding | ~20M-parameter model trained from scratch; qualitative translation examples |
| 4b | [04b_huggingface_pipelines_uk.ipynb](04b_huggingface_pipelines_uk.ipynb) | Ukrainian NLP with pretrained models | Hugging Face pipelines: GPT-2 text generation (beam search, temperature), extractive QA, text classification | Comparison of decoding settings |
| 4c | [04c_dcgan_fashion_mnist.ipynb](04c_dcgan_fashion_mnist.ipynb) | Generate clothing images | DCGAN (Conv2DTranspose generator, convolutional discriminator, custom adversarial training loop) on Fashion-MNIST | Generated samples after training |

## Notes
- Datasets (chess games, Amazon reviews, images) are not included in the repository because of their size.
