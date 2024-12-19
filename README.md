# Phishing Analysis
## **Introduction**
This project focuses on email classification, similar to sentiment analysis, where text is categorized based on emotional tones. The objective is to classify emails as either safe or phishing. To achieve this, two distinct approaches are explored: converting email text into term frequency vectors using TfidfVectorizer or CountVectorizer, and leveraging the transformer-based BERT model for training and classification.

## **Installation**
This project is run on Kaggle instead of Google Colab. Before running the code, ensure that the following libraries are pre-installed:

- **Torch**: For deep learning support and model training.
- **Transformers**: For utilizing pre-trained models and tokenizers.
- **Sklearn**: For various machine learning utilities and metrics.
- **Tqdm**: For displaying progress bars in loops.
- **Pandas**: For data manipulation and analysis.

To install these libraries, you can use the following command:

```python
!pip install torch transformers scikit-learn tqdm pandas
```

## **Data Sources**
The email and labels for this project were downloaded from [Phishing Email Detection](https://www.kaggle.com/datasets/subhajournal/phishingemails).
The dataset includes 18600 emails where 61% are safe and 39% are phishing.

## Training Results

The training process yielded impressive outcomes:

- **Logistic Regression**: Trained on term frequency vectors, achieving an **F1 Score of 98%**.  
- **BERT**: Further improved performance with an **F1 Score of 99%**.

## Future Work

- Explore **bert-finetuned-phishing** and leverage transfer learning instead of using **bert-base-uncased**.  
- Experiment with classifiers beyond Logistic Regression, such as **XGBoost**, **Random Forest**, or **MLP**, which might yield better results.  
