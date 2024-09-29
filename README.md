# IMDb Movie Ratings Prediction

This project aims to predict IMDb scores for movies using machine learning models based on various movie-related features such as the director, cast members, critic reviews, and audience reactions. The IMDb score is a widely recognized metric that reflects the average rating given by viewers and critics, providing insights into a movie's reception.

## Project Overview

The project focuses on predicting IMDb movie ratings by employing machine learning models such as Support Vector Classifier (SVC), Random Forest Classifier, and Gradient Boosting Classifier. The dataset contains information about various movies, including their director, cast, critic reviews, and audience responses.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - `numpy` (Numerical computations)
  - `pandas` (Data manipulation and analysis)
  - `matplotlib` & `seaborn` (Data visualization)
  - `plotnine` (Additional data visualization)
  - `scikit-learn` (Machine Learning models)

## Dataset
Link - https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset?resource=download

The dataset used in this project provides various movie features such as:
- **Movie Title**
- **Director**
- **Actors**
- **Critic Reviews**
- **Audience Reactions**
- **IMDb Score**

Data was imported using pandas and is stored in a CSV format, which includes several important movie attributes that were cleaned and preprocessed for the model training.

## Data Preprocessing

The following steps were undertaken to preprocess the data:

1. **Handling Missing Data**: Missing values were dealt with appropriately to ensure the integrity of the dataset.
2. **Encoding Categorical Features**: Categorical features such as the director and actors were encoded into numerical representations.
3. **Feature Scaling**: Numerical features were normalized for model optimization.
4. **Train-Test Split**: The dataset was split into training and testing sets for model evaluation.

## Exploratory Data Analysis (EDA)

EDA was performed to visualize and better understand the dataset. Some key visualizations include:

- **Histograms** for distribution of numerical data
- **Scatter plots** to explore relationships between features such as budget, duration, and IMDb score
- **Correlation heatmaps** to identify significant relationships among features

## Modeling

The following machine learning models were trained on the dataset:

1. **Support Vector Classifier (SVC)**: Classifies IMDb scores into categories.
2. **Random Forest Classifier**: Utilizes an ensemble of decision trees for better accuracy.
3. **Gradient Boosting Classifier**: A boosting technique to improve the model's predictive power.

Hyperparameter tuning was performed using `GridSearchCV` to optimize the model's performance.

## Results

The performance of the models was evaluated using accuracy, precision, recall, and F1-score. The key findings include:

## Conclusion

The IMDb movie rating prediction model demonstrated promising results, achieving an accuracy of 85% using Random Forest Classifier. With further refinement and inclusion of additional features such as genre and review sentiment analysis, the model can be further optimized.

## How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/IMDb-Movie-Ratings-Prediction.git

2. pip install -r requirements.txt

3. jupyter notebook IMDb_Movie_Ratings_Prediction.ipynb

