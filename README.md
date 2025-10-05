🌌 Exoplanet Predictor - Django App

A web application to predict the likelihood of Kepler Objects of Interest (KOIs) being confirmed exoplanets. Built with Django and a stacked ensemble machine learning model trained on cleaned KOI data.

--------------------------------------------
Features
--------------------------------------------

- Predicts the probability of KOIs being confirmed as exoplanets.

- Interactive table of top predicted candidates.

- Visualizations of predicted probabilities.

- Machine learning backend using Random Forest + Gradient Boosting ensemble.

- ROC curve and performance metrics for evaluation.

--------------------------------------------
Requirements
--------------------------------------------

Python 3.10+

Django 4.x

scikit-learn

pandas, numpy, matplotlib, seaborn

TensorFlow 

Joblib (for saving/loading trained models, which are already saved)

Install Python packages via:

pip install -r requirements.txt


--------------------------------------------
Setup Instructions
--------------------------------------------

Clone the repository:

git clone https://github.com/yourusername/exoplanet-predictor.git

cd exoplanet-predictor


Create a virtual environment:

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt


Set up the Django project:

python manage.py migrate


Collect static files:

python manage.py collectstatic


Run the development server:

python manage.py runserver


Open your browser and navigate to:

http://127.0.0.1:8000/
