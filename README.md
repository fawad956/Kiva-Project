# Kiva Loan Funding Prediction

A machine learning project analyzing Kiva microfinance loan data to predict whether loans will be successfully funded based on loan characteristics and borrower information.

## Overview

This project analyzes data from Kiva.org, an online crowdfunding platform that provides microfinance services to underserved communities worldwide. Using historical loan data, we build predictive models to determine the likelihood of loan funding success.

### Key Statistics
- **Dataset Size**: 20,872 loan records
- **Geographic Coverage**: 77 countries
- **Total Loans Analyzed**: Over $1 billion in loans to 2+ million borrowers
- **Loan Repayment Rate**: 95.8%

## Dataset Description

The project uses three main datasets:

### 1. Loans Dataset (`loans.csv`)
- **Records**: 20,872 entries with 34 columns
- **Key Variables**: Loan amount, status, activity, sector, location, borrower demographics
- **Target Variable**: `STATUS` (funded/not funded)

### 2. Lenders Dataset (`lenders.csv`)
- **Records**: 15,615 entries with 14 columns
- **Information**: Lender demographics, location, occupation, lending history

### 3. Loan-Lenders Dataset (`loans_lenders.csv`)
- **Records**: 20,342 entries with 2 columns
- **Purpose**: Links loans to their respective lenders

## Data Processing Pipeline

### 1. Data Cleaning
- Remove incomplete loan records (`fundRaising` status)
- Recode loan status to binary classification (funded vs. not funded)
- Handle outliers using 3×IQR rule for loan amounts
- Create log-transformed loan amount variable for better distribution

### 2. Missing Value Treatment
- `CURRENCY_EXCHANGE_COVERAGE_RATE`: Fill with 0 (indicates no coverage)
- `IMAGE_ID`: Convert to binary indicator (1 if image exists, 0 otherwise)
- Remove irrelevant columns with excessive missing values

### 3. Feature Engineering
- `TIME_LENGTH`: Duration between loan posting and funding
- `PREDISBURSE`: Boolean indicating if disbursement occurred before posting
- `logAmount`: Log transformation of loan amount
- `Year`: Extract year from posting date
- `Days`: Convert time length to days

## Model Features

### Predictor Variables
- **Loan Characteristics**: Log amount, lender term, repayment interval
- **Borrower Info**: Original language, sector, country
- **Platform Features**: Image presence, journal entries, bulk entries
- **Timing**: Days to funding, pre-disbursement status
- **Financial**: Currency policy, exchange coverage rate, distribution model

### Target Variable
- `STATUS`: Binary classification (funded vs. not funded)

## Machine Learning Models

### 1. Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
logRegressor = LogisticRegression()
logRegressor.fit(X_train, y_train)
```

### 2. Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

### Model Evaluation
- **Metrics**: Accuracy score, confusion matrix
- **Train/Test Split**: 67%/33% with random state for reproducibility

## Key Findings

### Loan Distribution by Sector
- **Agriculture**: Largest sector by volume
- **Retail**: Second most common sector
- **Services**: Significant funding volume

### Geographic Patterns
- Loan distribution varies significantly by country
- Some regions show higher funding success rates

### Temporal Trends
- Loan volumes fluctuate by year
- Seasonal patterns in funding behavior

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Running the Analysis
1. Clone the repository
2. Place the CSV files in the appropriate directory
3. Open and run the Jupyter notebook
4. Follow the step-by-step analysis pipeline

### File Structure
```
kiva-loan-prediction/
├── loans.csv
├── lenders.csv
├── loans_lenders.csv
├── kiva_analysis.ipynb
└── README.md
```

## Results & Business Impact

### Model Performance
- **Best Model**: Logistic Regression (based on accuracy comparison)
- **Use Case**: Risk assessment for loan approval
- **Business Value**: Improved funding allocation and risk management

### Managerial Implications
- Loan characteristics significantly impact funding success
- Geographic and sectoral patterns inform strategic decisions
- Timing factors affect funding probability

## Technical Requirements

- **Python Version**: 3.7+
- **Key Libraries**:
  - pandas (data manipulation)
  - numpy (numerical operations)
  - matplotlib (visualization)
  - scikit-learn (machine learning)
  - jupyter (notebook environment)

## Data Ethics & Privacy

This analysis uses publicly available Kiva loan data in accordance with Kiva's data sharing policies. No personally identifiable information is used beyond what's publicly available on the Kiva platform.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Model improvements
- Additional feature engineering
- Visualization enhancements
- Code optimization

## License

This project is for educational purposes. Please refer to Kiva's terms of service for data usage guidelines.

## References

- [Kiva.org](https://www.kiva.org) - Official Kiva website
- Kiva API Documentation: http://api.kivaws.org/v1/partners.json
- Small Business Trends: "How to Get a Kiva Loan: A Step by Step Guide"

## Author

**Fawad Khan**  
BCIS 5110 - Data Analytics Project

---

*This project demonstrates the application of machine learning techniques to real-world microfinance data, contributing to financial inclusion research and practice.*
