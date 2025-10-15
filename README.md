# Market Basket Recommendation System

A collection of market basket recommendation models using **FP-Growth**, **Apriori**, and **XGBoost** for item-to-item recommendations.

## Overview

This project demonstrates multiple approaches to building item recommendation systems using transaction data.  
It includes implementations of **FP-Growth**, **Apriori**, and **XGBoost** to identify relationships between purchased items and recommend complementary products.

**Key Features:**
- Clean, self-contained scripts for each algorithm
- Sample hardcoded datasets included for quick testing
- Demonstrates both association rule mining and machine learning–based recommendation
- Ideal for beginners and ML enthusiasts to understand different recommendation techniques


## Tech Stack

- Python 3.x
- Pandas
- scikit-learn
- mlxtend
- XGBoost

## Usage

### FP-Growth & Apriori
1. Navigate to the respective folder (`fpgrowth/` or `apriori/`)
2. Run the script:
```bash
  python fpgrowth.py
  # or
  python apriori.py

Example Output
If someone orders ['Bacon Cheese'], recommend:
  Fries (confidence: 0.86)
  Chocolate Shake (confidence: 0.72)

If someone orders ['Chef Burger'], recommend:
  Fries (confidence: 0.83)
  Chocolate Shake (confidence: 0.68)

