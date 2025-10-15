import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# ------------------------------
# Step 1: Hardcoded transaction data
# ------------------------------
transactions = [
    ["Bacon Cheese", "Fries", "Coke"],
    ["Chef Burger", "Fries"],
    ["Freestyle Soda", "Chocolate Shake"],
    ["Bacon Cheese", "Chocolate Shake"],
    ["Chef Burger", "Coke"],
    ["Bacon Cheese", "Fries", "Chocolate Shake"],
    ["Freestyle Soda", "Fries"],
    ["Chef Burger", "Chocolate Shake"],
    ["Bacon Cheese", "Fries", "Coke"],
    ["Chef Burger", "Fries", "Chocolate Shake"],
    ["Freestyle Soda", "Fries", "Chocolate Shake"]
]

# ------------------------------
# Step 2: Encode items
# ------------------------------
mlb = MultiLabelBinarizer()
X_all = mlb.fit_transform(transactions)
item_names = mlb.classes_
df = pd.DataFrame(X_all, columns=item_names)
print("Encoded transaction matrix:\n", df.head())

# ------------------------------
# Step 3: Prepare train data
# ------------------------------
# For each item, we'll train a binary classifier to predict its presence.
train_data = {}
for item in item_names:
    X = df.drop(columns=[item])
    y = df[item]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3
    )
    model.fit(X_train, y_train)
    train_data[item] = (model, X.columns)

print("\n XGBoost models trained for all items!")

# ------------------------------
# Step 4: Recommendation function
# ------------------------------
def recommend(order_items, top_n=3):
    order_vector = np.zeros(len(item_names))
    for i, item in enumerate(item_names):
        if item in order_items:
            order_vector[i] = 1

    # Predict probabilities for all items not in order
    recommendations = {}
    for target_item, (model, features) in train_data.items():
        if target_item not in order_items:
            X_input = pd.DataFrame([order_vector], columns=item_names)
            X_input = X_input[features]
            prob = model.predict_proba(X_input)[0][1]
            recommendations[target_item] = prob

    # Sort and show top recommendations
    top_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\nIf someone orders {order_items}, recommend:")
    for item, score in top_recs:
        print(f"  {item} (confidence: {score:.2f})")

# ------------------------------
# Step 5: Test Recommendations
# ------------------------------
test_orders = [
    ["Bacon Cheese"],
    ["Chef Burger"],
    ["Freestyle Soda"],
    ["Chocolate Shake"]
]

for order in test_orders:
    recommend(order, top_n=2)
