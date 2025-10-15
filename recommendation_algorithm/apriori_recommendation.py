import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from collections import defaultdict, Counter

# ------------------------------
# Step 1: Hardcoded dataset
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
# Step 2: One-hot encode
# ------------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# ------------------------------
# Step 3: Apriori
# ------------------------------
min_support = max(1 / len(transactions), 0.1)  # dynamically based on dataset size
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# ------------------------------
# Step 4: Association Rules
# ------------------------------
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules = rules.sort_values(by="confidence", ascending=False)

# ------------------------------
# Step 5: Build Recommendation Dictionary
# ------------------------------
recommendation_dict = defaultdict(list)

for _, row in rules.iterrows():
    antecedents = [x.lower().strip() for x in row["antecedents"]]
    consequents = [x.lower().strip() for x in row["consequents"]]

    for item in antecedents:
        recommendation_dict[item].append({
            "consequents": consequents,
            "confidence": row["confidence"],
            "lift": row["lift"]
        })

# ------------------------------
# Step 6: Recommendation Function
# ------------------------------
def recommend(Item_Name, top_n=3, sort_by="confidence"):
    item = Item_Name.strip().lower()

    # Fallback for items without rules: most frequent co-occurring items
    if item not in recommendation_dict or not recommendation_dict[item]:
        co_occurrences = []
        for t in transactions:
            if Item_Name in t:
                co_occurrences.extend([i for i in t if i != Item_Name])
        if not co_occurrences:
            return f"No recommendation found for '{Item_Name}'."
        top_items = [x for x, _ in Counter(co_occurrences).most_common(top_n)]
        return f"If someone orders '{Item_Name}', recommend:\n" + "\n".join(top_items)

    recs = recommendation_dict[item]
    recs_sorted = sorted(recs, key=lambda x: x[sort_by], reverse=True)

    # Remove duplicates and keep best score
    unique_recs = {}
    for r in recs_sorted:
        for c in r["consequents"]:
            if c not in unique_recs or r[sort_by] > unique_recs[c][sort_by]:
                unique_recs[c] = r

    final_recs = list(unique_recs.items())[:top_n]
    result = [f"{c}" for c, r in final_recs]

    return f"If someone orders '{Item_Name}', recommend:\n" + "\n".join(result)

# ------------------------------
# Step 7: Test Recommendations
# ------------------------------
test_items = ["Bacon Cheese", "Chef Burger", "Freestyle Soda", "Chocolate Shake"]
for item in test_items:
    print("\n" + recommend(item, top_n=2))
