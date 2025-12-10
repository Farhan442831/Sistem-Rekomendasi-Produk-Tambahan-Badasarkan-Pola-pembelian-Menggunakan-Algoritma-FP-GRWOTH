# utils.py
def recommend(rules, input_products, top_n=5):
    input_set = set(p.strip() for p in input_products)
    results = []

    for antecedent, (consequents, confidence) in rules.items():
        antecedent_set = set(antecedent)
        if antecedent_set.issubset(input_set):
            for c in consequents:
                results.append({
                    "recommended_item": c,
                    "antecedent": ", ".join(antecedent_set),
                    "confidence": round(confidence, 3)
                })

    # Sort by highest confidence
    results = sorted(results, key=lambda x: x['confidence'], reverse=True)

    return results[:top_n]
