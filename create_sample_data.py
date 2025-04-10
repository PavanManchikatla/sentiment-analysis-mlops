import pandas as pd

# Create a simple sentiment dataset
data = {
    'text': [
        "This product is amazing, I love it!",
        "I'm very disappointed with the quality.",
        "The service was excellent and prompt.",
        "This is the worst experience I've ever had.",
        "The team was helpful and responsive.",
        "I can't recommend this enough, it's perfect.",
        "The product broke after one week.",
        "I'm not satisfied with my purchase.",
        "Great value for money, highly recommend.",
        "The customer support was terrible."
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 
                 'positive', 'positive', 'negative', 'negative', 
                 'positive', 'negative']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/raw/sample_sentiments.csv', index=False)
print("Sample data created at data/raw/sample_sentiments.csv")