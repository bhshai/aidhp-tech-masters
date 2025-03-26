import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Sample customer data (replace with real data)
data = {
    'Age': [25, 40, 35, 50],
    'Income': [30000, 80000, 50000, 100000],
    'SpendingHabits': ['Low', 'High', 'Medium', 'High'],
    'PreferredRewards': ['Travel', 'Cashback', 'Travel', 'Shopping'],
    'CreditCardRecommended': ['CardA', 'CardB', 'CardA', 'CardC']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical data
le = LabelEncoder()
df['SpendingHabits'] = le.fit_transform(df['SpendingHabits'])
df['PreferredRewards'] = le.fit_transform(df['PreferredRewards'])
df['CreditCardRecommended'] = le.fit_transform(df['CreditCardRecommended'])

# Features and target
X = df[['Age', 'Income', 'SpendingHabits', 'PreferredRewards']]
y = df['CreditCardRecommended']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Example customer for recommendation
new_customer = [[30, 60000, le.transform(['Medium'])[0], le.transform(['Cashback'])[0]]]
recommended_card = model.predict(new_customer)
print(f"Recommended Credit Card: {le.inverse_transform(recommended_card)[0]}")
