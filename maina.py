import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

frame = pd.read_csv("C:\\Users\\waqas\\OneDrive\\Desktop\\tested.csv")
print(frame.head())
frame = frame.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)
frame = pd.get_dummies(frame, columns=['Sex', 'Pclass'])
frame['Age'].fillna(frame['Age'].median(), inplace=True)
frame['Fare'].fillna(frame['Fare'].median(), inplace=True)
X = frame.drop('Survived', axis=1)
y = frame['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)