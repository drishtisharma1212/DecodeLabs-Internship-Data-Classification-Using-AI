from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("===== IRIS FLOWER CLASSIFICATION SYSTEM =====")

iris = load_iris()

x = iris.data 
y = iris.target

print("\nDataset Information")
print("Dataset Shape:", x.shape)
print("Total Samples:", len(x))

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== MODEL TRAINED SUCCESSFULLY =====")
print(f"Accuracy: {accuracy * 100: .2f}%")

print("\nFlower Classes:")
print("0 = Setosa")
print("1 = Versicolor")
print("2 = Virginica")

while True:
    print("\n===== FLOWER PREDICTION =====")

    try:
        sepal_length = float(input("Sepal Length: "))
        sepal_width = float(input("Sepal Width: "))
        petal_length = float(input("Petal Length: "))
        petal_width = float(input("Petal Width: "))
    except ValueError:
        print("Please enter valid numeric values only.")
        continue
    
    custom_flower = [[sepal_length, sepal_width, petal_length, petal_width]]
    
    prediction = model.predict(custom_flower)
    
    flower_names = iris.target_names
    
    print("\nPredicted Flower Type:")
    print(flower_names[prediction][0])

    choice = input("\nDo you want to predict another flower? (yes/no): ")

    if choice.lower() == "no":
        print("\nProgram Ended.")
        break