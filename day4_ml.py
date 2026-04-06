import pandas as pd
from sklearn.linear_model import LinearRegression

# load csv
df = pd.read_csv("students_ml.csv")

x = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(x, y)

# read input and validate
while True:
    Hours = float(input("enter study hours (0 to exit): "))

    if Hours == 0:
        break

# pass a numeric 2D array (shape: [1, 1]) to predict
prediction = model.predict([[Hours]])
print(f"expected marks:{prediction[0]:.2f}")


print("predicted marks:", prediction[0])

import matplotlib.pyplot as plt

plt.scatter(x,y)

plt.plot(x, model.predict(x))

plt.xlabel("study Hours")

plt.ylabel("Marks")

plt.title("study Hours vs Marks")
plt.show()

print("model accuracy:",model.score(x,y))