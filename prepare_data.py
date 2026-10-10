import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("students.csv") 

# print(data.isnull().sum())

## clean the data

data["study_hours"] = data["study_hours"].fillna(
    data["study_hours"].median()
)

data["attendance"] = data["attendance"].fillna(
    data["attendance"].median()
)

data["previous_marks"] = data["previous_marks"].fillna(
    data["previous_marks"].median()

)
data["assignments_completed"] = data["assignments_completed"].fillna(
    data["assignments_completed"].median()

)
data["result"] = data["result"].fillna(
    data["result"].median()
)


#separate the data

X= data[[
    "study_hours",
    "attendance",
    "previous_marks",
    "assignments_completed"]]


y= data[["result"]]


## split data into testing and training


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


