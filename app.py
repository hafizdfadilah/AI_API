from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Optional
from mangum import Mangum
from feature import new_feature

app = FastAPI()
handler = Mangum(app)

class Item(BaseModel):
    Pclass: int = Field(example=3)
    Name: str = Field(example="Braund, Mr. Owen Harris")
    Sex: str = Field(example="male")
    Age: float = Field(example=22.0)
    SibSp: int = Field(example=1)
    Parch: int = Field(example=0)
    Ticket: str = Field(example="A/5 21171")
    Fare: float = Field(example=7.25)
    Cabin: Optional[str] = Field(example=None)
    Embarked: str = Field(example="S")

@app.post("/predict")
def predict(item: Item):
    print(item)
    data = pd.DataFrame([dict(item)])
    new_feature(data)
    data.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

    loaded_model = joblib.load('best_model.joblib')
    predictions = loaded_model.predict(data)

    prediction_result = {
        "Anak-Anak Perempuan": int(data['WA'].values[0]),
        "Anak-Anak Laki-lai": int(data['MA'].values[0]),
        "Survival Prediction": int(data['Survival'].values[0])
    }

    return prediction_result

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer()),
                                 ('scaler', StandardScaler())]),
         ['Age', 'SibSp', 'Parch', 'Fare', 'umur', 'WA', 'MA', 'Survival']),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                 ('onehot', OneHotEncoder())]),
         ['Pclass', 'Sex', 'Embarked'])
    ]
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


# uvicorn app:app --reload


# reference
# https://www.youtube.com/watch?v=VYk3lwZbHBU&t=814s
