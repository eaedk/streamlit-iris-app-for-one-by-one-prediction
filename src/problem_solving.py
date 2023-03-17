# Imports

import pickle
import os
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn import datasets
from subprocess import call

# PATHS
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_fp = os.path.join(DIRPATH, "assets", "ml", "ml_components.pkl")
req_fp = os.path.join(DIRPATH, "assets", "ml", "requirements.txt")
eda_report_fp = os.path.join(DIRPATH, "assets", "ml", "eda-report.html")

# import some data to play with
iris = datasets.load_iris(return_X_y=False, as_frame=True)

df = iris['frame']
target_col = 'target'
# pandas profiling
profile = ProfileReport(df, title="Dataset", html={
                        'style': {'full_width': True}})
profile.to_file(eda_report_fp)

# Dataset Splitting
# Please specify
to_ignore_cols = [
    "ID",  # ID
    "Id", "id",
    target_col
]


num_cols = list(set(df.select_dtypes('number')) - set(to_ignore_cols))
cat_cols = list(set(df.select_dtypes(exclude='number')) - set(to_ignore_cols))
print(f"\n[Info] The '{len(num_cols)}' numeric columns are : {num_cols}\nThe '{len(cat_cols)}' categorical columns are : {cat_cols}")

X, y = df.iloc[:, :-1], df.iloc[:, -1].values


X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

print(
    f"\n[Info] Dataset splitted : (X_train , y_train) = {(X_train.shape , y_train.shape)}, (X_eval y_eval) = {(X_eval.shape , y_eval.shape)}. \n")

y_train

# Modeling

# Imputers
num_imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")
cat_imputer = SimpleImputer(
    strategy="most_frequent").set_output(transform="pandas")

# Scaler & Encoder
if len(cat_cols) > 0:
    df_imputed_stacked_cat = cat_imputer.fit_transform(
        df
        .append(df)
        .append(df)
        [cat_cols])
    cat_ = OneHotEncoder(sparse=False, drop="first").fit(
        df_imputed_stacked_cat).categories_
else:
    cat_ = 'auto'

encoder = OneHotEncoder(categories=cat_, sparse=False, drop="first")
scaler = StandardScaler().set_output(transform="pandas")


# feature pipelines
num_pipe = Pipeline(steps=[("num_imputer", num_imputer), ("scaler", scaler)])
cat_pipe = Pipeline(steps=[("cat_imputer", cat_imputer), ("encoder", encoder)])

# end2end features preprocessor

transformers = []

transformers.append(("numerical", num_pipe, num_cols)) if len(
    num_cols) > 0 else None
transformers.append(("categorical", cat_pipe, cat_cols,)) if len(
    cat_cols) > 0 else None
#  ("date", date_pipe, date_cols,),

preprocessor = ColumnTransformer(
    transformers=transformers).set_output(transform="pandas")

print(
    f"\n[Info] Features Transformer : {transformers}. \n")


# end2end pipeline
end2end_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=10))
]).set_output(transform="pandas")

# Training
print(
    f"\n[Info] Training.\n[Info] X_train : columns( {X_train.columns.tolist()}), shape: {X_train.shape} .\n")

end2end_pipeline.fit(X_train, y_train)

# Evaluation
print(
    f"\n[Info] Evaluation.\n")
y_eval_pred = end2end_pipeline.predict(X_eval)

print(classification_report(y_eval, y_eval_pred,
      target_names=iris['target_names']))

# ConfusionMatrixDisplay.from_predictions(
#     y_eval, y_eval_pred, display_labels=iris['target_names'])

# Exportation
print(
    f"\n[Info] Exportation.\n")
to_export = {
    "labels": iris['target_names'],
    "pipeline": end2end_pipeline,
}


# save components to file
with open(ml_fp, 'wb') as file:
    pickle.dump(to_export, file)

# Requirements
# ! pip freeze > requirements.txt
call(f"pip freeze > {req_fp}", shell=True)
