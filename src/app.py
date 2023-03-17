import streamlit as st
import pandas as pd
import os
import pickle


# PAGE CONFIG : Must be the first line after the importation section
st.set_page_config(
    page_title="[Demo] Iris Classification App", page_icon="💐", layout="centered")

# Setup variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
tmp_df_fp = os.path.join(DIRPATH, "assets", "tmp", "history.csv")
ml_core_fp = os.path.join(DIRPATH, "assets", "ml", "ml_components.pkl")
init_df = pd.DataFrame(
    {"petal length (cm)": [], "petal width (cm)": [],
     "sepal length (cm)": [], "sepal width (cm)": [], }
)

# FUNCTIONS


@st.cache_resource()  # stop the hot-reload to the function just bellow
def load_ml_components(fp):
    "Load the ml component to re-use in app"
    with open(fp, "rb") as f:
        object = pickle.load(f)
    return object


def convert_df(df):
    "Convert a dataframe so that it will be downloadable"
    return df.to_csv(index=False).encode('utf-8')


def setup(fp):
    "Setup the required elements like files, models, global variables, etc"

    # history frame
    if not os.path.exists(fp):
        df_history = init_df.copy()
    else:
        df_history = pd.read_csv(fp)

    df_history.to_csv(fp, index=False)

    return df_history


# Setup execution
ml_components_dict = load_ml_components(fp=ml_core_fp)
labels = ml_components_dict['labels']
end2end_pipeline = ml_components_dict['pipeline']
print(f"\n[Info] ML components loaded: {list(ml_components_dict.keys())}")
print(f"\n[Info] Predictable labels: {labels}")
idx_to_labels = {i: l for (i, l) in enumerate(labels)}
print(f"\n[Info] Indexes to labels: {idx_to_labels}")

try:
    df_history
except:
    df_history = setup(tmp_df_fp)

# APP Interface
st.image(
    "https://www.thespruce.com/thmb/GXt55Sf9RIzADYAG5zue1hXtlqc=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/iris-flowers-plant-profile-5120188-01-04a464ab8523426fab852b55d3bb04f0.jpg",
)
# Title
st.title("💐 [Demo] Iris Classification App")

# Sidebar
st.sidebar.write(f"Description")
st.sidebar.write(
    f"This app shows a simple demo of a Streamlit app for Iris flowers classification.")

# Main page

# Form
form = st.form(key="information", clear_on_submit=True)
with form:

    cols = st.columns((1, 1))
    # petal_length = cols[0].slider("What's the petal length?: ", 0.0, 10.0, 1.0)

    df_input = pd.DataFrame(
        {"petal length (cm)": [cols[0].slider("What's the petal length? :", 0.0, 10.0, 1.0)],
         "petal width (cm)": [cols[1].slider("What's the petal width? :", 0.0, 10.0, 1.0)],
         "sepal length (cm)": [cols[0].slider("What's the sepal length? :", 0.0, 10.0, 1.0)],
         "sepal width (cm)": [cols[1].slider("What's the sepal width? :", 0.0, 10.0, 1.0)], }
    )
    print(f"\n[Info] Input information as dataframe: \n{df_input.to_string()}")

    submitted = st.form_submit_button(label="Submit")

    if submitted:
        try:
            st.success("Thanks!")
            st.balloons()
            # # Prediction of just the labels...!
            # prediction_output = end2end_pipeline.predict(df_input)
            # print(
            #     f"[Info] Prediction output (of type '{type(prediction_output)}') from passed input: {prediction_output}")
            # df_input['pred_label'] = prediction_output
            # df_input['pred_label'] = df_input['pred_label'].replace(
            #     idx_to_labels)
            # Prediction of just the labels and confident scores...!
            prediction_output = end2end_pipeline.predict_proba(df_input)
            print(
                f"[Info] Prediction output (of type '{type(prediction_output)}') from passed input: {prediction_output} of shape {prediction_output.shape}")
            predicted_idx = prediction_output.argmax(axis=-1)
            print(f"[Info] Predicted indexes: {predicted_idx}")
            df_input['pred_label'] = predicted_idx
            predicted_labels = df_input['pred_label'].replace(idx_to_labels)
            df_input['pred_label'] = predicted_labels
            predicted_score = prediction_output[:, predicted_idx]
            df_input['confidence_score'] = predicted_score
            df_history = pd.concat([df_history, df_input],
                                   ignore_index=True).convert_dtypes()
            df_history.to_csv(tmp_df_fp, index=False)

        except Exception as e:
            st.error(
                "Oops something went wrong, contact the client service or the admin!")
            print(
                f"\n[Error] {e} \n")


# Expander
expander = st.expander("Check the history")
with expander:

    if submitted:
        st.dataframe(df_history)
        st.download_button(
            "Download this table as CSV",
            convert_df(df_history),
            "prediction_history.csv",
            "text/csv",
            key='download-csv'
        )
