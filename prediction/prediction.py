import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# gcap_data = "/oak/stanford/groups/maggiori/GCAP/data"
gcap_data = "/Volumes/data"


# -----------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------

# ADJUST
target_col = "sanctions_any"  #  "sanctions_any" or "export_controls_any"
quarter_cutoff_start = pd.Period("2021Q4", freq="Q")
year_cutoff_start = 2021
epochs_num = 100
quarter_future_cutoff_start = pd.Period("2024Q4", freq="Q")

# -----------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------

def plot_training_validation_loss(history):
    """
    Plots training and validation loss curves from Keras model fit history.

    Args:
        history: Keras History object returned by model.fit()
    """
    train_loss = history.history["loss"]
    val_loss = history.history.get("val_loss")  # might be None if no validation

    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Training loss")
    if val_loss is not None:
        plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary crossentropy loss")
    plt.title("Training/validation loss over epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

# VERIFY CLASSDISTRIBUTION - TO CHECK MODEL IS NOT ALWAYS PREDICTING 0
def plot_true_vs_predicted_class_distribution(y_true, y_pred, class_labels=[0, 1]):
    """
    Plot the distribution of true vs. predicted class labels.

    Args:
        y_true: Array-like of true labels.
        y_pred: Array-like of predicted labels.
        class_labels: List of class labels to display (default [0, 1]).
    """
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    index = class_labels  # class labels

    plt.figure()
    width = 0.35
    plt.bar([i - width/2 for i in index], true_counts.reindex(index, fill_value=0), width=width, label="True")
    plt.bar([i + width/2 for i in index], pred_counts.reindex(index, fill_value=0), width=width, label="Predicted")

    plt.xticks(index, [str(i) for i in index])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("True vs Predicted class distribution")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()


# -----------------------------------------------------------------------
# IMPORT AND CLEAN DATA
# -----------------------------------------------------------------------   

# DATASET 1: FIRM-QUARTER LEVEL

# path to your file
file_path = f"{gcap_data}/scratch/chiara/cs230/orbis/output/blw_bvdid_exploded_embedding_10_cpu.csv"

# read the CSV
df = pd.read_csv(file_path)   # add options here if needed

# convert sic code to int
df["primary_sic_code"] = df["primary_sic_code"].astype(int)

# clean quarter
df["quarter"] = pd.to_datetime(df["quarter"])
df["quarter"] = df["quarter"].dt.to_period("Q") 
# df["quarter"] = df["quarter"].astype(str)

# SELECT DATA AFTER 2021Q4
df = df[df["quarter"] > quarter_cutoff_start].copy()

# SELECT USA AND CHINA companies only (to reduce zeroes)
df = df[df["country_iso"].isin(["USA", "CHN"])].copy()

# PRINT SHARE OF SANCTIONED FIRMS
print("Mean export_controls_any:", df["export_controls_any"].mean())
print("Mean sanctions_any:", df["sanctions_any"].mean())

# DATASET 2: FIRM-YEAR LEVEL

group_cols = ["factset_entity_id", "year"]

# find embedding columns: either rep_0..rep_63 or 0..63
rep_cols = [f"rep_{i}" for i in range(64) if f"rep_{i}" in df.columns]
if not rep_cols:
    rep_cols = [str(i) for i in range(64) if str(i) in df.columns]

# columns where we want the first non-group value
first_cols = [
    "country_iso",
    "primary_sic_code",
    *rep_cols,
]

# columns where we want the max over the year
max_cols = [
    "sanctions_any",
    "export_controls_any",
]

agg_dict = {
    **{col: "first" for col in first_cols if col in df.columns},
    **{col: "max"   for col in max_cols   if col in df.columns},
}

df_firm_year = df.groupby(group_cols, as_index=False).agg(agg_dict)

df_firm_year = df_firm_year[["factset_entity_id", "year", "country_iso", "primary_sic_code", "sanctions_any", "export_controls_any"] + rep_cols]

# DATASET 3: FIRM LEVEL

group_cols = ["factset_entity_id"]

# find embedding columns: either rep_0..rep_63 or 0..63
rep_cols = [f"rep_{i}" for i in range(64) if f"rep_{i}" in df.columns]
if not rep_cols:
    rep_cols = [str(i) for i in range(64) if str(i) in df.columns]

# columns where we want the first non-group value
first_cols = [
    "country_iso",
    "primary_sic_code",
    *rep_cols,
]

# columns where we want the max over the year
max_cols = [
    "sanctions_any",
    "export_controls_any",
]

agg_dict = {
    **{col: "first" for col in first_cols if col in df.columns},
    **{col: "max"   for col in max_cols   if col in df.columns},
}

df_firm = df.groupby(group_cols, as_index=False).agg(agg_dict)

df_firm = df_firm[["factset_entity_id", "country_iso", "primary_sic_code", "sanctions_any", "export_controls_any"] + rep_cols]

# ----------------------------------------------------------------------------------------------------------------------------------------------
# SMALL NN (3 layers) - COUNTRY, INDUSTRY, AND QUARTER FIXED EFFECTS - REDEFINED LOSS with sanctions/export_controls at t or t+n
# ----------------------------------------------------------------------------------------------------------------------------------------------   

# -------------------------------------------------
# 1. Choose target and feature columns
# -------------------------------------------------

# numeric columns 0–63 (handles int or string column names)
num_0_63 = []
for i in range(64):
    if i in df.columns:
        num_0_63.append(i)
    if str(i) in df.columns:
        num_0_63.append(str(i))
# remove any accidental duplicates
num_0_63 = list(dict.fromkeys(num_0_63))

numeric_features = num_0_63
categorical_features = ["primary_sic_code", "country_iso", "quarter"]

feature_cols = numeric_features + categorical_features

# Keep only rows where we have everything we need
df_model = df.dropna(subset=feature_cols + [target_col]).copy()

# -------------------------------------------------
# 2. Generate new true label - "ever_sanctioned_before_C"
# -------------------------------------------------

# we want to use a different loss function, that penalizes the cases where the model predicts 1, but the firm is never sanctioned at t or t+n 
# this implied that we only consider it a wrong prediction if the model predicts 1 and the firm NEVER gets sanctioned at that time or in the future
# the intuition is that we want our model to learn to predict not only firms that are currently sanctioned, but also firms that will be sanctioned in the future
# the easiest implementation is to generate a new label, that is 1 if the firm was ever sanctioned at t or t+n, and 0 otherwise
# we can then use this new label to train our model

# generate a new label, that is 1 if the firm was ever sanctioned at t or t+n, and 0 otherwise
# Set flag to 1 if company is currently sanctioned or will be sanctioned at any future period; otherwise 0
def sanctioned_current_or_future(arr):
    # arr is a pandas Series; reverse so any future 1 will set all previous entries to 1
    return (arr[::-1].cummax()[::-1]).astype(int)

df_model["ever_sanctioned_before_C"] = (
    df_model.groupby("factset_entity_id")[target_col]
    .transform(sanctioned_current_or_future)
)

# useful check for what the new label looks like
# df_model[df_model["factset_entity_id"] == "0N67X5-E"]

# -------------------------------------------------
# 2. Train / test split
# -------------------------------------------------

X = df_model[feature_cols]
y = df_model["ever_sanctioned_before_C"].astype("float32") # this is the new label


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# quantify the problem of imbalanced classes
print(y_train.value_counts())
print("\nShare of each class:")
print(y_train.value_counts(normalize=True))


# -------------------------------------------------
# 3. Preprocessing: scale nums, one-hot encode cats
# -------------------------------------------------
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False  # make it dense so Keras is happy
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)

input_dim = X_train_proc.shape[1]
print("Final number of features going into Keras:", input_dim)

# -------------------------------------------------
# 4. Keras model
# -------------------------------------------------
model_mlp = Sequential([
    Dense(64, activation="relu",
          input_shape=(input_dim,),
          kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(32, activation="relu",
          kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model_mlp.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,           # how many epochs of no improvement before stopping
    restore_best_weights=True
)

# compute weights for classes 0 and 1 based on y_train
classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

# turn weights into a dict
class_weight = {0: weights[0], 1: weights[1]}
print(class_weight)

history = model_mlp.fit(
    X_train_proc, y_train,
    epochs=epochs_num*3,                # upper bound
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    # callbacks=[early_stop],
    verbose=1
)

# -------------------------------------------------
# 5. Evaluate on test set
# -------------------------------------------------
test_loss, test_acc = model_mlp.evaluate(X_test_proc, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")

# Predicted probabilities and classes
y_proba = model_mlp.predict(X_test_proc).ravel()     # probabilities in [0,1]
y_pred = (y_proba >= 0.5).astype(int)           # 0/1 predictions


## -------------------------------------------------
# 6. Plot training/validation loss
# -------------------------------------------------

plot_training_validation_loss(history)


## -------------------------------------------------
# 7. Plot true vs predicted class distribution
# -------------------------------------------------

plot_true_vs_predicted_class_distribution(y_test, y_pred)


## -------------------------------------------------
# 8. BUILD RESULTING DATAFRAME with PREDICTIONS
# -------------------------------------------------
df_test = df_model.loc[X_test.index].copy()
df_test["y_proba"] = y_proba   # from model_mlp.predict(...)
df_test["y_pred"] = y_pred     # 0/1 from thresholding y_proba

#-------------------------------------------------
# 9. Compute confusion matrix-like values
# -------------------------------------------------

# True Positive (TP): Model predicts 1 and ever_sanctioned_before_C is 1
tp = ((df_test["y_pred"] == 1) & (df_test["ever_sanctioned_before_C"] == 1)).sum()

# False Positive (FP): Model predicts 1 but ever_sanctioned_before_C is 0
fp = ((df_test["y_pred"] == 1) & (df_test["ever_sanctioned_before_C"] == 0)).sum()

# True Negative (TN): Model predicts 0 and ever_sanctioned_before_C is 0
tn = ((df_test["y_pred"] == 0) & (df_test["ever_sanctioned_before_C"] == 0)).sum()

# False Negative (FN): Model predicts 0 but ever_sanctioned_before_C is 1
fn = ((df_test["y_pred"] == 0) & (df_test["ever_sanctioned_before_C"] == 1)).sum()

print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")

# Compute confusion matrix-like values conditional on y_pred == 1
tp_cond = ((df_test["y_pred"] == 1) & (df_test["ever_sanctioned_before_C"] == 1)).sum()
fp_cond = ((df_test["y_pred"] == 1) & (df_test["ever_sanctioned_before_C"] == 0)).sum()
total_pred_1 = (df_test["y_pred"] == 1).sum()

print("\nAmong cases where y_pred == 1:")
print(f"  True Positives (TP): {tp_cond}")
print(f"  False Positives (FP): {fp_cond}")
print(f'Total predicted 1: {total_pred_1}')
print(f"  Proportion TP: {tp_cond/total_pred_1 if total_pred_1>0 else float('nan'):.2f}")
print(f"  Proportion FP: {fp_cond/total_pred_1 if total_pred_1>0 else float('nan'):.2f}")


# Compute confusion matrix-like values conditional on y_true == 1
tp_cond_true = ((df_test["ever_sanctioned_before_C"] == 1) & (df_test["y_pred"] == 1)).sum()
fn_cond_true = ((df_test["ever_sanctioned_before_C"] == 1) & (df_test["y_pred"] == 0)).sum()
total_true_1 = (df_test["ever_sanctioned_before_C"] == 1).sum()

print("\nAmong cases where ever_sanctioned_before_C == 1 (true label is 1):")
print(f"  True Positives (TP): {tp_cond_true}")
print(f"  False Negatives (FN): {fn_cond_true}")
print(f'Total true 1: {total_true_1}')
print(f"  Proportion TP: {tp_cond_true/total_true_1 if total_true_1>0 else float('nan'):.2f}")
print(f"  Proportion FN: {fn_cond_true/total_true_1 if total_true_1>0 else float('nan'):.2f}")

#### SEE ALL OUR ALTERNATIVE SPECIFICATIONS (with less successful results) BELOW

# -----------------------------------------------------------------------
# LOGISTIC REGRESSION
# -----------------------------------------------------------------------   

# -------------------------------------------------
# Choose target and feature columns
# -------------------------------------------------

# numeric columns 0–63 (handles int or string column names)
num_0_63 = []
for i in range(64):
    if i in df.columns:
        num_0_63.append(i)
    if str(i) in df.columns:
        num_0_63.append(str(i))
# remove any accidental duplicates
num_0_63 = list(dict.fromkeys(num_0_63))

numeric_features = num_0_63
categorical_features = ["country_iso", "primary_sic_code", "quarter"]

feature_cols = numeric_features + categorical_features

# Keep only rows where we have everything we need
df_model = df.dropna(subset=feature_cols + [target_col]).copy()

X = df_model[feature_cols]
y = df_model[target_col].astype("float32")

# -------------------------------------------------
# 2. Train / test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 3. Preprocessing: scale nums, one-hot encode cats
# -------------------------------------------------
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False  # make it dense so Keras is happy
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)

input_dim = X_train_proc.shape[1]
print("Final number of features going into Keras:", input_dim)

# -------------------------------------------------
# 4. Keras "logistic regression" model
# -------------------------------------------------
model_lg = Sequential([
    Dense(1, activation="sigmoid", input_shape=(input_dim,))
])

model_lg.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,           # how many epochs of no improvement before stopping
    restore_best_weights=True
)

history = model_lg.fit(
    X_train_proc, y_train,
    epochs=30,            # upper bound, probably won't reach this
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------------------------
# 5. Evaluate on test set
# -------------------------------------------------
test_loss, test_acc = model_lg.evaluate(X_test_proc, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")

# Predicted probabilities and classes
y_proba = model_lg.predict(X_test_proc).ravel()     # probabilities in [0,1]
y_pred = (y_proba >= 0.5).astype(int)           # 0/1 predictions

# -------------------------------------------------
# 6. Plot loss
# -------------------------------------------------

plot_training_validation_loss(history)

# -------------------------------------------------
# 7. Visualize the model is always predicting 0
# -------------------------------------------------

plot_true_vs_predicted_class_distribution(y_test, y_pred)


# -----------------------------------------------------------------------
# LOGISTIC REGRESSION WITH CLASS WEIGHTS - COUNTRY, INDUSTRY, AND QUARTER FIXED EFFECTS
# -----------------------------------------------------------------------   

# -------------------------------------------------
# Choose target and feature columns
# -------------------------------------------------

# numeric columns 0–63 (handles int or string column names)
num_0_63 = []
for i in range(64):
    if i in df.columns:
        num_0_63.append(i)
    if str(i) in df.columns:
        num_0_63.append(str(i))
# remove any accidental duplicates
num_0_63 = list(dict.fromkeys(num_0_63))

numeric_features = num_0_63
categorical_features = ["primary_sic_code", "quarter", "country_iso"] # NO TIME or COUNTRY FIXED EFFECTS

feature_cols = numeric_features + categorical_features

# Keep only rows where we have everything we need
df_model = df.dropna(subset=feature_cols + [target_col]).copy()

X = df_model[feature_cols]
y = df_model[target_col].astype("float32")

# -------------------------------------------------
# 2. Train / test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# quantify the problem of imbalanced classes in the FIRM-LEVEL DATA
print(y_train.value_counts())
print("\nShare of each class:")
print(y_train.value_counts(normalize=True))


# -------------------------------------------------
# 3. Preprocessing: scale nums, one-hot encode cats
# -------------------------------------------------
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False  # make it dense so Keras is happy
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)

input_dim = X_train_proc.shape[1]
print("Final number of features going into Keras:", input_dim)

# -------------------------------------------------
# 4. Keras "logistic regression" model
# -------------------------------------------------
model_lg = Sequential([
    Dense(1, activation="sigmoid", input_shape=(input_dim,))
])

model_lg.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,           # how many epochs of no improvement before stopping
    restore_best_weights=True
)

# compute weights for classes 0 and 1 based on y_train
classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

# turn weights into a dict
class_weight = {0: weights[0], 1: weights[1]}
print(class_weight)

history = model_lg.fit(
    X_train_proc, y_train,
    epochs=epochs_num,           
    batch_size=32,
    validation_split=0.2,
    # callbacks=[early_stop],
    class_weight=class_weight, 
    verbose=1
)

# -------------------------------------------------
# 5. Evaluate on test set
# -------------------------------------------------
test_loss, test_acc = model_lg.evaluate(X_test_proc, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")

# Predicted probabilities and classes
y_proba = model_lg.predict(X_test_proc).ravel()     # probabilities in [0,1]
y_pred = (y_proba >= 0.5).astype(int)           # 0/1 predictions

# -------------------------------------------------
# 6. Plot loss
# -------------------------------------------------

plot_training_validation_loss(history)

# -------------------------------------------------
# 7. Visualize the model is always predicting 0
# -------------------------------------------------

plot_true_vs_predicted_class_distribution(y_test, y_pred)

# -----------------------------------------------------------------------
# LOGISTIC REGRESSION WITH CLASS WEIGHTS - COUNTRY, INDUSTRY, AND YEAR FIXED EFFECTS
# -----------------------------------------------------------------------   

# -------------------------------------------------
# Choose target and feature columns
# -------------------------------------------------

# numeric columns 0–63 (handles int or string column names)
num_0_63 = []
for i in range(64):
    if i in df.columns:
        num_0_63.append(i)
    if str(i) in df.columns:
        num_0_63.append(str(i))
# remove any accidental duplicates
num_0_63 = list(dict.fromkeys(num_0_63))

numeric_features = num_0_63
categorical_features = ["primary_sic_code", "year", "country_iso"] # NO TIME or COUNTRY FIXED EFFECTS

feature_cols = numeric_features + categorical_features

# Keep only rows where we have everything we need
df_model = df_firm_year.dropna(subset=feature_cols + [target_col]).copy()

X = df_model[feature_cols]
y = df_model[target_col].astype("float32")

# -------------------------------------------------
# 2. Train / test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# quantify the problem of imbalanced classes in the FIRM-LEVEL DATA
print(y_train.value_counts())
print("\nShare of each class:")
print(y_train.value_counts(normalize=True))


# -------------------------------------------------
# 3. Preprocessing: scale nums, one-hot encode cats
# -------------------------------------------------
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False  # make it dense so Keras is happy
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)

input_dim = X_train_proc.shape[1]
print("Final number of features going into Keras:", input_dim)

# -------------------------------------------------
# 4. Keras "logistic regression" model
# -------------------------------------------------
model_lg = Sequential([
    Dense(1, activation="sigmoid", input_shape=(input_dim,))
])

model_lg.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,           # how many epochs of no improvement before stopping
    restore_best_weights=True
)

# compute weights for classes 0 and 1 based on y_train
classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

# turn weights into a dict
class_weight = {0: weights[0], 1: weights[1]}
print(class_weight)

history = model_lg.fit(
    X_train_proc, y_train,
    epochs=epochs_num,            # upper bound, probably won't reach this
    batch_size=32,
    validation_split=0.2,
    # callbacks=[early_stop],
    class_weight=class_weight, 
    verbose=1
)

# -------------------------------------------------
# 5. Evaluate on test set
# -------------------------------------------------
test_loss, test_acc = model_lg.evaluate(X_test_proc, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")

# Predicted probabilities and classes
y_proba = model_lg.predict(X_test_proc).ravel()     # probabilities in [0,1]
y_pred = (y_proba >= 0.5).astype(int)           # 0/1 predictions

# -------------------------------------------------
# 6. Plot loss
# -------------------------------------------------

plot_training_validation_loss(history)

# -------------------------------------------------
# 7. Visualize the model is always predicting 0
# -------------------------------------------------

plot_true_vs_predicted_class_distribution(y_test, y_pred)

# -----------------------------------------------------------------------
# LOGISTIC REGRESSION WITH CLASS WEIGHTS - FIRM LEVEL
# -----------------------------------------------------------------------   

# -------------------------------------------------
# Choose target and feature columns
# -------------------------------------------------

# numeric columns 0–63 (handles int or string column names)
num_0_63 = []
for i in range(64):
    if i in df.columns:
        num_0_63.append(i)
    if str(i) in df.columns:
        num_0_63.append(str(i))
# remove any accidental duplicates
num_0_63 = list(dict.fromkeys(num_0_63))

numeric_features = num_0_63
categorical_features = ["primary_sic_code"] # NO TIME or COUNTRY FIXED EFFECTS

feature_cols = numeric_features + categorical_features

# Keep only rows where we have everything we need
df_model = df_firm.dropna(subset=feature_cols + [target_col]).copy()

X = df_model[feature_cols]
y = df_model[target_col].astype("float32")

# -------------------------------------------------
# 2. Train / test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# quantify the problem of imbalanced classes in the FIRM-LEVEL DATA
print(y_train.value_counts())
print("\nShare of each class:")
print(y_train.value_counts(normalize=True))


# -------------------------------------------------
# 3. Preprocessing: scale nums, one-hot encode cats
# -------------------------------------------------
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False  # make it dense so Keras is happy
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)

input_dim = X_train_proc.shape[1]
print("Final number of features going into Keras:", input_dim)

# -------------------------------------------------
# 4. Keras "logistic regression" model
# -------------------------------------------------
model_lg = Sequential([
    Dense(1, activation="sigmoid", input_shape=(input_dim,))
])

model_lg.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,           # how many epochs of no improvement before stopping
    restore_best_weights=True
)

# compute weights for classes 0 and 1 based on y_train
classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

# turn weights into a dict
class_weight = {0: weights[0], 1: weights[1]}
print(class_weight)

history = model_lg.fit(
    X_train_proc, y_train,
    epochs=epochs_num,            # upper bound, probably won't reach this
    batch_size=32,
    validation_split=0.2,
    # callbacks=[early_stop],
    class_weight=class_weight, 
    verbose=1
)

# -------------------------------------------------
# 5. Evaluate on test set
# -------------------------------------------------
test_loss, test_acc = model_lg.evaluate(X_test_proc, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")

# Predicted probabilities and classes
y_proba = model_lg.predict(X_test_proc).ravel()     # probabilities in [0,1]
y_pred = (y_proba >= 0.5).astype(int)           # 0/1 predictions

# -------------------------------------------------
# 6. Plot loss
# -------------------------------------------------

plot_training_validation_loss(history)

# -------------------------------------------------
# 7. Visualize the model is always predicting 0
# -------------------------------------------------

plot_true_vs_predicted_class_distribution(y_test, y_pred)



# -----------------------------------------------------------------------
# SMALL NN (3 layers) - COUNTRY, INDUSTRY, AND QUARTER FIXED EFFECTS
# -----------------------------------------------------------------------   

# -------------------------------------------------
# Choose target and feature columns
# -------------------------------------------------

# numeric columns 0–63 (handles int or string column names)
num_0_63 = []
for i in range(64):
    if i in df.columns:
        num_0_63.append(i)
    if str(i) in df.columns:
        num_0_63.append(str(i))
# remove any accidental duplicates
num_0_63 = list(dict.fromkeys(num_0_63))

numeric_features = num_0_63
categorical_features = ["primary_sic_code", "quarter", "country_iso"]

feature_cols = numeric_features + categorical_features

# Keep only rows where we have everything we need
df_model = df.dropna(subset=feature_cols + [target_col]).copy()

X = df_model[feature_cols]
y = df_model[target_col].astype("float32")

# -------------------------------------------------
# 2. Train / test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# quantify the problem of imbalanced classes in the FIRM-LEVEL DATA
print(y_train.value_counts())
print("\nShare of each class:")
print(y_train.value_counts(normalize=True))


# -------------------------------------------------
# 3. Preprocessing: scale nums, one-hot encode cats
# -------------------------------------------------
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False  # make it dense so Keras is happy
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)

input_dim = X_train_proc.shape[1]
print("Final number of features going into Keras:", input_dim)

# -------------------------------------------------
# 4. Keras model
# -------------------------------------------------
model_mlp = Sequential([
    Dense(64, activation="relu",
          input_shape=(input_dim,),
          kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(32, activation="relu",
          kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model_mlp.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,           # how many epochs of no improvement before stopping
    restore_best_weights=True
)

# compute weights for classes 0 and 1 based on y_train
classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

# turn weights into a dict
class_weight = {0: weights[0], 1: weights[1]}
print(class_weight)

history = model_mlp.fit(
    X_train_proc, y_train,
    epochs=epochs_num,                # upper bound
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    # callbacks=[early_stop],
    verbose=1
)

# -------------------------------------------------
# 5. Evaluate on test set
# -------------------------------------------------
test_loss, test_acc = model_mlp.evaluate(X_test_proc, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")

# Predicted probabilities and classes
y_proba = model_mlp.predict(X_test_proc).ravel()     # probabilities in [0,1]
y_pred = (y_proba >= 0.5).astype(int)           # 0/1 predictions

# -------------------------------------------------
# 6. Plot loss
# -------------------------------------------------

plot_training_validation_loss(history)

# -------------------------------------------------
# 7. Visualize the model is always predicting 0
# -------------------------------------------------

plot_true_vs_predicted_class_distribution(y_test, y_pred)


# -----------------------------------------------------------------------
# SMALL NN (3 layers) - COUNTRY, INDUSTRY, AND YEAR FIXED EFFECTS
# -----------------------------------------------------------------------   

# -------------------------------------------------
# Choose target and feature columns
# -------------------------------------------------

# numeric columns 0–63 (handles int or string column names)
num_0_63 = []
for i in range(64):
    if i in df.columns:
        num_0_63.append(i)
    if str(i) in df.columns:
        num_0_63.append(str(i))
# remove any accidental duplicates
num_0_63 = list(dict.fromkeys(num_0_63))

numeric_features = num_0_63
categorical_features = ["primary_sic_code", "country_iso", "year"]

feature_cols = numeric_features + categorical_features

# Keep only rows where we have everything we need
df_model = df_firm_year.dropna(subset=feature_cols + [target_col]).copy()

X = df_model[feature_cols]
y = df_model[target_col].astype("float32")

# -------------------------------------------------
# 2. Train / test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# quantify the problem of imbalanced classes in the FIRM-LEVEL DATA
print(y_train.value_counts())
print("\nShare of each class:")
print(y_train.value_counts(normalize=True))


# -------------------------------------------------
# 3. Preprocessing: scale nums, one-hot encode cats
# -------------------------------------------------
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False  # make it dense so Keras is happy
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)

input_dim = X_train_proc.shape[1]
print("Final number of features going into Keras:", input_dim)

# -------------------------------------------------
# 4. Keras model
# -------------------------------------------------
model_mlp = Sequential([
    Dense(64, activation="relu",
          input_shape=(input_dim,),
          kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(32, activation="relu",
          kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model_mlp.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,           # how many epochs of no improvement before stopping
    restore_best_weights=True
)

# compute weights for classes 0 and 1 based on y_train
classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

# turn weights into a dict
class_weight = {0: weights[0], 1: weights[1]}
print(class_weight)

history = model_mlp.fit(
    X_train_proc, y_train,
    epochs=epochs_num*2,                # upper bound
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    # callbacks=[early_stop],
    verbose=1
)

# -------------------------------------------------
# 5. Evaluate on test set
# -------------------------------------------------
test_loss, test_acc = model_mlp.evaluate(X_test_proc, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")

# Predicted probabilities and classes
y_proba = model_mlp.predict(X_test_proc).ravel()     # probabilities in [0,1]
y_pred = (y_proba >= 0.5).astype(int)           # 0/1 predictions


## -------------------------------------------------
# 6. Plot training/validation loss
# -------------------------------------------------

plot_training_validation_loss(history)


## -------------------------------------------------
# 7. Plot true vs predicted class distribution
# -------------------------------------------------

plot_true_vs_predicted_class_distribution(y_test, y_pred)



# -----------------------------------------------------------------------
# SMALL NN (3 layers) - FIRM LEVEL
# -----------------------------------------------------------------------   

# -------------------------------------------------
# Choose target and feature columns
# -------------------------------------------------

# numeric columns 0–63 (handles int or string column names)
num_0_63 = []
for i in range(64):
    if i in df.columns:
        num_0_63.append(i)
    if str(i) in df.columns:
        num_0_63.append(str(i))
# remove any accidental duplicates
num_0_63 = list(dict.fromkeys(num_0_63))

numeric_features = num_0_63
categorical_features = ["primary_sic_code"]

feature_cols = numeric_features + categorical_features

# Keep only rows where we have everything we need
df_model = df_firm.dropna(subset=feature_cols + [target_col]).copy()

X = df_model[feature_cols]
y = df_model[target_col].astype("float32")

# -------------------------------------------------
# 2. Train / test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# quantify the problem of imbalanced classes in the FIRM-LEVEL DATA
print(y_train.value_counts())
print("\nShare of each class:")
print(y_train.value_counts(normalize=True))


# -------------------------------------------------
# 3. Preprocessing: scale nums, one-hot encode cats
# -------------------------------------------------
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False  # make it dense so Keras is happy
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)

input_dim = X_train_proc.shape[1]
print("Final number of features going into Keras:", input_dim)

# -------------------------------------------------
# 4. Keras model
# -------------------------------------------------
model_mlp = Sequential([
    Dense(64, activation="relu",
          input_shape=(input_dim,),
          kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(32, activation="relu",
          kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model_mlp.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,           # how many epochs of no improvement before stopping
    restore_best_weights=True
)

# compute weights for classes 0 and 1 based on y_train
classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

# turn weights into a dict
class_weight = {0: weights[0], 1: weights[1]}
print(class_weight)

history = model_mlp.fit(
    X_train_proc, y_train,
    epochs=epochs_num*2,                # upper bound
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    # callbacks=[early_stop],
    verbose=1
)

# -------------------------------------------------
# 5. Evaluate on test set
# -------------------------------------------------
test_loss, test_acc = model_mlp.evaluate(X_test_proc, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")

# Predicted probabilities and classes
y_proba = model_mlp.predict(X_test_proc).ravel()     # probabilities in [0,1]
y_pred = (y_proba >= 0.5).astype(int)           # 0/1 predictions


## -------------------------------------------------
# 6. Plot training/validation loss
# -------------------------------------------------

plot_training_validation_loss(history)

## -------------------------------------------------
# 7. Plot true vs predicted class distribution
# -------------------------------------------------

plot_true_vs_predicted_class_distribution(y_test, y_pred)
