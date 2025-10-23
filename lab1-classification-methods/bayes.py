import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def load_and_prepare_data(fake_path="./data/Fake.csv", true_path="./data/True.csv"):
    """Load and prepare the dataset with labels."""
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df["label"] = 0
    true_df["label"] = 1
    # Reset index so that the concatenated DataFrame has unique integer indices
    return pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)


def vectorize_text(data, text_column="text"):
    """Convert text data into TF-IDF features."""
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X = vectorizer.fit_transform(data[text_column])
    return X, vectorizer


def train_and_evaluate(X, y, data, random_state=42):
    """Train and evaluate the Naive Bayes model."""
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, data.index, test_size=0.2, random_state=random_state
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred, test_idx


def display_results(model, X_train, y_train, X_test, y_test, y_pred, data, test_idx):
    """Display model performance and sample predictions.

    The function now accepts the trained `model` so it can compute scores
    and predicted probabilities. It also builds a results DataFrame indexed
    by the original data indices so rows align with `y_test` / `y_pred`.
    """
    print("Naive Bayes for Fake News Classification")
    print("Training Accuracy:", model.score(X_train, y_train))
    print("Testing Accuracy:", model.score(X_test, y_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Build a results DataFrame aligned to the original indices
    test_rows = data.loc[test_idx].copy()

    # y_test and y_pred are arrays in the same order as test_idx; use a
    # Series with index=test_idx so they align correctly with test_rows
    test_rows["true_label"] = pd.Series(y_test, index=test_idx)
    test_rows["predicted_label"] = pd.Series(y_pred, index=test_idx)

    # Add predicted probabilities when available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        # probability for the positive class (label 1)
        test_rows["prob_positive"] = pd.Series(probs[:, 1], index=test_idx)

    # Optional: map numeric labels to human-friendly names
    label_map = {0: "fake", 1: "true"}
    test_rows["true_label_name"] = test_rows["true_label"].map(label_map)
    test_rows["predicted_label_name"] = test_rows["predicted_label"].map(label_map)

    # Create a short summary column for console display (no full news text)
    def make_short(s, maxlen=140):
        if not isinstance(s, str):
            return ""
        s_clean = " ".join(s.split())  # collapse whitespace/newlines
        return s_clean if len(s_clean) <= maxlen else s_clean[: maxlen - 3] + "..."

    test_rows["short_text"] = test_rows["text"].apply(make_short)

    # Print a tidy sample of classified test rows without full text
    print("\nSample Predictions (first 10 test rows) — short text only:")
    cols = ["short_text", "true_label_name", "predicted_label_name"]
    if "prob_positive" in test_rows.columns:
        cols.append("prob_positive")
    # use to_string for compact, aligned console output
    print(test_rows[cols].head(10).to_string(index=True))

    # Save full test predictions to CSV for further inspection
    out_path = "./predictions/predictions.csv"
    # include index so original row indices are preserved
    test_rows.to_csv(out_path, index=True)
    print(f"\nFull predictions saved to {out_path} ({len(test_rows)} rows)")


def run_algo():
    # Load and prepare data
    data = load_and_prepare_data()
    X, vectorizer = vectorize_text(data)

    # Train and evaluate
    model, X_train, X_test, y_train, y_test, y_pred, test_idx = train_and_evaluate(
        X, data["label"], data
    )

    # Display results
    display_results(model, X_train, y_train, X_test, y_test, y_pred, data, test_idx)


if __name__ == "__main__":
    run_algo()
