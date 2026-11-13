import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from utils import compute_metrics, save_queries_and_records


def lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngram_min", type=int, default=2)
    parser.add_argument("--ngram_max", type=int, default=5)
    args = parser.parse_args()

    train_nl, train_sql = lines("data/train.nl"), lines("data/train.sql")
    dev_nl = lines("data/dev.nl")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(args.ngram_min, args.ngram_max),
        min_df=2, sublinear_tf=True, max_features=250000)
    train_x = vectorizer.fit_transform(train_nl)
    dev_x = vectorizer.transform(dev_nl)
    similarities = linear_kernel(dev_x, train_x)
    best = similarities.argmax(axis=1)
    predictions = [train_sql[i] for i in best]
    Path("results").mkdir(exist_ok=True)
    Path("records").mkdir(exist_ok=True)
    save_queries_and_records(
        predictions, "results/retrieval_dev.sql", "records/retrieval_dev.pkl")
    sql_em, record_em, record_f1, errors = compute_metrics(
        "data/dev.sql", "results/retrieval_dev.sql",
        "records/ground_truth_dev.pkl", "records/retrieval_dev.pkl")
    print({"record_f1": record_f1, "record_em": record_em,
           "sql_em": sql_em, "errors": sum(bool(e) for e in errors)})


if __name__ == "__main__":
    main()
