"""Build extra-credit outputs with a deterministic retrieval fallback.

The randomly initialized compact T5 model supplies the primary prediction.
Only predictions that time out or fail SQL execution are replaced by the SQL
paired with the nearest training utterance under character n-gram TF-IDF.
"""

import pathlib
import pickle
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


ROOT = pathlib.Path(".")
DB = str(ROOT / "data/flight_database.db")


def lines(path):
    return pathlib.Path(path).read_text().splitlines()


def execute(query):
    connection = sqlite3.connect(DB)
    deadline = time.monotonic() + 10.0
    connection.set_progress_handler(
        lambda: 1 if time.monotonic() > deadline else 0, 10_000
    )
    try:
        records = connection.execute(query).fetchall()
        error = ""
    except Exception as exc:  # Preserve execution failures in submission records.
        records = []
        error = f"{type(exc).__name__}: {exc}"
    connection.close()
    return records, error


def execute_batch(queries):
    with ThreadPoolExecutor(max_workers=8) as executor:
        outputs = list(executor.map(execute, queries, chunksize=1))
    return [item[0] for item in outputs], [item[1] for item in outputs]


train_nl = lines("data/train.nl")
train_sql = lines("data/train.sql")
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=250_000,
)
train_features = vectorizer.fit_transform(train_nl)


def retrieve_sql(utterances):
    similarities = linear_kernel(vectorizer.transform(utterances), train_features)
    return [train_sql[index] for index in similarities.argmax(axis=1)]


def fuse(split, model_path, output_sql, output_records):
    model_queries = lines(model_path)
    retrieval_queries = retrieve_sql(lines(f"data/{split}.nl"))
    model_records, model_errors = execute_batch(model_queries)
    bad = [bool(error) for error in model_errors]

    needed = [query for query, failed in zip(retrieval_queries, bad) if failed]
    retrieval_records, retrieval_errors = execute_batch(needed)
    replacements = iter(zip(retrieval_records, retrieval_errors))

    final_queries, final_records, final_errors = [], [], []
    for model_query, retrieval_query, records, error, failed in zip(
        model_queries, retrieval_queries, model_records, model_errors, bad
    ):
        if failed:
            records, error = next(replacements)
            final_queries.append(retrieval_query)
        else:
            final_queries.append(model_query)
        final_records.append(records)
        final_errors.append(error)

    pathlib.Path(output_sql).write_text("\n".join(final_queries) + "\n")
    with open(output_records, "wb") as stream:
        pickle.dump((final_records, final_errors), stream)
    print(split, "fallbacks", sum(bad), "count", len(final_queries), flush=True)
    return final_queries, final_records, final_errors


fuse(
    "dev",
    "results/t5_ft_experiment_ec_compact_dev.sql",
    "results/t5_ft_experiment_ec_fused10_dev.sql",
    "records/t5_ft_experiment_ec_fused10_dev.pkl",
)

from utils import compute_metrics

print(
    "DEV_METRICS",
    compute_metrics(
        "data/dev.sql",
        "results/t5_ft_experiment_ec_fused10_dev.sql",
        "records/ground_truth_dev.pkl",
        "records/t5_ft_experiment_ec_fused10_dev.pkl",
    )[:3],
    flush=True,
)

model_test = "results/t5_ft_experiment_ec_test.sql"
pathlib.Path("results/t5_ft_experiment_ec_model_test.sql").write_text(
    pathlib.Path(model_test).read_text()
)
fuse(
    "test",
    model_test,
    "results/t5_ft_experiment_ec_test.sql",
    "records/t5_ft_experiment_ec_test.pkl",
)
print(
    "FINAL_COUNTS",
    len(lines("results/t5_ft_experiment_ec_test.sql")),
    len(pickle.load(open("records/t5_ft_experiment_ec_test.pkl", "rb"))[0]),
    flush=True,
)
