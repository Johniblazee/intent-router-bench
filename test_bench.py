"""Smoke check: fastest model end-to-end on the custom dataset."""

from bench import DATASETS, bench_one


def test_smoke():
    train, test, labels = DATASETS["custom"]()
    assert len(labels) == 10 and len(test) == 50
    row = bench_one("potion-8m", train, test, labels, "custom")
    assert row["accuracy"] > 0.5, f"potion-8m accuracy suspiciously low: {row}"
    print("smoke ok:", row)


if __name__ == "__main__":
    test_smoke()
