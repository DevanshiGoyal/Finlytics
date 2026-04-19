"""Example usage for local NLP2SQL wrapper."""

from nlp2sql import generate_sql_query


if __name__ == "__main__":
    examples = [
        "show top 5 customers by amount",
        "average amount by category",
        "count records where status is active",
        "sum amount where amount greater than 1000",
    ]

    for text in examples:
        sql = generate_sql_query(text)
        print(f"Input: {text}")
        print(f"SQL:   {sql}\n")
