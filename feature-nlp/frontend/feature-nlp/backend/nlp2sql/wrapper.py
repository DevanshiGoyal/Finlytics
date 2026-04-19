"""
Local NLP2SQL wrapper.

This module keeps everything local:
- Model loading: reads keyword/column mapping from model_config.json
- Preprocessing: normalizes and tokenizes the user question
- Inference: converts intent to a safe read-only SQL query

No external APIs are called.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Read-only safety checks.
_FORBIDDEN_SQL_PATTERNS = [
    re.compile(r"\bDROP\b", re.IGNORECASE),
    re.compile(r"\bDELETE\b", re.IGNORECASE),
    re.compile(r"\bTRUNCATE\b", re.IGNORECASE),
    re.compile(r"\bINSERT\b", re.IGNORECASE),
    re.compile(r"\bUPDATE\b", re.IGNORECASE),
    re.compile(r"\bALTER\b", re.IGNORECASE),
    re.compile(r"\bCREATE\b", re.IGNORECASE),
    re.compile(r"\bMERGE\b", re.IGNORECASE),
    re.compile(r"\bEXEC\b", re.IGNORECASE),
    re.compile(r"\bCALL\b", re.IGNORECASE),
    re.compile(r"\bGRANT\b", re.IGNORECASE),
    re.compile(r"\bREVOKE\b", re.IGNORECASE),
    re.compile(r"--"),
    re.compile(r"/\*"),
]

_AGGREGATION_KEYWORDS = {
    "count": ["count", "how many", "number of", "total records"],
    "avg": ["average", "avg", "mean"],
    "sum": ["sum", "total", "overall"],
    "max": ["max", "maximum", "highest", "largest", "top"],
    "min": ["min", "minimum", "lowest", "smallest", "bottom"],
}

_GROUP_BY_PATTERN = re.compile(
    r"\b(?:by|per)\s+([a-zA-Z_ ]+?)(?:\bwhere\b|\bwith\b|\btop\b|\blimit\b|$)"
)
_LIMIT_PATTERN = re.compile(r"\b(?:top|first|last|limit)\s+(\d+)\b")
_TOP_GROUP_BY_PATTERN = re.compile(r"\btop\s+\d+\s+([a-zA-Z_ ]+?)\s+by\b")

_STOPWORD_TOKENS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "to",
    "for",
    "of",
    "in",
    "on",
    "with",
    "from",
    "show",
    "list",
    "give",
    "records",
    "record",
    "data",
    "all",
    "top",
    "by",
    "per",
}

_SEMANTIC_COLUMN_HINTS = {
    "amount": [
        "amount",
        "amt",
        "value",
        "revenue",
        "sales",
        "deposit",
        "balance",
        "price",
        "cost",
        "transaction amount",
    ],
    "category": [
        "category",
        "type",
        "segment",
        "group",
        "class",
        "merchant",
        "channel",
    ],
    "status": [
        "status",
        "state",
        "stage",
        "condition",
        "flag",
    ],
    "customer_id": [
        "customer",
        "client",
        "user",
        "account",
        "member",
        "customer id",
        "client id",
    ],
    "created_at": [
        "date",
        "time",
        "timestamp",
        "created",
        "updated",
        "month",
        "year",
        "day",
        "transaction date",
        "transaction time",
    ],
}


class LocalNLP2SQL:
    """Simple local NLP2SQL engine using rule-based inference."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        schema_columns: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.model = self.load_local_model(config_path)
        self.table_name: str = self.model["table_name"]
        self.default_limit: int = self.model["default_limit"]
        self.default_numeric_column: str = self.model["default_numeric_column"]
        self.alias_to_column: Dict[str, str] = self.model["alias_to_column"]
        self.column_types: Dict[str, str] = self.model["column_types"]
        self.numeric_columns: List[str] = self.model["numeric_columns"]

        # If dataset schema is provided, adapt aliases/types to real uploaded columns.
        if schema_columns:
            self._adapt_to_schema(schema_columns)

    def load_local_model(self, config_path: Optional[str] = None) -> Dict[str, object]:
        """Load local NLP2SQL config from JSON (no network calls)."""
        default_path = Path(__file__).resolve().parent / "model_config.json"
        path = Path(config_path) if config_path else default_path

        try:
            with path.open("r", encoding="utf-8") as fp:
                config = json.load(fp)
        except FileNotFoundError as exc:
            raise ValueError(f"NLP2SQL config not found at: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"NLP2SQL config is not valid JSON: {path}") from exc

        columns = config.get("columns", [])
        if not columns:
            raise ValueError("NLP2SQL config must include at least one column definition.")

        alias_to_column: Dict[str, str] = {}
        column_types: Dict[str, str] = {}
        numeric_columns: List[str] = []

        for column in columns:
            name = str(column.get("name", "")).strip()
            if not name:
                continue

            col_type = str(column.get("type", "text")).strip().lower()
            aliases = [str(a).strip().lower() for a in column.get("aliases", []) if str(a).strip()]

            column_types[name] = col_type
            alias_to_column[name.lower()] = name

            for alias in aliases:
                alias_to_column[alias] = name

            if col_type == "numeric":
                numeric_columns.append(name)

        if not alias_to_column:
            raise ValueError("NLP2SQL config produced no usable aliases.")

        model = {
            "table_name": str(config.get("table_name", "data")).strip() or "data",
            "default_limit": int(config.get("default_limit", 100)),
            "default_numeric_column": str(config.get("default_numeric_column", "amount")).strip() or "amount",
            "alias_to_column": alias_to_column,
            "column_types": column_types,
            "numeric_columns": numeric_columns,
        }

        logger.info("Loaded local NLP2SQL model from %s", path)
        return model

    def preprocess_query(self, query: str) -> Tuple[str, List[str]]:
        """Normalize user query for deterministic parsing."""
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        cleaned = query.strip().lower()
        if not cleaned:
            raise ValueError("Query cannot be empty.")

        cleaned = re.sub(r"[^a-z0-9_\s<>=.\-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        tokens = cleaned.split(" ")
        return cleaned, tokens

    def run_inference(self, cleaned_query: str, tokens: List[str]) -> Dict[str, object]:
        """Infer SQL intent from a normalized query."""
        del tokens

        aggregation = self._infer_aggregation(cleaned_query)
        mentioned_columns = self._find_columns_in_text(cleaned_query)
        group_by_column = self._infer_group_by(cleaned_query)
        top_n, sort_direction = self._infer_limit(cleaned_query)
        where_clauses = self._infer_where_clauses(cleaned_query)

        target_column = self._infer_target_column(aggregation, mentioned_columns)

        return {
            "aggregation": aggregation,
            "target_column": target_column,
            "group_by_column": group_by_column,
            "top_n": top_n,
            "sort_direction": sort_direction,
            "where_clauses": where_clauses,
        }

    def generate_sql(self, natural_language_query: str) -> str:
        """Main wrapper: natural language input -> SQL output."""
        cleaned_query, tokens = self.preprocess_query(natural_language_query)
        plan = self.run_inference(cleaned_query, tokens)
        sql = self._build_sql(plan)
        return self.validate_sql(sql)

    def _adapt_to_schema(self, schema_columns: List[Dict[str, Any]]) -> None:
        """Map semantic aliases to the real uploaded dataset columns."""
        schema_names: List[str] = []
        schema_types: Dict[str, str] = {}

        for entry in schema_columns:
            raw_name = str(entry.get("name", "")).strip()
            if not raw_name:
                continue

            dtype = self._normalize_column_type(str(entry.get("type", "text")))
            schema_names.append(raw_name)
            schema_types[raw_name] = dtype

        if not schema_names:
            return

        alias_map: Dict[str, str] = {}

        for column in schema_names:
            self._add_alias(alias_map, column, column)

            normalized = self._normalize_alias(column)
            if normalized != column.lower():
                self._add_alias(alias_map, normalized, column)

            for token in normalized.split(" "):
                if len(token) >= 3 and token not in _STOPWORD_TOKENS:
                    self._add_alias(alias_map, token, column)

        for semantic_key, hints in _SEMANTIC_COLUMN_HINTS.items():
            preferred_type = None
            if semantic_key == "amount":
                preferred_type = "numeric"
            elif semantic_key == "created_at":
                preferred_type = "date"

            matched_column = self._best_matching_column(
                schema_names,
                schema_types,
                hints,
                preferred_type=preferred_type,
            )
            if not matched_column:
                continue

            self._add_alias(alias_map, semantic_key, matched_column)
            for hint in hints:
                self._add_alias(alias_map, hint, matched_column)

        self.alias_to_column = alias_map
        self.column_types = schema_types
        self.numeric_columns = [
            col for col in schema_names if schema_types.get(col) == "numeric"
        ]

        if self.default_numeric_column not in self.column_types:
            if self.numeric_columns:
                self.default_numeric_column = self.numeric_columns[0]
            else:
                self.default_numeric_column = ""

    def validate_sql(self, sql: str) -> str:
        """Allow only safe read-only SQL."""
        stripped = sql.strip()
        if not stripped:
            raise ValueError("Generated SQL is empty.")

        if not re.match(r"^\s*(SELECT|WITH)\b", stripped, flags=re.IGNORECASE):
            raise ValueError("Only read-only SELECT queries are allowed.")

        if ";" in stripped[:-1]:
            raise ValueError("Multiple SQL statements are not allowed.")

        for pattern in _FORBIDDEN_SQL_PATTERNS:
            if pattern.search(stripped):
                raise ValueError("Generated SQL failed safety validation.")

        return stripped

    def _infer_aggregation(self, cleaned_query: str) -> str:
        for agg, keywords in _AGGREGATION_KEYWORDS.items():
            if any(keyword in cleaned_query for keyword in keywords):
                return agg
        return "select"

    def _infer_target_column(self, aggregation: str, mentioned_columns: List[str]) -> str:
        if aggregation == "count":
            return "*"

        if aggregation != "select":
            numeric_mentions = [c for c in mentioned_columns if c in self.numeric_columns]
            if numeric_mentions:
                return numeric_mentions[0]
            if self.default_numeric_column in self.column_types:
                return self.default_numeric_column
            if self.numeric_columns:
                return self.numeric_columns[0]

        if mentioned_columns:
            return mentioned_columns[0]

        return "*"

    def _find_columns_in_text(self, text: str) -> List[str]:
        matched: List[str] = []
        for alias in sorted(self.alias_to_column.keys(), key=len, reverse=True):
            column = self.alias_to_column[alias]
            if re.search(rf"\b{re.escape(alias)}\b", text) and column not in matched:
                matched.append(column)
        return matched

    def _infer_group_by(self, cleaned_query: str) -> Optional[str]:
        top_match = _TOP_GROUP_BY_PATTERN.search(cleaned_query)
        if top_match:
            resolved = self._resolve_phrase_to_column(top_match.group(1))
            if resolved:
                return resolved

        match = _GROUP_BY_PATTERN.search(cleaned_query)
        if not match:
            return None

        return self._resolve_phrase_to_column(match.group(1))

    def _resolve_phrase_to_column(self, phrase: str) -> Optional[str]:
        phrase = self._normalize_alias(phrase)
        words = phrase.split()

        if phrase in self.alias_to_column:
            return self.alias_to_column[phrase]

        for width in range(min(4, len(words)), 0, -1):
            candidate = " ".join(words[:width]).strip()
            column = self.alias_to_column.get(candidate)
            if column:
                return column

        # Singularize simple plurals (e.g., categories -> category).
        singular_words: List[str] = []
        for word in words:
            if word.endswith("ies") and len(word) > 3:
                singular_words.append(word[:-3] + "y")
            elif word.endswith("s") and len(word) > 3:
                singular_words.append(word[:-1])
            else:
                singular_words.append(word)

        for width in range(min(4, len(singular_words)), 0, -1):
            candidate = " ".join(singular_words[:width]).strip()
            column = self.alias_to_column.get(candidate)
            if column:
                return column

        return None

    def _normalize_column_type(self, dtype: str) -> str:
        dt = dtype.lower()
        if any(token in dt for token in ["int", "float", "double", "decimal", "real", "numeric"]):
            return "numeric"
        if any(token in dt for token in ["date", "time", "timestamp"]):
            return "date"
        if "bool" in dt:
            return "boolean"
        return "text"

    def _normalize_alias(self, text: str) -> str:
        # Split camelCase / PascalCase before token normalization.
        cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        cleaned = cleaned.lower().strip()
        cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _add_alias(self, mapping: Dict[str, str], alias: str, column: str) -> None:
        normalized = self._normalize_alias(alias)
        if not normalized:
            return

        # Keep first mapping to avoid unstable collisions.
        mapping.setdefault(normalized, column)

        plural = self._pluralize(normalized)
        if plural and plural != normalized:
            mapping.setdefault(plural, column)

    def _pluralize(self, alias: str) -> str:
        if " " in alias:
            return alias
        if alias.endswith("y") and len(alias) > 2:
            return alias[:-1] + "ies"
        if alias.endswith("s"):
            return alias
        return alias + "s"

    def _best_matching_column(
        self,
        schema_names: List[str],
        schema_types: Dict[str, str],
        hints: List[str],
        preferred_type: Optional[str] = None,
    ) -> Optional[str]:
        best_column: Optional[str] = None
        best_score = 0

        normalized_hints = [self._normalize_alias(h) for h in hints]

        for column in schema_names:
            normalized_column = self._normalize_alias(column)
            score = 0

            for hint in normalized_hints:
                if hint == normalized_column:
                    score = max(score, 12)
                elif " " in hint and hint in normalized_column:
                    score = max(score, 10)
                elif hint in normalized_column.split(" "):
                    score = max(score, 8)
                elif hint and hint in normalized_column:
                    score = max(score, 6)

            if preferred_type:
                if schema_types.get(column) == preferred_type:
                    score += 3
                else:
                    score -= 1

            if score > best_score:
                best_score = score
                best_column = column

        return best_column if best_score > 0 else None

    def _infer_limit(self, cleaned_query: str) -> Tuple[Optional[int], str]:
        match = _LIMIT_PATTERN.search(cleaned_query)
        if not match:
            return None, "DESC"

        limit = int(match.group(1))
        sort_direction = "ASC" if any(word in cleaned_query for word in ["lowest", "smallest", "bottom"]) else "DESC"
        return max(1, min(limit, 1000)), sort_direction

    def _infer_where_clauses(self, cleaned_query: str) -> List[str]:
        clauses: List[str] = []

        for alias, column in self.alias_to_column.items():
            col_ref = f'"{column}"'

            between_match = re.search(
                rf"\b{re.escape(alias)}\b\s+between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)",
                cleaned_query,
            )
            if between_match:
                start_val = between_match.group(1)
                end_val = between_match.group(2)
                clauses.append(f"{col_ref} BETWEEN {start_val} AND {end_val}")
                continue

            comparisons = [
                (r"(?:>=|greater than or equal to|at least)", ">="),
                (r"(?:<=|less than or equal to|at most)", "<="),
                (r"(?:>|greater than|more than|above)", ">"),
                (r"(?:<|less than|below|under)", "<"),
            ]

            found_numeric = False
            for pattern_text, operator in comparisons:
                cmp_match = re.search(
                    rf"\b{re.escape(alias)}\b\s*{pattern_text}\s*(-?\d+(?:\.\d+)?)",
                    cleaned_query,
                )
                if cmp_match:
                    clauses.append(f"{col_ref} {operator} {cmp_match.group(1)}")
                    found_numeric = True
                    break

            if found_numeric:
                continue

            eq_match = re.search(
                rf"\b{re.escape(alias)}\b\s*(?:=|is|equals)\s+([a-z0-9_\- ]+?)(?:\s+and\b|\s+or\b|\s+by\b|\s+per\b|\s+top\b|\s+limit\b|$)",
                cleaned_query,
            )
            if eq_match:
                raw_value = eq_match.group(1).strip()
                if raw_value:
                    clauses.append(f"{col_ref} = {self._to_sql_literal(column, raw_value)}")

        deduped: List[str] = []
        for clause in clauses:
            if clause not in deduped:
                deduped.append(clause)
        return deduped

    def _to_sql_literal(self, column: str, raw_value: str) -> str:
        col_type = self.column_types.get(column, "text")

        if col_type == "numeric":
            if re.fullmatch(r"-?\d+(?:\.\d+)?", raw_value):
                return raw_value
            raise ValueError(f"Expected a numeric value for column '{column}'.")

        escaped = raw_value.replace("'", "''")
        return f"'{escaped}'"

    def _build_sql(self, plan: Dict[str, object]) -> str:
        aggregation = str(plan["aggregation"])
        target_column = str(plan["target_column"])
        group_by_column = plan.get("group_by_column")
        top_n = plan.get("top_n")
        sort_direction = str(plan.get("sort_direction", "DESC"))
        where_clauses = list(plan.get("where_clauses", []))

        from_clause = f'FROM "{self.table_name}"'
        where_clause = ""
        if where_clauses:
            where_clause = " WHERE " + " AND ".join(where_clauses)

        group_clause = ""
        order_clause = ""
        limit_clause = ""

        if aggregation == "count":
            if group_by_column:
                select_clause = f'SELECT "{group_by_column}", COUNT(*) AS value'
                group_clause = f' GROUP BY "{group_by_column}"'
                if top_n:
                    order_clause = f" ORDER BY value {sort_direction}"
                    limit_clause = f" LIMIT {int(top_n)}"
            else:
                select_clause = "SELECT COUNT(*) AS value"
        elif aggregation in {"avg", "sum", "max", "min"}:
            agg_map = {
                "avg": "AVG",
                "sum": "SUM",
                "max": "MAX",
                "min": "MIN",
            }
            if target_column == "*":
                raise ValueError("Unable to infer a target column for aggregation.")

            if group_by_column:
                select_clause = (
                    f'SELECT "{group_by_column}", '
                    f'{agg_map[aggregation]}("{target_column}") AS value'
                )
                group_clause = f' GROUP BY "{group_by_column}"'
                if top_n:
                    order_clause = f" ORDER BY value {sort_direction}"
                    limit_clause = f" LIMIT {int(top_n)}"
            else:
                select_clause = f'SELECT {agg_map[aggregation]}("{target_column}") AS value'
        else:
            if target_column != "*":
                select_clause = f'SELECT "{target_column}"'
            else:
                select_clause = "SELECT *"

            if top_n:
                sort_column = target_column
                if sort_column == "*":
                    sort_column = self.default_numeric_column
                if sort_column in self.column_types:
                    order_clause = f' ORDER BY "{sort_column}" {sort_direction}'
                limit_clause = f" LIMIT {int(top_n)}"
            else:
                limit_clause = f" LIMIT {self.default_limit}"

        sql = f"{select_clause} {from_clause}{where_clause}{group_clause}{order_clause}{limit_clause};"
        return sql


_ENGINE: Optional[LocalNLP2SQL] = None


def generate_sql_query(
    query: str,
    schema_columns: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Public reusable function.

    Input: natural language query (string)
    Output: SQL query (string)
    """
    # For per-dataset calls, build a schema-aware engine to avoid stale aliases.
    if schema_columns:
        return LocalNLP2SQL(schema_columns=schema_columns).generate_sql(query)

    global _ENGINE
    if _ENGINE is None:
        _ENGINE = LocalNLP2SQL()

    return _ENGINE.generate_sql(query)
