# Retrieval ranking set

`ranking.json` preserves the existing 60-question local set previously stored only
under ignored `data/eval/`. It contains 45 Feature and 15 CaseStudy questions generated
from source passages. Gold ids refer to the Supabase corpus; they must be checked if
documents are re-ingested. This is a development ranking set, not a human-reviewed
confidence benchmark or a held-out end-to-end test.

The set has no unanswerable queries and no exhaustive candidate relevance labels.
Use the production scorer to collect candidates, then review them and add hard
negative queries before threshold calibration. See [the workflow](../../docs/retrieval-quality.md).

Reports may contain document content. Keep scored/reviewed reports in an appropriate
private location; only source questions and ids are versioned here.
