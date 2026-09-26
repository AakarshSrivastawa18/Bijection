#!/usr/bin/env bash
# Assemble the final submission archive:
#
#   bijection_submission.zip
#   ├── output/{matching_results,candidate_pairs}.tsv
#   ├── code/business_entity_resolution/{src,tests,README.md,requirements.txt}
#   └── Documentation_template.md
#
# Run from the project root. Validates the outputs first and refuses to package
# a submission that would be rejected.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TEAM="bijection"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

DATASET_DIR="${BIJECTION_DATA_DIR:?set BIJECTION_DATA_DIR to the dataset directory}"
VALIDATOR="${VALIDATOR:-$DATASET_DIR/../utils/validate_submission.py}"

echo "==> validating outputs"
if [[ -f "$VALIDATOR" ]]; then
    python3 "$VALIDATOR" \
        --matching "$ROOT/output/matching_results.tsv" \
        --candidate "$ROOT/output/candidate_pairs.tsv" \
        --test-dir "$DATASET_DIR/test"
else
    echo "    WARNING: validator not found at $VALIDATOR - skipping" >&2
fi

echo "==> staging"
mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution"
cp "$ROOT/output/matching_results.tsv" "$ROOT/output/candidate_pairs.tsv" "$STAGE/output/"
cp "$ROOT/Documentation_template.md" "$STAGE/"

SRC="$ROOT/code/business_entity_resolution"
cp -R "$SRC/src" "$SRC/tests" "$STAGE/code/business_entity_resolution/"
cp "$SRC/README.md" "$SRC/requirements.txt" "$STAGE/code/business_entity_resolution/"

# strip caches - they bloat the archive and are not reproducible artefacts
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true
find "$STAGE" -name '.DS_Store' -delete 2>/dev/null || true

echo "==> zipping"
OUT="$ROOT/${TEAM}_submission.zip"
rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" .)

echo "==> done"
ls -lh "$OUT"
unzip -l "$OUT" | tail -25
