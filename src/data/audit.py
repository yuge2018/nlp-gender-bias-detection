"""Deterministic counts, exact/normalized conflicts and lexical near duplicates."""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import random
import re
import statistics
import unicodedata

SPLIT_NAMES = ("train", "dev", "test")
PRIME = (1 << 61) - 1


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_for_audit(text):
    # This key is NEVER substituted for the original model input.
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def _reference(split, row):
    return {"split": split, "id": row["id"], "label": row["label"]}


def split_statistics(records):
    result = {}
    for split in SPLIT_NAMES:
        rows = records[split]
        lengths = sorted(len(row["text"]) for row in rows)
        labels = Counter(row["label"] for row in rows)
        result[split] = {
            "n": len(rows), "positive_n": labels[1], "negative_n": labels[0],
            "positive_rate": round(labels[1] / len(rows), 8),
            "empty_n": sum(not row["text"].strip() for row in rows),
            "duplicate_text_excess": len(rows) - len({row["text"] for row in rows}),
            "characters": {"min": lengths[0], "median": statistics.median(lengths), "mean": round(statistics.mean(lengths), 6), "max": lengths[-1], "p95_nearest_rank": lengths[max(0, (95 * len(lengths) + 99) // 100 - 1)], "over_threshold": {str(threshold): sum(length > threshold for length in lengths) for threshold in (128, 256, 512)}},
        }
    return result


def duplicate_audit(records):
    reports, groups = {}, []
    for mode, key_function in (("exact", lambda text: text), ("normalized", normalize_for_audit)):
        index = defaultdict(list)
        for split in SPLIT_NAMES:
            for row in records[split]:
                index[key_function(row["text"])].append(_reference(split, row))
        within = {}
        for split in SPLIT_NAMES:
            counts = [sum(ref["split"] == split for ref in refs) for refs in index.values()]
            within[split] = {"groups": sum(n > 1 for n in counts), "excess_rows": sum(max(n - 1, 0) for n in counts)}
        cross = {}
        for left, right in combinations(SPLIT_NAMES, 2):
            common = [refs for refs in index.values() if {left, right}.issubset({ref["split"] for ref in refs})]
            cross[f"{left}__{right}"] = {"unique_texts": len(common), "left_rows": sum(ref["split"] == left for refs in common for ref in refs), "right_rows": sum(ref["split"] == right for refs in common for ref in refs)}
        conflicts = 0
        for text, refs in sorted(index.items(), key=lambda item: fingerprint(item[0])):
            if len(refs) < 2:
                continue
            conflict = len({ref["label"] for ref in refs}) > 1
            conflicts += conflict
            groups.append({"mode": mode, "text_sha256": fingerprint(text), "n": len(refs), "label_conflict": conflict, "references": refs})
        reports[mode] = {"within_split": within, "cross_split": cross, "conflicting_text_groups": conflicts}
    return reports, groups


def _shingles(text, width):
    return frozenset(text[index:index + width] for index in range(len(text) - width + 1))


def near_duplicate_audit(records, settings):
    width = settings["shingle_chars"]
    threshold = settings["jaccard_threshold"]
    bands, rows_per_band = settings["bands"], settings["rows_per_band"]
    if not (type(width) is int and width > 0 and 0 < threshold <= 1 and type(bands) is int and bands > 0 and type(rows_per_band) is int and rows_per_band > 0):
        raise ValueError("Invalid lexical near-duplicate audit settings")
    references = defaultdict(list)
    for split in SPLIT_NAMES:
        for row in records[split]:
            text = normalize_for_audit(row["text"])
            if len(text) >= max(settings["minimum_chars"], width):
                references[text].append(_reference(split, row))
    texts = sorted(references, key=fingerprint)
    tokens = [_shingles(text, width) for text in texts]
    split_sets = [{ref["split"] for ref in references[text]} for text in texts]
    rng = random.Random(settings["seed"])
    parameters = [(rng.randrange(1, PRIME), rng.randrange(PRIME)) for _ in range(bands * rows_per_band)]
    buckets, candidates = defaultdict(list), set()
    for index, token_set in enumerate(tokens):
        hashes = [int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big") % PRIME for token in token_set]
        signature = [min((a * value + b) % PRIME for value in hashes) for a, b in parameters]
        for band in range(bands):
            key = (band, tuple(signature[band * rows_per_band:(band + 1) * rows_per_band]))
            for previous in buckets[key]:
                if len(split_sets[previous] | split_sets[index]) > 1:
                    candidates.add((previous, index))
            buckets[key].append(index)
    matches = []
    for left, right in sorted(candidates):
        a, b = tokens[left], tokens[right]
        if min(len(a), len(b)) / max(len(a), len(b)) < threshold:
            continue
        similarity = len(a & b) / len(a | b)
        if similarity >= threshold:
            matches.append({"left_normalized_sha256": fingerprint(texts[left]), "right_normalized_sha256": fingerprint(texts[right]), "jaccard": round(similarity, 8), "left_references": references[texts[left]], "right_references": references[texts[right]]})
    summary = {"method": "deterministic_minhash_lsh_then_exact_character_shingle_jaccard", "settings": settings, "eligible_normalized_unique_texts": len(texts), "candidate_pairs_checked": len(candidates), "matched_cross_split_text_pairs": len(matches), "limitations": "Lexical, approximate candidate recall; not exhaustive or semantic equivalence. Exact/normalized duplicates are reported separately. Short texts are excluded only from this near-duplicate search."}
    return summary, matches


def audit_records(records, settings):
    duplicates, groups = duplicate_audit(records)
    near, matches = near_duplicate_audit(records, settings)
    return {"split_statistics": split_statistics(records), "duplicates": duplicates, "near_duplicates": near, "length_note": "Unicode character counts, not tokenizer token counts or measured model truncation rates.", "gender_coverage": "No gender-target metadata exists in the locked input. Keyword counts do not establish target identity or balanced coverage."}, groups, matches
