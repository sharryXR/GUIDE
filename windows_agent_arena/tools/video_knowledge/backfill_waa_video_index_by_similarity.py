#!/usr/bin/env python3
"""Backfill WAA video knowledge by matching similar annotated OSWorld tasks.

The canonical WAA video index only reuses OSWorld annotations when the task id
matches. This script handles the remaining tasks conservatively: it compares
missing WAA tasks against OSWorld tasks that already have converted video
planning/grounding, and copies the annotation only when the best candidate is
above a configurable similarity threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_WAA_INDEX = "evaluation_examples_windows/test_all_queries_with_videos_with_converted.json"
DEFAULT_OUTPUT = "evaluation_examples_windows/test_all_queries_with_videos_with_similarity.json"

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "for",
    "from",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "make",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "set",
    "that",
    "the",
    "this",
    "to",
    "up",
    "use",
    "with",
    "would",
    "you",
}

APP_ALIASES = {
    "chrome": {"chrome", "google", "browser", "web"},
    "msedge": {"edge", "microsoftedge", "browser", "web"},
    "file_explorer": {"file", "explorer", "files", "folder", "windows"},
    "libreoffice_calc": {"libreoffice", "calc", "spreadsheet", "excel"},
    "libreoffice_writer": {"libreoffice", "writer", "document", "word"},
    "microsoft_paint": {"paint", "mspaint", "image", "drawing"},
    "notepad": {"notepad", "text", "txt"},
    "settings": {"settings", "windows"},
    "vlc": {"vlc", "media", "video", "player"},
    "vs_code": {"vscode", "code", "visualstudio", "editor"},
    "windows_calc": {"calculator", "calc", "windows"},
    "clock": {"clock", "timer", "alarm", "windows"},
}

APP_FAMILIES = {
    "chrome": "browser",
    "msedge": "browser",
    "file_explorer": "file_manager",
    "libreoffice_calc": "spreadsheet",
    "libreoffice_writer": "document",
    "microsoft_paint": "image_editor",
    "notepad": "text_editor",
    "settings": "system_settings",
    "vlc": "media_player",
    "vs_code": "code_editor",
    "windows_calc": "calculator",
    "clock": "clock",
}


@dataclass(frozen=True)
class MatchScore:
    source: Dict[str, Any]
    score: float
    cosine: float
    sequence: float
    jaccard: float
    app_score: float


def canonical_task_id(task_id: Optional[str]) -> str:
    return re.sub(r"(?i)-wos$", "", (task_id or "").strip())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def has_video(entry: Dict[str, Any]) -> bool:
    return int(entry.get("video_count") or 0) > 0 and (
        bool(entry.get("planning_results")) or bool(entry.get("grounding_results"))
    )


def entry_id(entry: Dict[str, Any]) -> str:
    return str(entry.get("canonical_id") or entry.get("id") or "")


def normalize_app(value: Optional[str]) -> str:
    value = (value or "").lower().strip()
    compact = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    compact_no_sep = compact.replace("_", "")

    if compact in APP_ALIASES:
        return compact
    for app, aliases in APP_ALIASES.items():
        if compact_no_sep == app.replace("_", ""):
            return app
        if compact_no_sep in aliases:
            return app
        if compact in aliases:
            return app
    return compact


def app_terms(entry: Dict[str, Any]) -> set[str]:
    raw_values = [
        str(entry.get("domain") or ""),
        str(entry.get("web") or ""),
    ]
    terms: set[str] = set()
    for raw in raw_values:
        app = normalize_app(raw)
        if app:
            terms.add(app)
            terms.update(APP_ALIASES.get(app, set()))
            family = APP_FAMILIES.get(app)
            if family:
                terms.add(family)
    return terms


def app_similarity(target: Dict[str, Any], source: Dict[str, Any]) -> float:
    target_apps = app_terms(target)
    source_apps = app_terms(source)
    if not target_apps or not source_apps:
        return 0.0
    if target_apps & source_apps:
        return 1.0

    target_families = {
        APP_FAMILIES.get(normalize_app(str(target.get("domain") or ""))),
        APP_FAMILIES.get(normalize_app(str(target.get("web") or ""))),
    }
    source_families = {
        APP_FAMILIES.get(normalize_app(str(source.get("domain") or ""))),
        APP_FAMILIES.get(normalize_app(str(source.get("web") or ""))),
    }
    target_families.discard(None)
    source_families.discard(None)
    if target_families & source_families:
        return 0.6
    return 0.0


def extract_demo_titles(text: str, max_titles: int = 4) -> List[str]:
    if not text:
        return []
    titles: List[str] = []
    patterns = [
        r"The (?:planning|grounding) trajectory of Demo \d+:\s*([^\n:]+)",
        r"\*\*task\*\*:\s*([^\n]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            title = match.group(1).strip(" -*:")
            if title and title not in titles:
                titles.append(title)
            if len(titles) >= max_titles:
                return titles
    return titles


def searchable_text(entry: Dict[str, Any], include_annotation_titles: bool = False) -> str:
    pieces = [
        str(entry.get("domain") or ""),
        str(entry.get("web") or ""),
        str(entry.get("instruction") or ""),
        str(entry.get("query") or ""),
    ]
    if include_annotation_titles:
        pieces.extend(extract_demo_titles(str(entry.get("planning_results") or "")))
        pieces.extend(extract_demo_titles(str(entry.get("grounding_results") or "")))
    return " ".join(piece for piece in pieces if piece)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[_/\\.-]+", " ", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    cleaned: List[str] = []
    for token in tokens:
        if len(token) < 2:
            continue
        if token in STOP_WORDS:
            continue
        if re.fullmatch(r"[0-9a-f]{8,}", token):
            continue
        cleaned.append(token)
    return cleaned


def build_idf(documents: Sequence[Sequence[str]]) -> Dict[str, float]:
    df: Counter[str] = Counter()
    for tokens in documents:
        df.update(set(tokens))
    total_docs = max(len(documents), 1)
    return {
        token: math.log((1 + total_docs) / (1 + freq)) + 1.0
        for token, freq in df.items()
    }


def vectorize(tokens: Sequence[str], idf: Dict[str, float]) -> Dict[str, float]:
    counts = Counter(tokens)
    if not counts:
        return {}
    max_tf = max(counts.values())
    return {
        token: (0.5 + 0.5 * count / max_tf) * idf.get(token, 1.0)
        for token, count in counts.items()
    }


def cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def sequence_ratio(left: str, right: str) -> float:
    # Import lazily to keep startup cheap in report-only use.
    from difflib import SequenceMatcher

    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def score_pair(
    target: Dict[str, Any],
    source: Dict[str, Any],
    target_tokens: Sequence[str],
    source_tokens: Sequence[str],
    target_vector: Dict[str, float],
    source_vector: Dict[str, float],
) -> MatchScore:
    target_instruction = str(target.get("instruction") or "")
    source_instruction = " ".join(
        value
        for value in [
            str(source.get("instruction") or ""),
            str(source.get("query") or ""),
        ]
        if value
    )
    cosine_score = cosine(target_vector, source_vector)
    sequence_score = sequence_ratio(target_instruction, source_instruction)
    jaccard_score = jaccard(target_tokens, source_tokens)
    app_score = app_similarity(target, source)
    final = (
        0.62 * cosine_score
        + 0.16 * sequence_score
        + 0.10 * jaccard_score
        + 0.12 * app_score
    )
    return MatchScore(
        source=source,
        score=min(final, 1.0),
        cosine=cosine_score,
        sequence=sequence_score,
        jaccard=jaccard_score,
        app_score=app_score,
    )


def candidate_summary(match: MatchScore) -> Dict[str, Any]:
    source = match.source
    return {
        "source_id": source.get("id", ""),
        "source_web": source.get("web", ""),
        "source_instruction": source.get("instruction", ""),
        "source_query": source.get("query", ""),
        "score": round(match.score, 4),
        "cosine": round(match.cosine, 4),
        "sequence": round(match.sequence, 4),
        "jaccard": round(match.jaccard, 4),
        "app_score": round(match.app_score, 4),
        "video_count": source.get("video_count", 0),
        "converted_video_count": source.get("converted_video_count", 0),
    }


def should_consider_target(entry: Dict[str, Any], overwrite_existing: bool) -> bool:
    if overwrite_existing:
        return True
    return not has_video(entry)


def copy_annotation(
    target: Dict[str, Any],
    match: MatchScore,
    threshold: float,
    margin: float,
    second_score: float,
) -> Dict[str, Any]:
    source = match.source
    updated = dict(target)
    updated["query"] = source.get("query", "")
    updated["video_count"] = source.get("video_count", 0)
    updated["converted_video_count"] = source.get("converted_video_count", 0)
    updated["planning_results"] = source.get("planning_results", "")
    updated["grounding_results"] = source.get("grounding_results", "")
    updated["source"] = "osworld_similarity"
    updated["similarity_score"] = round(match.score, 4)
    updated["similarity_threshold"] = threshold
    updated["similarity_margin"] = round(match.score - second_score, 4)
    updated["similarity_min_margin"] = margin
    updated["similarity_candidate_id"] = source.get("id", "")
    updated["similarity_candidate_web"] = source.get("web", "")
    updated["similarity_candidate_instruction"] = source.get("instruction", "")
    updated["similarity_candidate_query"] = source.get("query", "")
    updated["similarity_method"] = "tfidf_instruction_query_app_v1"
    return updated


def build_vectors(entries: Sequence[Dict[str, Any]], include_titles: bool) -> Tuple[List[List[str]], List[Dict[str, float]]]:
    tokenized = [
        tokenize(searchable_text(entry, include_annotation_titles=include_titles))
        for entry in entries
    ]
    idf = build_idf(tokenized)
    return tokenized, [vectorize(tokens, idf) for tokens in tokenized]


def best_matches(
    target: Dict[str, Any],
    target_tokens: Sequence[str],
    target_vector: Dict[str, float],
    candidates: Sequence[Dict[str, Any]],
    candidate_tokens: Sequence[Sequence[str]],
    candidate_vectors: Sequence[Dict[str, float]],
    top_k: int,
    require_same_app_family: bool,
) -> List[MatchScore]:
    matches: List[MatchScore] = []
    target_canonical = canonical_task_id(entry_id(target))

    for source, tokens, vector in zip(candidates, candidate_tokens, candidate_vectors):
        source_canonical = canonical_task_id(entry_id(source))
        if target_canonical and target_canonical == source_canonical:
            continue

        match = score_pair(target, source, target_tokens, tokens, target_vector, vector)
        if require_same_app_family and match.app_score <= 0:
            continue
        matches.append(match)

    matches.sort(key=lambda item: item.score, reverse=True)
    return matches[:top_k]


def backfill(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    waa_entries = load_json(Path(args.input_index).resolve())
    osworld_entries = load_json(Path(args.osworld_converted).resolve())

    if not isinstance(waa_entries, list):
        raise ValueError(f"WAA index must be a list: {args.input_index}")
    if not isinstance(osworld_entries, list):
        raise ValueError(f"OSWorld converted JSON must be a list: {args.osworld_converted}")

    candidates = [entry for entry in osworld_entries if has_video(entry)]
    target_tokenized, target_vectors = build_vectors(waa_entries, include_titles=False)
    candidate_tokenized, candidate_vectors = build_vectors(candidates, include_titles=True)

    output: List[Dict[str, Any]] = []
    accepted = 0
    eligible = 0
    already_annotated = 0
    rejected_below_threshold = 0
    rejected_ambiguous = 0
    per_domain = defaultdict(lambda: {"eligible": 0, "accepted": 0})
    report_tasks: List[Dict[str, Any]] = []

    for index, target in enumerate(waa_entries):
        if has_video(target) and not args.overwrite_existing:
            already_annotated += 1
            output.append(target)
            continue

        if not should_consider_target(target, args.overwrite_existing):
            output.append(target)
            continue

        eligible += 1
        domain = str(target.get("domain") or "unknown")
        per_domain[domain]["eligible"] += 1

        matches = best_matches(
            target=target,
            target_tokens=target_tokenized[index],
            target_vector=target_vectors[index],
            candidates=candidates,
            candidate_tokens=candidate_tokenized,
            candidate_vectors=candidate_vectors,
            top_k=max(args.report_top_k, 2),
            require_same_app_family=args.require_same_app_family,
        )

        top = matches[0] if matches else None
        second_score = matches[1].score if len(matches) > 1 else 0.0
        accepted_match = False

        if top and top.score >= args.threshold:
            is_ambiguous = (top.score - second_score) < args.margin
            if is_ambiguous and not args.allow_ambiguous:
                rejected_ambiguous += 1
            else:
                output.append(copy_annotation(target, top, args.threshold, args.margin, second_score))
                accepted += 1
                per_domain[domain]["accepted"] += 1
                accepted_match = True
        elif top:
            rejected_below_threshold += 1
        else:
            rejected_below_threshold += 1

        if not accepted_match:
            output.append(target)

        report_tasks.append(
            {
                "target_id": target.get("id", ""),
                "target_domain": target.get("domain", ""),
                "target_web": target.get("web", ""),
                "target_instruction": target.get("instruction", ""),
                "accepted": accepted_match,
                "reason": (
                    "accepted"
                    if accepted_match
                    else "ambiguous"
                    if top and top.score >= args.threshold and (top.score - second_score) < args.margin
                    else "below_threshold"
                ),
                "top_candidates": [candidate_summary(match) for match in matches[: args.report_top_k]],
            }
        )

    report = {
        "input_index": str(Path(args.input_index).resolve()),
        "osworld_converted": str(Path(args.osworld_converted).resolve()),
        "output": str(Path(args.output).resolve()),
        "threshold": args.threshold,
        "margin": args.margin,
        "allow_ambiguous": args.allow_ambiguous,
        "require_same_app_family": args.require_same_app_family,
        "similarity_method": "tfidf_instruction_query_app_v1",
        "summary": {
            "waa_tasks": len(waa_entries),
            "osworld_annotated_candidates": len(candidates),
            "already_annotated": already_annotated,
            "eligible_for_similarity": eligible,
            "accepted": accepted,
            "rejected_below_threshold": rejected_below_threshold,
            "rejected_ambiguous": rejected_ambiguous,
            "final_with_video": sum(1 for entry in output if has_video(entry)),
            "per_domain": dict(sorted(per_domain.items())),
        },
        "tasks": report_tasks,
    }
    return output, report


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-index",
        default=DEFAULT_WAA_INDEX,
        help="Existing WAA video-knowledge index.",
    )
    parser.add_argument(
        "--osworld-converted",
        required=True,
        help="OSWorld converted video-knowledge JSON.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSON path for the similarity-backfilled WAA index.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional report JSON path. Defaults to <output>.report.json.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.58,
        help="Minimum similarity score required to copy video annotations.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.03,
        help="Minimum top1-top2 score margin unless --allow-ambiguous is used.",
    )
    parser.add_argument(
        "--report-top-k",
        type=int,
        default=5,
        help="Number of top candidates to include for each target in the report.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Also reconsider tasks that already have canonical video annotations.",
    )
    parser.add_argument(
        "--allow-ambiguous",
        action="store_true",
        help="Accept matches over threshold even when the top1-top2 margin is small.",
    )
    parser.add_argument(
        "--require-same-app-family",
        action="store_true",
        help="Reject candidates whose app/domain family has no overlap with the target.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary and report path without writing the output index.",
    )
    return parser.parse_args()


def main() -> None:
    args = config()
    output, report = backfill(args)
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve() if args.report else output_path.with_suffix(".report.json")

    if not args.dry_run:
        write_json(output_path, output)
    write_json(report_path, report)

    summary = report["summary"]
    print(
        "Similarity backfill complete: "
        f"eligible={summary['eligible_for_similarity']} "
        f"accepted={summary['accepted']} "
        f"final_with_video={summary['final_with_video']} "
        f"threshold={args.threshold:.2f} "
        f"margin={args.margin:.2f}"
    )
    if args.dry_run:
        print("Dry run: output index was not written.")
    else:
        print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
