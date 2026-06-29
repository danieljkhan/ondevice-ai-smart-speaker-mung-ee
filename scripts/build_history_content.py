"""Build runtime assets for the curated Korean history mode."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from PIL import Image, ImageColor, ImageOps  # type: ignore[import-not-found, import-untyped]

logger = logging.getLogger("mungi.scripts.build_history_content")

JsonObject = dict[str, Any]
EraName = Literal[
    "선사",
    "고조선",
    "삼국",
    "통일신라·발해",
    "고려",
    "조선",
    "근대",
    "일제강점기",
    "현대",
]
EraSource = Literal["keyword", "docnum"]

DEFAULT_DATASET = Path("assets/dataset_korean history")
DEFAULT_OUT = Path("assets/history")
HISTORY_SCHEMA_VERSION = 2
MODE_TITLE = "재미있는 우리역사"
SOURCE_FILE_RE = re.compile(r"^eh_(?P<kind>[nr])(?P<docnum>\d{4})_")

ERA_ORDER: tuple[EraName, ...] = (
    "선사",
    "고조선",
    "삼국",
    "통일신라·발해",
    "고려",
    "조선",
    "근대",
    "일제강점기",
    "현대",
)

ERA_MATCH_ORDER: tuple[EraName, ...] = (
    "현대",
    "일제강점기",
    "근대",
    "조선",
    "고려",
    "통일신라·발해",
    "삼국",
    "고조선",
    "선사",
)

ERA_KEYWORDS: dict[EraName, tuple[str, ...]] = {
    "현대": (
        "현대",
        "광복 후",
        "대한민국 정부 수립",
        "4·19",
        "4ㆍ19",
        "5·18",
        "5ㆍ18",
        "6월 민주",
        "민주 항쟁",
        "민주화 운동",
        "제주 4·3",
        "비무장지대",
        "DMZ",
        "판문점",
        "경제의 성장",
        "사회의 변화",
        "통일 노력",
        "전태일",
        "조선건국준비위원회",
    ),
    "일제강점기": (
        "일제",
        "일본의 식민",
        "식민지",
        "식민 지배",
        "을사늑약",
        "경술국치",
        "독립",
        "항일",
        "3·1",
        "3ㆍ1",
        "만세",
        "임시 정부",
        "의열단",
        "신흥 무관",
        "봉오동",
        "청산리",
        "국채보상",
        "민족 실력",
        "유관순",
        "안중근",
        "김구",
        "윤동주",
        "손기정",
    ),
    "근대": (
        "근대",
        "대한제국",
        "개항",
        "개화",
        "갑신정변",
        "동학",
        "농민 운동",
        "흥선대원군",
        "대원군",
        "천주교",
        "박규수",
        "김옥균",
        "전봉준",
        "최익현",
        "헤이그",
        "서재필",
        "배재학당",
        "이화학당",
        "강화도 유적",
        "정동",
    ),
    "조선": (
        "조선",
        "한양",
        "훈민정음",
        "세종",
        "정조",
        "영조",
        "임진왜란",
        "병자호란",
        "정묘",
        "왕조실록",
        "경복궁",
        "창덕궁",
        "종묘",
        "사직",
        "백자",
        "분청사기",
        "사화",
        "경국대전",
        "의궤",
        "향교",
        "서당",
        "서원",
        "동의보감",
    ),
    "고려": (
        "고려",
        "왕건",
        "광종",
        "강감찬",
        "서희",
        "현종",
        "윤관",
        "의천",
        "지눌",
        "묘청",
        "김부식",
        "정중부",
        "만적",
        "공민왕",
        "최영",
        "정몽주",
        "팔만대장경",
        "직지",
        "청자",
        "삼별초",
        "만월대",
    ),
    "통일신라·발해": (
        "통일신라",
        "발해",
        "문무왕",
        "신문왕",
        "대조영",
        "혜초",
        "장보고",
        "최치원",
        "불국사",
        "석굴암",
        "성덕 대왕 신종",
        "첨성대",
        "동궁과 월지",
        "청해진",
    ),
    "삼국": (
        "삼국",
        "고구려",
        "백제",
        "신라",
        "가야",
        "마한",
        "주몽",
        "온조",
        "박혁거세",
        "김수로",
        "근초고왕",
        "광개토대왕",
        "장수왕",
        "이사부",
        "법흥왕",
        "진흥왕",
        "성왕",
        "을지문덕",
        "연개소문",
        "김유신",
        "김춘추",
        "무령왕릉",
        "금동 대향로",
    ),
    "고조선": (
        "고조선",
        "단군",
        "단군왕검",
        "아사달",
    ),
    "선사": (
        "선사",
        "구석기",
        "신석기",
        "청동기",
        "고인돌",
        "반구대",
        "암사동",
        "전곡리",
        "송국리",
    ),
}

CAPTION_BURN_IN_KEYWORDS = (
    "말풍선",
    "대화",
    "만화",
    "캐릭터",
    "삽화",
    "상상화",
)
CAPTION_CLEAN_KEYWORDS = (
    "유적",
    "터",
    "박물관",
    "문화재",
    "국보",
    "보물",
    "사적",
    "왕릉",
    "사진",
    "현재",
    "기념관",
    "건물",
    "지도",
)
INFOGRAPHIC_CAPTION_KEYWORDS = (
    "한눈에",
    "살펴보기",
    "비교",
    "지도",
    "연표",
    "표",
    "분포",
    "분포도",
)


class BuildError(RuntimeError):
    """Raised when the content build cannot produce valid output."""


@dataclass(frozen=True)
class BuildOptions:
    """Command-line options for one content build."""

    dataset: Path = DEFAULT_DATASET
    out: Path = DEFAULT_OUT
    bg: str = "#1B1B1F"
    max_dim: int = 720
    quality: int = 85
    force: bool = False
    images_only: bool = False
    manifest_only: bool = False
    limit: int | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class SourceInfo:
    """Parsed source-file metadata."""

    source_file: str
    kind: Literal["people", "artifact"]
    doc_num: int


@dataclass(frozen=True)
class TitleInfo:
    """Derived runtime title and whether it came directly from narration."""

    title: str
    curated: bool


@dataclass(frozen=True)
class EraInfo:
    """Resolved era and assignment source."""

    era: EraName
    source: EraSource


@dataclass(frozen=True)
class ImageWorkItem:
    """One source image that may need a derived letterboxed output."""

    source_path: Path
    output_path: Path
    source_size: tuple[int, int]


@dataclass(frozen=True)
class PreparedImage:
    """A letterboxed image and resize metadata."""

    image: Image.Image
    original_size: tuple[int, int]
    content_size: tuple[int, int]
    downscaled: bool


@dataclass
class BuildStats:
    """Counters collected during one build."""

    docs: int = 0
    scenes: int = 0
    sections: int = 0
    image_refs: int = 0
    image_files: int = 0
    images_processed: int = 0
    images_skipped: int = 0
    images_downscaled: int = 0
    images_copied: int = 0
    json_written: int = 0
    title_uncurated: int = 0
    era_source_docnum: int = 0
    clean_false: int = 0
    era_distribution: Counter[str] = field(default_factory=Counter)


def _read_json_object(path: Path) -> JsonObject:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON: {path}"
        raise BuildError(msg) from exc
    except OSError as exc:
        msg = f"Failed to read JSON: {path}"
        raise BuildError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"Expected JSON object: {path}"
        raise BuildError(msg)
    return cast(JsonObject, payload)


def _write_json_if_changed(
    path: Path,
    payload: JsonObject,
    *,
    options: BuildOptions,
) -> bool:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if options.dry_run:
        logger.info("dry-run: would write %s", _display_path(path))
        return False
    if not options.force and path.exists():
        try:
            if path.read_text(encoding="utf-8") == text:
                return False
        except OSError as exc:
            msg = f"Failed to read existing output: {path}"
            raise BuildError(msg) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        msg = f"Failed to write JSON: {path}"
        raise BuildError(msg) from exc
    logger.info("wrote %s", _display_path(path))
    return True


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json_path(path: Path) -> str:
    return _display_path(path)


def _parse_source_info(source_file: str) -> SourceInfo:
    match = SOURCE_FILE_RE.match(source_file)
    if match is None:
        msg = f"Unsupported source_file format: {source_file}"
        raise BuildError(msg)
    kind: Literal["people", "artifact"] = "people"
    if match.group("kind") != "n":
        kind = "artifact"
    return SourceInfo(source_file=source_file, kind=kind, doc_num=int(match.group("docnum")))


def _doc_sort_key(document: JsonObject) -> tuple[int, str]:
    source_file = _require_str(document, "source_file")
    source = _parse_source_info(source_file)
    return (source.doc_num, source.source_file)


def _require_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"Expected string field {key!r}"
        raise BuildError(msg)
    return value


def _require_int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        msg = f"Expected integer field {key!r}"
        raise BuildError(msg)
    return value


def _optional_str(payload: JsonObject, key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"Expected string-or-null field {key!r}"
        raise BuildError(msg)
    return value


def _str_list(payload: JsonObject, key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        msg = f"Expected list field {key!r}"
        raise BuildError(msg)
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            msg = f"Expected only strings in field {key!r}"
            raise BuildError(msg)
        strings.append(item)
    return strings


def _scene_list(document: JsonObject) -> list[JsonObject]:
    value = document.get("scenes")
    if not isinstance(value, list):
        msg = "Expected list field 'scenes'"
        raise BuildError(msg)
    scenes: list[JsonObject] = []
    for item in value:
        if not isinstance(item, dict):
            msg = "Expected each scene to be a JSON object"
            raise BuildError(msg)
        scenes.append(cast(JsonObject, item))
    return scenes


def derive_title(first_scene: JsonObject) -> TitleInfo:
    """Derive the runtime title from scene 1 according to plan section 4.2."""
    narration = _require_str(first_scene, "narration")
    first_line = narration.splitlines()[0].strip() if narration.splitlines() else ""
    stripped = first_line[:-1].strip() if first_line.endswith(",") else first_line
    needs_fallback = first_line.endswith(",") or len(stripped) < 4
    if not needs_fallback:
        return TitleInfo(title=stripped, curated=True)
    fallback = (_optional_str(first_scene, "section_title") or "").strip()
    return TitleInfo(title=fallback or stripped, curated=False)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _contains_keyword(haystack: str, keyword: str) -> bool:
    if keyword == "조선":
        return _contains_joseon_keyword(haystack)
    return keyword in haystack or _compact(keyword) in _compact(haystack)


def _contains_joseon_keyword(haystack: str) -> bool:
    raw_match = any(
        match.start() == 0 or haystack[match.start() - 1] != "고"
        for match in re.finditer("조선", haystack)
    )
    compact_haystack = _compact(haystack)
    compact_match = any(
        match.start() == 0 or compact_haystack[match.start() - 1] != "고"
        for match in re.finditer("조선", compact_haystack)
    )
    return raw_match or compact_match


def map_era(title: str, section_title: str | None, source_file: str) -> EraInfo:
    """Map a document to an era using keywords, then source doc-number fallback."""
    haystack = "\n".join(part for part in (title, section_title or "") if part)
    for era in ERA_MATCH_ORDER:
        if any(_contains_keyword(haystack, keyword) for keyword in ERA_KEYWORDS[era]):
            return EraInfo(era=era, source="keyword")
    return EraInfo(era=_era_from_docnum(_parse_source_info(source_file)), source="docnum")


def _era_from_docnum(source: SourceInfo) -> EraName:
    if source.kind == "people":
        return _people_era_from_docnum(source.doc_num)
    return _artifact_era_from_docnum(source.doc_num)


def _people_era_from_docnum(doc_num: int) -> EraName:
    if doc_num < 20:
        return "고조선"
    if doc_num < 150:
        return "삼국"
    if doc_num < 220:
        return "통일신라·발해"
    if doc_num < 370:
        return "고려"
    if doc_num < 630:
        return "조선"
    if doc_num < 690:
        return "근대"
    if doc_num < 920:
        return "일제강점기"
    return "현대"


def _artifact_era_from_docnum(doc_num: int) -> EraName:
    if doc_num <= 5:
        return "고조선"
    if doc_num < 45:
        return "선사"
    if doc_num < 117:
        return "삼국"
    if doc_num < 122:
        return "통일신라·발해"
    if doc_num < 183:
        return "고려"
    if doc_num < 317:
        return "조선"
    if doc_num < 320:
        return "근대"
    if doc_num < 382:
        return "일제강점기"
    return "현대"


def _load_scene_documents(options: BuildOptions) -> list[JsonObject]:
    if not options.dataset.exists():
        msg = f"Missing dataset: {options.dataset}"
        raise BuildError(msg)
    scenes_dir = options.dataset / "data" / "scenes"
    scene_files = sorted(path for path in scenes_dir.glob("*.json") if path.is_file())
    if not scene_files:
        msg = f"No scene JSON files found under {scenes_dir}"
        raise BuildError(msg)
    documents = [_read_json_object(path) for path in scene_files]
    documents.sort(key=_doc_sort_key)
    if options.limit is not None:
        documents = documents[: options.limit]
    return documents


def _load_clean_overrides(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    payload = _read_json_object(path)
    raw_overrides = payload.get("overrides")
    if isinstance(raw_overrides, dict):
        source = raw_overrides
    else:
        source = payload
    overrides: dict[str, bool] = {}
    for key, value in source.items():
        if key == "schema_version":
            continue
        if isinstance(key, str) and isinstance(value, bool):
            overrides[key] = value
        else:
            logger.warning("Ignoring invalid clean override entry for %s", key)
    return overrides


def _clean_overrides_payload(overrides: dict[str, bool]) -> JsonObject:
    return {
        "schema_version": 1,
        "overrides": dict(sorted(overrides.items())),
    }


def _source_image_path(dataset: Path, raw_path: str) -> Path:
    return dataset / Path(PurePosixPath(raw_path))


def _derived_image_path(out: Path, doc_hash: str, raw_path: str) -> Path:
    return out / "images" / doc_hash / PurePosixPath(raw_path).name


def _read_image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as probe:
            return cast(tuple[int, int], probe.size)
    except OSError as exc:
        msg = f"Failed to read source image: {path}"
        raise BuildError(msg) from exc


def _is_wide(size: tuple[int, int]) -> bool:
    width, height = size
    return width >= int(height * 1.2)


def _caption_has_any(caption: str | None, keywords: tuple[str, ...]) -> bool:
    if not caption:
        return False
    return any(_contains_keyword(caption, keyword) for keyword in keywords)


def _captions_have_any(captions: list[str], keywords: tuple[str, ...]) -> bool:
    return any(_caption_has_any(caption, keywords) for caption in captions)


def infer_clean_flag(
    *,
    source: SourceInfo,
    scene_seq: int,
    source_size: tuple[int, int],
    caption: str | None,
) -> bool:
    """Infer whether an image can be displayed without character-margin handling."""
    clean = source.kind == "artifact"
    if source.kind == "people":
        clean = not (scene_seq == 1 and _is_wide(source_size))
    if _caption_has_any(caption, CAPTION_BURN_IN_KEYWORDS):
        clean = False
    elif not clean and _caption_has_any(caption, CAPTION_CLEAN_KEYWORDS):
        clean = True
    return clean


def _build_image_records(
    *,
    scene: JsonObject,
    source: SourceInfo,
    doc_hash: str,
    dataset: Path,
    out: Path,
    overrides: dict[str, bool],
    stats: BuildStats,
    image_work: dict[str, ImageWorkItem],
) -> list[JsonObject]:
    raw_paths = _str_list(scene, "image_paths")
    raw_captions = _str_list(scene, "image_captions")
    captions_match = len(raw_paths) == len(raw_captions)
    scene_is_infographic = _captions_have_any(raw_captions, INFOGRAPHIC_CAPTION_KEYWORDS)
    scene_seq = _require_int(scene, "seq")
    records: list[JsonObject] = []
    for index, raw_path in enumerate(raw_paths):
        caption = raw_captions[index] if captions_match else None
        is_infographic = _caption_has_any(caption, INFOGRAPHIC_CAPTION_KEYWORDS)
        if caption is None:
            is_infographic = scene_is_infographic
        source_path = _source_image_path(dataset, raw_path)
        output_path = _derived_image_path(out, doc_hash, raw_path)
        output_json_path = _json_path(output_path)
        source_size = _read_image_size(source_path)
        heuristic_clean = infer_clean_flag(
            source=source,
            scene_seq=scene_seq,
            source_size=source_size,
            caption=caption,
        )
        clean = overrides.get(output_json_path, heuristic_clean)
        if not clean:
            stats.clean_false += 1
        stats.image_refs += 1
        image_work[output_json_path] = ImageWorkItem(
            source_path=source_path,
            output_path=output_path,
            source_size=source_size,
        )
        records.append(
            {
                "path": output_json_path,
                "caption": caption,
                "letterboxed": True,
                "clean": clean,
                "is_infographic": is_infographic,
            }
        )
    return records


def _scene_section_title(scene: JsonObject) -> str | None:
    title = _optional_str(scene, "section_title")
    if title is None:
        return None
    stripped = title.strip()
    return stripped or None


def _build_document_outputs(
    *,
    document: JsonObject,
    options: BuildOptions,
    overrides: dict[str, bool],
    stats: BuildStats,
    image_work: dict[str, ImageWorkItem],
) -> tuple[JsonObject, JsonObject, JsonObject]:
    source_file = _require_str(document, "source_file")
    source = _parse_source_info(source_file)
    doc_hash = _require_str(document, "doc_hash")
    scenes = _scene_list(document)
    if not scenes:
        msg = f"Document has no scenes: {source_file}"
        raise BuildError(msg)
    title_info = derive_title(scenes[0])
    if not title_info.title:
        msg = f"Empty derived title: {source_file}"
        raise BuildError(msg)
    section_title = _optional_str(scenes[0], "section_title")
    era_info = map_era(title_info.title, section_title, source_file)

    output_scenes: list[JsonObject] = []
    sections: list[JsonObject] = []
    current_section: JsonObject | None = None
    current_section_title: str | None = None
    current_raw_section_index: int | None = None
    est_total_ms = 0
    image_count = 0
    for scene in scenes:
        raw_section_title = _scene_section_title(scene)
        raw_section_index = scene.get("section_index")
        has_section_index = isinstance(raw_section_index, int)
        starts_indexed_section = (
            has_section_index and raw_section_index != current_raw_section_index
        )
        starts_legacy_section = not has_section_index and raw_section_title is not None
        if current_section is None or starts_indexed_section or starts_legacy_section:
            current_raw_section_index = raw_section_index if has_section_index else None
            current_section_title = raw_section_title
            current_section = {
                "section_index": len(sections),
                "section_title": current_section_title,
                "scene_indices": [],
                "scene_seq": [],
                "image_captions": [],
                "is_infographic": False,
            }
            sections.append(current_section)
        assert current_section is not None
        est_speech_ms = _require_int(scene, "est_speech_ms")
        tail_silence_ms = _require_int(scene, "tail_silence_ms")
        raw_captions = _str_list(scene, "image_captions")
        images = _build_image_records(
            scene=scene,
            source=source,
            doc_hash=doc_hash,
            dataset=options.dataset,
            out=options.out,
            overrides=overrides,
            stats=stats,
            image_work=image_work,
        )
        image_count += len(images)
        est_total_ms += est_speech_ms + tail_silence_ms
        section_index = int(current_section["section_index"])
        current_section["scene_indices"].append(len(output_scenes))
        current_section["scene_seq"].append(_require_int(scene, "seq"))
        current_section["image_captions"].extend(raw_captions)
        if any(bool(image.get("is_infographic")) for image in images):
            current_section["is_infographic"] = True
        output_scenes.append(
            {
                "seq": _require_int(scene, "seq"),
                "section_index": section_index,
                "section_title": current_section_title,
                "narration": _require_str(scene, "narration"),
                "est_speech_ms": est_speech_ms,
                "tail_silence_ms": tail_silence_ms,
                "image_captions": raw_captions,
                "images": images,
            }
        )

    doc_path = options.out / "docs" / f"{doc_hash}.json"
    manifest_entry: JsonObject = {
        "doc_hash": doc_hash,
        "source_file": source_file,
        "title": title_info.title,
        "kind": source.kind,
        "era": era_info.era,
        "scene_count": len(output_scenes),
        "section_count": len(sections),
        "image_count": image_count,
        "est_total_ms": est_total_ms,
        "doc_path": _json_path(doc_path),
        "title_curated": title_info.curated,
        "era_source": era_info.source,
    }
    doc_payload: JsonObject = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "doc_hash": doc_hash,
        "source_file": source_file,
        "title": title_info.title,
        "kind": source.kind,
        "era": era_info.era,
        "era_source": era_info.source,
        "scene_count": len(output_scenes),
        "section_count": len(sections),
        "image_count": image_count,
        "est_total_ms": est_total_ms,
        "sections": sections,
        "scenes": output_scenes,
    }
    era_payload: JsonObject = {
        "era": era_info.era,
        "kind": source.kind,
        "era_source": era_info.source,
    }
    if not title_info.curated:
        stats.title_uncurated += 1
    if era_info.source == "docnum":
        stats.era_source_docnum += 1
    stats.docs += 1
    stats.scenes += len(output_scenes)
    stats.sections += len(sections)
    stats.era_distribution.update([era_info.era])
    return manifest_entry, doc_payload, era_payload


def _manifest_payload(entries: list[JsonObject]) -> JsonObject:
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "title": MODE_TITLE,
        "era_order": list(ERA_ORDER),
        "docs": entries,
    }


def _era_map_payload(entries: dict[str, JsonObject]) -> JsonObject:
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "docs": dict(sorted(entries.items())),
    }


def _parse_rgb(value: str) -> tuple[int, int, int]:
    try:
        color = ImageColor.getrgb(value)
    except ValueError as exc:
        msg = f"Invalid background color: {value}"
        raise BuildError(msg) from exc
    if len(color) < 3:
        msg = f"Invalid RGB color: {value}"
        raise BuildError(msg)
    return (int(color[0]), int(color[1]), int(color[2]))


def prepare_letterboxed_image(
    source_path: Path,
    *,
    bg_rgb: tuple[int, int, int],
    max_dim: int,
) -> PreparedImage:
    """Load one image and return a square letterboxed RGB canvas."""
    try:
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened)
            image = image.convert("RGB")
    except OSError as exc:
        msg = f"Failed to prepare source image: {source_path}"
        raise BuildError(msg) from exc

    original_size = cast(tuple[int, int], image.size)
    width, height = original_size
    max_side = max(width, height)
    downscaled = max_side > max_dim
    if downscaled:
        if width >= height:
            content_size = (max_dim, max(1, round(height * max_dim / width)))
        else:
            content_size = (max(1, round(width * max_dim / height)), max_dim)
        image = image.resize(content_size, Image.Resampling.LANCZOS)
    else:
        content_size = original_size
        image = image.copy()

    canvas = Image.new("RGB", (max_dim, max_dim), bg_rgb)
    left = (max_dim - content_size[0]) // 2
    top = (max_dim - content_size[1]) // 2
    canvas.paste(image, (left, top))
    return PreparedImage(
        image=canvas,
        original_size=original_size,
        content_size=content_size,
        downscaled=downscaled,
    )


def _image_output_is_fresh(item: ImageWorkItem) -> bool:
    if not item.output_path.exists():
        return False
    try:
        return item.output_path.stat().st_mtime >= item.source_path.stat().st_mtime
    except OSError as exc:
        msg = f"Failed to inspect image freshness: {item.output_path}"
        raise BuildError(msg) from exc


def _process_images(
    items: list[ImageWorkItem],
    *,
    options: BuildOptions,
    stats: BuildStats,
) -> None:
    bg_rgb = _parse_rgb(options.bg)
    stats.image_files = len(items)
    for item in items:
        would_downscale = max(item.source_size) > options.max_dim
        if options.dry_run:
            if would_downscale:
                stats.images_downscaled += 1
            else:
                stats.images_copied += 1
            continue
        if not options.force and _image_output_is_fresh(item):
            stats.images_skipped += 1
            continue
        prepared = prepare_letterboxed_image(
            item.source_path,
            bg_rgb=bg_rgb,
            max_dim=options.max_dim,
        )
        item.output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            prepared.image.save(
                item.output_path,
                format="JPEG",
                quality=options.quality,
                optimize=True,
                progressive=True,
            )
        except OSError as exc:
            msg = f"Failed to write derived image: {item.output_path}"
            raise BuildError(msg) from exc
        stats.images_processed += 1
        if prepared.downscaled:
            stats.images_downscaled += 1
        else:
            stats.images_copied += 1


def build_history_content(options: BuildOptions) -> BuildStats:
    """Build manifest JSON and derived images for curated history playback."""
    if options.max_dim <= 0:
        msg = "--max-dim must be positive"
        raise BuildError(msg)
    if options.quality < 1 or options.quality > 95:
        msg = "--quality must be between 1 and 95"
        raise BuildError(msg)
    if options.images_only and options.manifest_only:
        msg = "--images-only and --manifest-only are mutually exclusive"
        raise BuildError(msg)
    if options.limit is not None and options.limit < 1:
        msg = "--limit must be positive"
        raise BuildError(msg)
    _parse_rgb(options.bg)

    documents = _load_scene_documents(options)
    overrides_path = options.out / "clean_overrides.json"
    overrides = _load_clean_overrides(overrides_path)
    stats = BuildStats()
    manifest_entries: list[JsonObject] = []
    doc_payloads: dict[str, JsonObject] = {}
    era_entries: dict[str, JsonObject] = {}
    image_work: dict[str, ImageWorkItem] = {}

    for document in documents:
        doc_hash = _require_str(document, "doc_hash")
        manifest_entry, doc_payload, era_payload = _build_document_outputs(
            document=document,
            options=options,
            overrides=overrides,
            stats=stats,
            image_work=image_work,
        )
        manifest_entries.append(manifest_entry)
        doc_payloads[doc_hash] = doc_payload
        era_entries[doc_hash] = era_payload

    if not options.images_only:
        json_outputs: list[tuple[Path, JsonObject]] = [
            (options.out / "manifest.json", _manifest_payload(manifest_entries)),
            (options.out / "era_map.json", _era_map_payload(era_entries)),
            (overrides_path, _clean_overrides_payload(overrides)),
        ]
        for doc_hash, payload in sorted(doc_payloads.items()):
            json_outputs.append((options.out / "docs" / f"{doc_hash}.json", payload))
        for path, payload in json_outputs:
            if _write_json_if_changed(path, payload, options=options):
                stats.json_written += 1

    if not options.manifest_only:
        _process_images(
            list(image_work.values()),
            options=options,
            stats=stats,
        )
    else:
        stats.image_files = len(image_work)

    _log_build_stats(stats)
    return stats


def _log_build_stats(stats: BuildStats) -> None:
    logger.info(
        "docs=%s sections=%s scenes=%s image_refs=%s",
        stats.docs,
        stats.sections,
        stats.scenes,
        stats.image_refs,
    )
    logger.info(
        "image_files=%s processed=%s skipped=%s downscaled=%s copied=%s",
        stats.image_files,
        stats.images_processed,
        stats.images_skipped,
        stats.images_downscaled,
        stats.images_copied,
    )
    logger.info("title_curated_false=%s", stats.title_uncurated)
    logger.info("era_distribution=%s", dict(sorted(stats.era_distribution.items())))
    logger.info("era_source_docnum=%s", stats.era_source_docnum)
    logger.info("clean_false=%s", stats.clean_false)
    logger.info("json_written=%s", stats.json_written)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bg", default="#1B1B1F")
    parser.add_argument("--max-dim", type=int, default=720)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--force", action="store_true")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--images-only", action="store_true")
    mode_group.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _options_from_args(args: argparse.Namespace) -> BuildOptions:
    return BuildOptions(
        dataset=args.dataset,
        out=args.out,
        bg=args.bg,
        max_dim=args.max_dim,
        quality=args.quality,
        force=args.force,
        images_only=args.images_only,
        manifest_only=args.manifest_only,
        limit=args.limit,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        build_history_content(_options_from_args(args))
    except BuildError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
