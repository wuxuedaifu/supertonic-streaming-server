"""Text normalisation for supertonic synthesis.

Supertonic-3 uses a 65 536-entry Unicode indexer (character-level
tokenizer) and can ingest raw Chinese / Japanese / Korean / Arabic
characters without transliteration. What it CAN'T do is read digits or
punctuation aloud sensibly — fed "200" the model emits "two zero zero"
(per-digit) rather than "two hundred". Likewise "Dr." becomes "dee
ar dot" and "5.5%" becomes "five point five percent sign".

This module fixes that by running, on the full text before sentence
chunking:

    * numbers          : ``200``  → ``two hundred`` / ``二百``
    * decimal points   : ``5.5``  → ``five point five``
    * currency         : ``$10``  → ``ten dollars``
    * ordinals (en/ru) : ``2nd``  → ``second``
    * symbols          : ``%``    → ``percent`` / ``百分之``
    * abbreviations    : ``Dr.``  → ``doctor``
    * whitespace       : ``"a  b\\n c"`` → ``"a b c"``

The number/currency/abbreviation tables are lifted verbatim from the
auralis XTTS tokenizer (`tokenizer-1.py`) — those are the same rules
the user has battle-tested. For Chinese we delegate to
:class:`zh_num2words.TextNorm`, which handles "200万 → 二百万" and
similar shapes that vanilla ``num2words`` doesn't.

What we deliberately do NOT do (vs. the XTTS tokenizer):

    * No pinyin / romaji / hangul transliteration — supertonic-3 takes
      raw Unicode and does its own phonemisation.
    * No spaCy sentence splitting — supertonic.utils.chunk_text already
      splits sentence-bounded; running our own splitter would just
      double-chunk.
    * No forced lowercase by default. XTTS was trained on lowercased
      text; supertonic-3 is case-aware, and force-lowercasing can hurt
      proper-noun pronunciation. Toggle with SUPERTONIC_PREPROCESS_LOWERCASE=1
      if your tests show otherwise.

Only one extra runtime dep: ``num2words`` (pure Python, no native deps).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from num2words import num2words

from zh_num2words import TextNorm as _ZhNum2Words

logger = logging.getLogger(__name__)


# supertonic-3 advertises 31 languages. We can't write XTTS-style
# number/symbol/abbreviation tables for all of them — but we CAN at
# least expand digits anywhere num2words ships a converter. So the
# preprocess has three tiers:
#
#   FULL_RULES_LANGS — XTTS-style multilingual_cleaners (numbers via
#     num2words + currency + ordinals + symbols + abbreviations).
#     "zh" gets number expansion via the locally-shipped TextNorm
#     instead, which handles "200万 / 5.5%" shapes vanilla num2words
#     misses.
#
#   NUMBER_ONLY_LANGS — supertonic-supported language that num2words
#     handles but we don't (yet) have symbol / abbrev tables for.
#     We still expand decimals, ordinals (where the regex exists) and
#     bare digits; symbols and abbreviations pass through.
#
#   PASSTHROUGH (bg, el, et, hi, hr) — supertonic-supported but
#     num2words has no converter at all. Digits pass through and the
#     model reads them per-character. Adding rules for these means
#     either a dedicated library (e.g. hindi-num2words) or hand-rolled
#     tables; out of scope until we see a real-world request for one.
# Partitioned to sum exactly to supertonic-3's official 31 languages,
# plus ``zh`` which the model accepts via its Unicode tokenizer even
# though it isn't on the advertised list — kept here so users who feed
# Chinese text (numbers like "200万", "5.5%") still get normalisation.
FULL_RULES_LANGS = {
    "ar", "cs", "de", "en", "es", "fr", "hu", "it", "ko",
    "nl", "pl", "pt", "ru", "tr",
    "zh",  # unofficial — supertonic-3 has no "zh" lang but the model accepts the chars
}
NUMBER_ONLY_LANGS = {
    "da", "fi", "id", "ja", "lt", "lv", "ro", "sk", "sl", "sv", "uk", "vi",
}
# Supertonic-supported langs we explicitly know we cannot expand
# (num2words has no converter). Kept for transparency / log clarity;
# functionally same as anything-not-in-SUPPORTED_LANGS.
PASSTHROUGH_LANGS = {"bg", "el", "et", "hi", "hr"}
SUPPORTED_LANGS = FULL_RULES_LANGS | NUMBER_ONLY_LANGS

_LOWERCASE = os.environ.get("SUPERTONIC_PREPROCESS_LOWERCASE", "0") == "1"


_whitespace_re = re.compile(r"\s+")


# ----- abbreviations -----
# Each entry is a (compiled regex, replacement) pair, applied
# case-insensitively against word-boundaried "<abbr>." matches.
_abbreviations = {
    "en": [
        (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"),
            ("st", "saint"),   ("co", "company"), ("jr", "junior"),
            ("maj", "major"),  ("gen", "general"), ("drs", "doctors"),
            ("rev", "reverend"), ("lt", "lieutenant"),
            ("hon", "honorable"), ("sgt", "sergeant"),
            ("capt", "captain"), ("esq", "esquire"),
            ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
        ]
    ],
    "es": [
        (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sra", "señora"), ("sr", "señor"), ("dr", "doctor"),
            ("dra", "doctora"), ("st", "santo"), ("co", "compañía"),
            ("jr", "junior"), ("ltd", "limitada"),
        ]
    ],
    "fr": [
        (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mme", "madame"), ("mr", "monsieur"), ("dr", "docteur"),
            ("st", "saint"), ("co", "compagnie"), ("jr", "junior"),
            ("ltd", "limitée"),
        ]
    ],
    "de": [
        (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("fr", "frau"), ("dr", "doktor"), ("st", "sankt"),
            ("co", "firma"), ("jr", "junior"),
        ]
    ],
    "pt": [
        (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sra", "senhora"), ("sr", "senhor"), ("dr", "doutor"),
            ("dra", "doutora"), ("st", "santo"), ("co", "companhia"),
            ("jr", "júnior"), ("ltd", "limitada"),
        ]
    ],
    "it": [
        (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sig", "signore"), ("dr", "dottore"), ("st", "santo"),
            ("co", "compagnia"), ("jr", "junior"), ("ltd", "limitata"),
        ]
    ],
    "pl": [
        (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("p", "pani"), ("m", "pan"), ("dr", "doktor"),
            ("sw", "święty"), ("jr", "junior"),
        ]
    ],
    "cs": [
        (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"), ("ing", "inženýr"), ("p", "pan"),
        ]
    ],
    "ru": [
        (re.compile(r"\b%s\b" % x[0], re.IGNORECASE), x[1])
        for x in [
            ("г-жа", "госпожа"), ("г-н", "господин"),
            ("д-р", "доктор"),
        ]
    ],
    "nl": [
        (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dhr", "de heer"), ("mevr", "mevrouw"),
            ("dr", "dokter"), ("jhr", "jonkheer"),
        ]
    ],
    "tr": [
        (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("b", "bay"), ("byk", "büyük"), ("dr", "doktor"),
        ]
    ],
    "hu": [
        (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"), ("b", "bácsi"), ("nőv", "nővér"),
        ]
    ],
}


def expand_abbreviations(text: str, lang: str) -> str:
    for regex, repl in _abbreviations.get(lang, []):
        text = regex.sub(repl, text)
    return text


# ----- symbols -----
_symbols = {
    "en": [("&", " and "), ("@", " at "),       ("%", " percent "),    ("#", " hash "),    ("$", " dollar "), ("£", " pound "),    ("°", " degree ")],
    "es": [("&", " y "),   ("@", " arroba "),   ("%", " por ciento "), ("#", " numeral "), ("$", " dolar "),  ("£", " libra "),    ("°", " grados ")],
    "fr": [("&", " et "),  ("@", " arobase "),  ("%", " pour cent "),  ("#", " dièse "),   ("$", " dollar "), ("£", " livre "),    ("°", " degrés ")],
    "de": [("&", " und "), ("@", " at "),       ("%", " prozent "),    ("#", " raute "),   ("$", " dollar "), ("£", " pfund "),    ("°", " grad ")],
    "pt": [("&", " e "),   ("@", " arroba "),   ("%", " por cento "),  ("#", " cardinal "),("$", " dólar "),  ("£", " libra "),    ("°", " graus ")],
    "it": [("&", " e "),   ("@", " chiocciola "), ("%", " per cento "), ("#", " cancelletto "), ("$", " dollaro "), ("£", " sterlina "), ("°", " gradi ")],
    "pl": [("&", " i "),   ("@", " małpa "),    ("%", " procent "),    ("#", " krzyżyk "), ("$", " dolar "),  ("£", " funt "),     ("°", " stopnie ")],
    "ar": [("&", " و "),    ("@", " على "),       ("%", " في المئة "),    ("#", " رقم "),     ("$", " دولار "),    ("£", " جنيه "),       ("°", " درجة ")],
    "zh": [("&", " 和 "),   ("@", " 在 "),        ("%", " 百分之 "),       ("#", " 号 "),       ("$", " 美元 "),     ("£", " 英镑 "),       ("°", " 度 ")],
    "cs": [("&", " a "),   ("@", " na "),       ("%", " procento "),   ("#", " křížek "),  ("$", " dolar "),  ("£", " libra "),    ("°", " stupně ")],
    "ru": [("&", " и "),   ("@", " собака "),   ("%", " процентов "),  ("#", " номер "),   ("$", " доллар "), ("£", " фунт "),     ("°", " градус ")],
    "nl": [("&", " en "),  ("@", " bij "),      ("%", " procent "),    ("#", " hekje "),   ("$", " dollar "), ("£", " pond "),     ("°", " graden ")],
    "tr": [("&", " ve "),  ("@", " at "),       ("%", " yüzde "),      ("#", " diyez "),   ("$", " dolar "),  ("£", " sterlin "),  ("°", " derece ")],
    "hu": [("&", " és "),  ("@", " kukac "),    ("%", " százalék "),   ("#", " kettőskereszt "), ("$", " dollár "), ("£", " font "), ("°", " fok ")],
    "ko": [("&", " 그리고 "), ("@", " 에 "),       ("%", " 퍼센트 "),       ("#", " 번호 "),     ("$", " 달러 "),     ("£", " 파운드 "),     ("°", " 도 ")],
}
_symbols_compiled = {
    lang: [(re.compile(re.escape(src), re.IGNORECASE), repl) for src, repl in items]
    for lang, items in _symbols.items()
}


def expand_symbols(text: str, lang: str) -> str:
    for regex, repl in _symbols_compiled.get(lang, []):
        text = regex.sub(repl, text)
        text = text.replace("  ", " ")
    return text.strip()


# ----- numbers / currency / ordinals -----
_ordinal_re = {
    "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
    "es": re.compile(r"([0-9]+)(º|ª|er|o|a|os|as)"),
    "fr": re.compile(r"([0-9]+)(º|ª|er|re|e|ème)"),
    "de": re.compile(r"([0-9]+)(st|nd|rd|th|º|ª|\.(?=\s|$))"),
    "pt": re.compile(r"([0-9]+)(º|ª|o|a|os|as)"),
    "it": re.compile(r"([0-9]+)(º|°|ª|o|a|i|e)"),
    "pl": re.compile(r"([0-9]+)(º|ª|st|nd|rd|th)"),
    "ar": re.compile(r"([0-9]+)(ون|ين|ث|ر|ى)"),
    "cs": re.compile(r"([0-9]+)\.(?=\s|$)"),
    "ru": re.compile(r"([0-9]+)(-й|-я|-е|-ое|-ье|-го)"),
    "nl": re.compile(r"([0-9]+)(de|ste|e)"),
    "tr": re.compile(r"([0-9]+)(\.|inci|nci|uncu|üncü|\.)"),
    "hu": re.compile(r"([0-9]+)(\.|adik|edik|odik|edik|ödik|ödike|ik)"),
    "ko": re.compile(r"([0-9]+)(번째|번|차|째)"),
}
_number_re = re.compile(r"[0-9]+")
_currency_re = {
    "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
    "GBP": re.compile(r"((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£))"),
    "EUR": re.compile(r"(([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+)))"),
}
_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
_dot_number_re = re.compile(r"\b\d{1,3}(\.\d{3})*(\,\d+)?\b")
_decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)")


def _lang_to_n2w(lang: str) -> str:
    # num2words uses "cz" for Czech, not "cs".
    return "cz" if lang == "cs" else lang


def _remove_commas(m: "re.Match[str]") -> str:
    t = m.group(0)
    return t.replace(",", "") if "," in t else t


def _remove_dots(m: "re.Match[str]") -> str:
    t = m.group(0)
    return t.replace(".", "") if "." in t else t


def _expand_decimal_point(m: "re.Match[str]", lang: str) -> str:
    amount = m.group(1).replace(",", ".")
    return num2words(float(amount), lang=_lang_to_n2w(lang))


_and_equiv = {
    "en": ", ", "es": " con ", "fr": " et ", "de": " und ", "pt": " e ",
    "it": " e ", "pl": ", ", "cs": ", ", "ru": ", ", "nl": ", ",
    "ar": ", ", "tr": ", ", "hu": ", ", "ko": ", ",
}


def _expand_currency(m: "re.Match[str]", lang: str, currency: str) -> str:
    amount = float(re.sub(r"[^\d.]", "", m.group(0).replace(",", ".")))
    full = num2words(amount, to="currency", currency=currency, lang=_lang_to_n2w(lang))
    if amount.is_integer():
        sep = _and_equiv.get(lang, ", ")
        last = full.rfind(sep)
        if last != -1:
            full = full[:last]
    return full


def _expand_ordinal(m: "re.Match[str]", lang: str) -> str:
    return num2words(int(m.group(1)), ordinal=True, lang=_lang_to_n2w(lang))


def _expand_number(m: "re.Match[str]", lang: str) -> str:
    return num2words(int(m.group(0)), lang=_lang_to_n2w(lang))


# Cached zh normalizer — TextNorm builds a regex pipeline at construction
# time and the same instance is fine to reuse across requests.
_zh_norm: Optional[_ZhNum2Words] = None


def _get_zh_norm() -> _ZhNum2Words:
    global _zh_norm
    if _zh_norm is None:
        _zh_norm = _ZhNum2Words()
    return _zh_norm


def expand_numbers(text: str, lang: str) -> str:
    if lang == "zh":
        return _get_zh_norm()(text)
    # Latin-script path: strip thousands separators, expand currency,
    # decimals, ordinals, then bare digits.
    if lang in ("en", "ru"):
        text = _comma_number_re.sub(_remove_commas, text)
    else:
        text = _dot_number_re.sub(_remove_dots, text)
    try:
        text = _currency_re["GBP"].sub(lambda m: _expand_currency(m, lang, "GBP"), text)
        text = _currency_re["USD"].sub(lambda m: _expand_currency(m, lang, "USD"), text)
        text = _currency_re["EUR"].sub(lambda m: _expand_currency(m, lang, "EUR"), text)
    except Exception:
        logger.debug("currency expand raised, continuing", exc_info=True)
    if lang != "tr":
        text = _decimal_number_re.sub(lambda m: _expand_decimal_point(m, lang), text)
    if lang in _ordinal_re:
        text = _ordinal_re[lang].sub(lambda m: _expand_ordinal(m, lang), text)
    text = _number_re.sub(lambda m: _expand_number(m, lang), text)
    return text


def collapse_whitespace(text: str) -> str:
    return _whitespace_re.sub(" ", text).strip()


def multilingual_cleaners(text: str, lang: str) -> str:
    """Normalise `text` for supertonic synthesis (full pipeline)."""
    text = text.replace('"', "")
    if lang == "tr":
        text = text.replace("İ", "i").replace("Ö", "ö").replace("Ü", "ü")
    if _LOWERCASE:
        text = text.lower()
    text = expand_numbers(text, lang)
    text = expand_abbreviations(text, lang)
    text = expand_symbols(text, lang)
    text = collapse_whitespace(text)
    return text


def numbers_only_cleaners(text: str, lang: str) -> str:
    """Lightweight cleaner for langs without symbol / abbreviation tables.

    Just expands digits via num2words (decimals, ordinals where the
    regex exists, then bare integers) and collapses whitespace.
    Symbols (%, $, ...) and abbreviations pass through unchanged.
    """
    if _LOWERCASE:
        text = text.lower()
    text = expand_numbers(text, lang)
    text = collapse_whitespace(text)
    return text


def basic_cleaners(text: str) -> str:
    if _LOWERCASE:
        text = text.lower()
    return collapse_whitespace(text)


# ----- language detection -----
# Statistical LID via lingua-py (75 languages, pure Python, ~90 MB
# of bundled language models). Replaces the earlier regex-based
# script checks because:
#   * regex distinguishes scripts, not languages: it can spot Thai
#     vs. Hebrew but cannot tell Norwegian from English (both Latin)
#     or Belarusian from Russian (both Cyrillic).
#   * lingua-py's models are trained on actual language statistics
#     and reach >95 % accuracy on text as short as one sentence.
#
# Tradeoffs we accept:
#   * +90 MB image weight (negligible against the 8.6 GB CUDA base)
#   * +2-3 s startup to decompress models on first detect call
#     (preloaded once at module import via _warmup_detector())
#   * ~1-5 ms per request after warmup
#
# Lingua's ISO 639-1 codes line up 1:1 with supertonic's language
# codes, so no translation table is needed.
import threading as _threading  # local alias to avoid shadowing
_DETECTOR = None
_DETECTOR_LOCK = _threading.Lock()
# Sentinel returned by _get_detector() if lingua can't be imported
# (e.g. the dep was stripped from the image). Caller treats `None`
# as "detection disabled, pass text through unchanged".
_DETECTOR_UNAVAILABLE = object()

# Supertonic-3's 31 official languages, by ISO 639-1 code. Matches
# supertonic.config.SUPPORTED_LANGUAGES one-for-one.
SUPERTONIC_LANGS = {
    "ar", "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hi", "hu", "id", "it", "ja", "ko", "lv", "lt", "pl", "pt",
    "ro", "ru", "sk", "sl", "es", "sv", "tr", "uk", "vi",
}

# Two thresholds:
#   _MIN_DETECT_CHARS — below this we don't even try (passthrough).
#   _MIN_LINGUA_CHARS — below this we trust *only* the script-based
#     pre-pass. Lingua's statistical model wobbles on short
#     Latin/Cyrillic text ("OK" → Norwegian, "ad" → Catalan) — those
#     false positives would wrongly trigger refusal for valid English
#     requests. The script pre-pass is reliable on any length because
#     scripts like Hiragana / Hebrew / Thai are unique to one language
#     family.
_MIN_DETECT_CHARS = int(os.environ.get("SUPERTONIC_MIN_DETECT_CHARS", "2"))
_MIN_LINGUA_CHARS = int(os.environ.get("SUPERTONIC_MIN_LINGUA_CHARS", "30"))
# Confidence margin between top-1 and top-2 lingua results below which
# we treat the detection as ambiguous (returns None → passthrough).
# Real-world false positive that motivated this: "Hello, I am kayan."
# (15 chars) → lingua picked Tagalog because "kayan" looks Filipino,
# top-2 was English with a tiny margin.
_LINGUA_MIN_RELATIVE_DISTANCE = float(
    os.environ.get("SUPERTONIC_LINGUA_MIN_DISTANCE", "0.15")
)

# Sentinel for "we know this text is not in supertonic's 31 but lingua
# can't / didn't run". is_supertonic_lang("und") → False so the
# refusal phrase fires. "und" is the ISO 639-3 code for undetermined.
_UNDETERMINED = "und"

# ----- script-range pre-pass -----
# These regexes resolve script-script ambiguity in a single pass, no
# language model needed. Works on 1-char inputs.
_HIRAGANA_RE = re.compile(r"[぀-ゟ]")
_KATAKANA_RE = re.compile(r"[゠-ヿㇰ-ㇿ]")
_HANGUL_RE   = re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]")
# CJK Unified Ideographs + Extension A + Compatibility Ideographs.
# Shared by zh/ja/ko, so presence alone isn't enough — we look at
# whether kana / hangul co-occur to disambiguate.
_CJK_HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
# Scripts that own a dedicated Unicode block AND are NOT used by any
# of supertonic-3's 31 langs. Hit any of these → unsupported.
_UNSUPPORTED_SCRIPT_RE = re.compile(
    "["
    "԰-֏"   # Armenian
    "֐-׿"   # Hebrew
    "ঀ-৿"   # Bengali / Assamese
    "਀-੿"   # Gurmukhi (Punjabi)
    "઀-૿"   # Gujarati
    "଀-୿"   # Oriya
    "஀-௿"   # Tamil
    "ఀ-౿"   # Telugu
    "ಀ-೿"   # Kannada
    "ഀ-ൿ"   # Malayalam
    "඀-෿"   # Sinhala
    "฀-๿"   # Thai
    "຀-໿"   # Lao
    "ༀ-࿿"   # Tibetan
    "က-႟"   # Myanmar (Burmese)
    "Ⴀ-ჿ"   # Georgian
    "ሀ-፿"   # Ethiopic (Amharic, Tigrinya)
    "ក-៿"   # Khmer
    "]"
)


def _script_based_detect(text: str) -> Optional[str]:
    """Best-effort language from Unicode script ranges. Reliable on
    any input length for scripts unique to one language family.
    Returns ISO 639-1 code, the _UNDETERMINED sentinel, or None when
    the script doesn't pin a single language family.
    """
    if _UNSUPPORTED_SCRIPT_RE.search(text):
        return _UNDETERMINED
    if _HIRAGANA_RE.search(text) or _KATAKANA_RE.search(text):
        return "ja"
    if _HANGUL_RE.search(text):
        return "ko"
    if _CJK_HAN_RE.search(text):
        # Han ideographs without kana or hangul → Chinese. Modern
        # Korean is hangul-only, modern Japanese always has kana —
        # so pure-Han is overwhelmingly Chinese in practice.
        return "zh"
    return None


# Cache of all ISO 639-1 codes lingua knows about (built lazily so a
# missing-import on lingua doesn't crash module load). Used so that
# an explicit lang_hint like "th" or "no" (real codes, but not in
# supertonic-31) refuses synthesis straight away.
_LINGUA_KNOWN_LANGS: Optional[set] = None


def _get_lingua_known_langs() -> set:
    global _LINGUA_KNOWN_LANGS
    if _LINGUA_KNOWN_LANGS is None:
        try:
            from lingua import Language
            # Language in lingua-py is a regular class, not an enum —
            # use .all() to iterate every defined language.
            _LINGUA_KNOWN_LANGS = {
                lang.iso_code_639_1.name.lower() for lang in Language.all()
            }
        except Exception:
            logger.exception("failed to enumerate lingua languages")
            _LINGUA_KNOWN_LANGS = set()
    return _LINGUA_KNOWN_LANGS


def _build_detector():
    """Construct the lingua detector eagerly (decompresses all 75
    language models, ~2-3 s, ~90 MB resident). Called once at startup
    via :func:`warmup_detector`; subsequent calls are instant.

    The ``with_minimum_relative_distance`` setting makes lingua
    return None when the top-1 and top-2 candidates are within that
    confidence distance — i.e. the model isn't sure. This catches
    short / ambiguous inputs like "Hello, I am kayan." that would
    otherwise be mis-classified.
    """
    from lingua import LanguageDetectorBuilder
    return (
        LanguageDetectorBuilder
        .from_all_languages()
        .with_minimum_relative_distance(_LINGUA_MIN_RELATIVE_DISTANCE)
        .with_preloaded_language_models()
        .build()
    )


def _get_detector():
    """Thread-safe singleton accessor for the lingua detector.

    Returns the built detector, or None if lingua isn't importable
    or model build failed. Callers treat None as "detection
    disabled, pass text through".
    """
    global _DETECTOR
    if _DETECTOR is _DETECTOR_UNAVAILABLE:
        return None
    if _DETECTOR is not None:
        return _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is _DETECTOR_UNAVAILABLE:
            return None
        if _DETECTOR is not None:
            return _DETECTOR
        try:
            _DETECTOR = _build_detector()
            logger.info(
                "lingua language detector ready (75 languages preloaded)"
            )
        except ImportError:
            logger.warning(
                "lingua-language-detector not installed; LID disabled, "
                "all inputs will pass through to supertonic unchanged"
            )
            _DETECTOR = _DETECTOR_UNAVAILABLE
            return None
        except Exception:
            logger.exception("lingua detector failed to build; LID disabled")
            _DETECTOR = _DETECTOR_UNAVAILABLE
            return None
    return _DETECTOR


def warmup_detector() -> bool:
    """Eagerly build the lingua detector. Call this from server
    startup so the 2-3 s decompression cost is paid before /ready
    flips true, not on the first inbound request.
    """
    return _get_detector() is not None


def detect_language(text: str, lang_hint: Optional[str] = None) -> Optional[str]:
    """Return ISO 639-1 code for `text`, or None if uncertain.

    Detection layers (applied in order, first hit wins):
      1. Explicit ``lang_hint`` matching supertonic-31 or "zh".
      2. Script-range pre-pass — catches Hiragana / Hangul / Han /
         Hebrew / Thai / Tamil / … on inputs as short as one
         character. Returns "ja" / "ko" / "zh" / `_UNDETERMINED`
         accordingly.
      3. Lingua-py LID — only invoked when text has at least
         ``_MIN_LINGUA_CHARS`` non-whitespace characters, because
         lingua's statistical model produces false positives on
         short Latin / Cyrillic text (e.g. "OK" → Norwegian) that
         would wrongly trigger a refusal.

    Returns None when none of the above can pin down a language —
    caller treats that as "pass through unchanged".
    """
    if not text or not text.strip():
        return None
    if lang_hint:
        base = lang_hint.split("-")[0].lower()
        if base in SUPERTONIC_LANGS or base == "zh":
            return base
        # Hint looks like a real ISO 639-1 code (2 letters, alpha)
        # but isn't in supertonic-31 → refuse outright. Examples:
        # "th" (Thai), "no" (Norwegian), "he" (Hebrew), "fa" (Persian).
        # Sentinel strings like "auto" / "na" / "" / "unk" / "und"
        # are explicitly excluded so they fall through to text-based
        # detection.
        if (
            len(base) == 2
            and base.isalpha()
            and base not in {"na", "un"}  # supertonic UNKNOWN sentinel + variants
        ):
            return _UNDETERMINED
    non_space = sum(1 for c in text if not c.isspace())
    if non_space < _MIN_DETECT_CHARS:
        return None

    # (2) script pre-pass — works on any length.
    script_guess = _script_based_detect(text)
    if script_guess is not None:
        return script_guess

    # (3) lingua — only on text long enough to be reliable.
    if non_space < _MIN_LINGUA_CHARS:
        return None
    det = _get_detector()
    if det is None:
        return None
    try:
        result = det.detect_language_of(text)
    except Exception:
        logger.exception("lingua detection raised on %r", text[:80])
        return None
    if result is None:
        return None
    return result.iso_code_639_1.name.lower()


def is_supertonic_lang(code: Optional[str]) -> bool:
    """True if `code` (ISO 639-1) is one of supertonic-3's 31 langs."""
    return bool(code) and code in SUPERTONIC_LANGS




def preprocess_text(text: str, lang: Optional[str]) -> str:
    """Entry point used by the streaming server.

    Returns ``text`` normalised for `lang`. Three tiers (see top of
    module): full rules → number-only → passthrough. Never raises —
    on any per-language failure the original text is returned so
    synthesis still proceeds (degraded pronunciation is better than
    a 500).
    """
    if not text:
        return text
    if not lang:
        return basic_cleaners(text)
    base = lang.split("-")[0].lower()
    try:
        if base in FULL_RULES_LANGS:
            return multilingual_cleaners(text, base)
        if base in NUMBER_ONLY_LANGS:
            return numbers_only_cleaners(text, base)
        # PASSTHROUGH_LANGS + everything unknown.
        return basic_cleaners(text)
    except Exception:
        logger.exception("preprocess_text failed for lang=%s; passing raw text", base)
        return text
