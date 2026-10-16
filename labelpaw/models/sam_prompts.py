"""Translate bounded Chinese object phrases while keeping annotation labels."""

import re

PERSON_PROMPT_ALIASES = {
    "人", "人体", "人物", "人员", "行人", "人群", "人类", "完整的人",
    "person", "people", "human", "humans",
}

HAND_PROMPT_ALIASES = {
    "手", "手掌", "手部", "人的手", "人的手掌", "双手", "手章",
    "hand", "hands", "human hand", "human hands",
}

_TAPE_COLORS = {
    "白": "white", "白色": "white",
    "蓝": "blue", "蓝色": "blue",
    "红": "red", "红色": "red",
    "黑": "black", "黑色": "black",
    "透明": "transparent",
}
_HAND_HELD_PREFIXES = (
    "人手中拿的", "人手中拿着的", "人手里拿的", "人手里拿着的",
    "手中拿的", "手中拿着的", "手里拿的", "手里拿着的",
    "拿在手里的", "拿在手中的",
)
_TAPE_PHRASE = re.compile(
    r"(?:(?P<held>" + "|".join(re.escape(prefix) for prefix in _HAND_HELD_PREFIXES) + r")\s*)?"
    r"(?:(?P<one_roll>一卷)\s*)?"
    r"(?:(?P<color>白色?|蓝色?|红色?|黑色?|透明)\s*(?:的\s*)?)?"
    r"(?:胶带|胶布)(?:\s*(?P<roll>卷))?"
)


def _normalize_tape_prompt(prompt):
    # Match the entire phrase. Substring replacement could silently discard
    # exclusions, different holders, shades, or other unknown qualifiers.
    match = _TAPE_PHRASE.fullmatch(prompt)
    if match is None or (match['one_roll'] and match['roll']):
        return None
    color = _TAPE_COLORS.get(match['color'])
    phrase = f"{color} tape" if color else "tape"
    if match['one_roll']:
        phrase = f"a roll of {phrase}"
    elif match['roll']:
        phrase = f"roll of {phrase}"
    if match['held']:
        phrase += " held in a hand"
    return phrase


def normalize_sam_prompt(prompt):
    prompt = prompt.strip()
    normalized = prompt.casefold()
    if normalized in PERSON_PROMPT_ALIASES:
        return "person"
    if normalized in HAND_PROMPT_ALIASES:
        return "hand"
    tape_prompt = _normalize_tape_prompt(prompt)
    return tape_prompt if tape_prompt is not None else prompt


def split_sam_prompts(prompt):
    return [part.strip() for part in re.split(r"[,，、]+", prompt) if part.strip()]
