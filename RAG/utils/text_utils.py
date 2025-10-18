import re

INPUT_ALIASES = {
    "gotenburg": "Göteborg","gothenburg":"Göteborg","goteborg":"Göteborg",
    "linkoping":"Linköping","ostergotland":"Östergötland",
    "vastra gotaland":"Västra Götaland","varmland":"Värmland",
    "orebro":"Örebro","gavle":"Gävle","angelholm":"Ängelholm",
}
NAME_REMAP = {
    r"\bGothenburg\b": "Göteborg", r"\bOrebro\b": "Örebro",
    r"\bOstergotland\b": "Östergötland", r"\bVastra Gotaland\b": "Västra Götaland",
    r"\bVarmland\b": "Värmland", r"\bGavle\b": "Gävle", r"\bAngelholm\b": "Ängelholm",
}

def _lower_ascii(s): 
    return s.lower().replace("å","a").replace("ä","a").replace("ö","o").replace("é","e")

def normalize_user_query_spelling(q):
    q_norm=q; q_lc=_lower_ascii(q)
    for bad,good in INPUT_ALIASES.items():
        if bad in q_lc: q_norm=q_norm.replace(bad,good)
    return q_norm

def preserve_swedish_names(t):
    for pat,swe in NAME_REMAP.items(): t=re.sub(pat,swe,t)
    return t

def is_safe_input(text: str) -> bool:
    banned = ["sex","suicide","kill","weapon","hate","politics","religion","terrorism","drugs"]
    return not any(b in text.lower() for b in banned)

def sanitize_output(text: str) -> str:
    banned = ["kill","hate","suicide","weapon","drugs","terrorism"]
    if any(b in text.lower() for b in banned):
        return "I’m sorry, I can’t discuss that. Let’s talk about Sweden instead!"
    return text
