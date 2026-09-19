"""Curated Croatian word/phrase lists backing the feature extractor (the specification, App. A).

Seed lists only: the AI-ism lexicon is replaced by an empirically discovered one in
M3+ (log-odds with Dirichlet prior); the ije/je and anglicism lists grow from corpus
work in M2. Every list is deliberately small and human-auditable — each entry must
be defensible at the oral defence.
"""

from __future__ import annotations

# High-frequency Croatian function words (prepositions, conjunctions, particles,
# pronouns, auxiliaries). Used for the content/function ratio.
FUNCTION_WORDS_HR = frozenset(
    """
    i a ali pa te ni niti ili jer da kad kada dok ako iako premda mada nego već
    u na za od do iz s sa o po pri kod bez kroz prema nad pod pred između iznad ispod
    uz oko protiv unatoč usprkos radi zbog tijekom nakon prije poslije
    ja ti on ona ono mi vi oni one ona sebe se sebi si
    me mene mi meni te tebe tebi ga njega mu njemu je ju nju joj njoj
    nas nam vas vam ih njih im njima njome njime
    moj tvoj njegov njezin njen naš vaš njihov svoj
    ovaj taj onaj ova ta ona ovo to ono ovi ti oni ove one
    tko što koji koja koje čiji kakav kolik
    sam si smo ste su bih bi bismo biste budem budeš bude budemo budete budu
    ću ćeš će ćemo ćete nisam nisi nije nismo niste nisu neću nećeš neće nećemo nećete
    ne li zar samo još čak već možda vrlo veoma tako kako ovako onako
    gdje kamo kuda odakle otkud ovdje ondje tu tamo sad sada tada onda uvijek nikad
    """.split()
)

# Obvious English loans (seed; assimilated borrowings deliberately excluded).
ANGLICISMS_SEED = frozenset(
    """
    menadžer menadžment marketing biznis brend event fashion lifestyle shopping šoping
    stage backstage feedback deadline meeting brainstorming startup startap
    online offline update upgrade download stream streaming influencer follower lajk
    lajkati hejter hejtati fajl mail mejl email selfie story storyji podcast
    """.split()
)

# AI-ism seed phrases (Appendix A). Lowercase; matched as substrings of the
# lowercased text. Replaced by the empirically discovered top-N in M3+.
AI_ISM_SEED_PHRASES: tuple[str, ...] = (
    "u zaključku",
    "važno je napomenuti",
    "važno je istaknuti",
    "s druge strane",
    "vrijedi istaknuti",
    "ključno je razumjeti",
    "u današnje vrijeme",
    "igra ključnu ulogu",
    "nudi mnoštvo",
    "zaronimo u",
    "u konačnici",
    "sve u svemu",
    "bitno je naglasiti",
)

# Discourse connectives whose sentence-initial over-use is an AI signal.
CONNECTIVE_STARTERS = frozenset(
    {"stoga", "nadalje", "također", "osim", "uz", "k", "zaključno", "naposljetku", "dodatno"}
)

# Known-wrong ije/je forms (human-error seed; correct forms in comments).
IJE_JE_WRONG_FORMS = frozenset(
    {
        "uvjek",  # uvijek
        "djete",  # dijete
        "ljepo",  # lijepo
        "ljepa",  # lijepa
        "htjeo",  # htio
        "vidjeo",  # vidio
        "letjeo",  # letio
        "donjeo",  # donio
        "razumjeo",  # razumio
        "uspjeo",  # uspio
        "cjena",  # cijena
    }
)

# Fused negated-conditional forms (human error; correct: "ne bi", "ne bih", ...).
NEBI_FORMS = frozenset({"nebi", "nebih", "nebismo", "nebiste"})

# "sa" is standard before s, š, z, ž (and before "mnom"); plain "s" elsewhere.
SA_ALLOWED_NEXT_INITIALS = ("s", "š", "z", "ž")
SA_ALLOWED_NEXT_WORDS = frozenset({"mnom", "psom", "pšenicom"})


# Fused *ne* + verb, which standard Croatian writes apart ("ne zna", not "nezna").
# Deliberately a curated list rather than a "starts with ne-" rule: several fusions
# are CORRECT — nemam/nemaš (ne+imati), neću/nećeš (ne+htjeti), nisam/nisi (ne+biti),
# nemoj — and *ne* is a genuine prefix on adjectives and nouns (nemoguć, nesretan,
# nezavisnost). A prefix rule would flag all of those as errors and invert the signal.
NE_VERB_FUSED_FORMS = frozenset(
    {
        "nezna",
        "neznam",
        "neznaš",
        "neznamo",
        "neznate",
        "neznaju",
        "nemogu",
        "nemože",
        "nemožeš",
        "nemožemo",
        "nemožete",
        "nevidim",
        "nevidiš",
        "nevidi",
        "nevidimo",
        "nerazumijem",
        "nerazumiješ",
        "nerazumije",
        "nepostoji",
        "netreba",
        "nemislim",
        "nesmijem",
        "nedam",
        "nekaže",
        "nepiše",
        "neradi",
        "neide",
        "nedolazi",
        "nežalim",
    }
)


# High-frequency English function words used to detect **language drift**, not
# anglicisms. Every entry is chosen to be unambiguous in Croatian: `a`, `i`, `to`, `on`,
# `do`, `no`, `sam`, `je`, `su`, `pa` are all common Croatian words and are excluded, so
# a Croatian sentence scores zero rather than a small false positive that would have to
# be tuned away later.
#
# Function words rather than content words on purpose: an anglicism ("marketing",
# "software") is normal in Croatian prose and carries no drift information, whereas
# "the", "of" and "which" appear only when the writer has switched language.
ENGLISH_DRIFT_WORDS = frozenset(
    {
        "the",
        "and",
        "of",
        "in",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "have",
        "has",
        "been",
        "which",
        "their",
        "there",
        "would",
        "could",
        "should",
        "about",
        "your",
        "you",
        "not",
        "but",
        "they",
        "its",
        "it",
        "is",
        "as",
        "by",
        "an",
        "or",
        "be",
        "at",
        # Instruction vocabulary: what an echoed English prompt looks like.
        "output",
        "note",
        "task",
        "text",
        "please",
        "following",
        "article",
        "write",
        "words",
    }
)
