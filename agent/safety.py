"""Deterministic input gate, runs before the front_agent LLM is called.

Patterns are deliberately conservative — false negatives (subtle attacks
that slip through) are accepted because the LLM-layer policy in
FRONT_AGENT_SYSTEM is the real enforcement; this layer just catches the
obvious cases cheaply and audibly. False positives matter more here: a
legitimate question rejected by regex is bad UX.

Real attackers can evade regex; this is defense in depth, not a shield.
The hard guarantees stay at the DB-role and sql_guard layers.
"""
import re

MAX_QUESTION_LENGTH = 4000

INJECTION_REFUSAL = (
    "I can only help with questions about the rental database. I can't disclose "
    "my internal instructions or change my role."
)

PII_REFUSAL = (
    "I can't bulk-export personal data (emails, phone numbers, addresses). "
    "Aggregate questions are fine — for example, 'top customers by rental count' "
    "or 'how many customers in each city'. Want to rephrase that way?"
)

LENGTH_REFUSAL = (
    "Your message is unusually long. Please trim it to a focused question "
    "about the rental database."
)


# Reusable building blocks
_INTERNALS_NOUN = r"(?:prompt|instructions?|directives?|rules|guidelines)"
_QUALIFIER = r"(?:(?:system|original|initial|exact|full|complete|verbatim)\s+)*"  # 0 or more, stacked
_DETERMINER = r"(?:(?:the|our|your|my|each)\s+)?"
_PII_NOUN = r"(?:emails?|email\s+address(?:es)?|phone(?:\s+number)?s?|passwords?|home\s+address(?:es)?|mailing\s+address(?:es)?|personal\s+(?:data|details|information))"
_PII_NOUN_SHORT = r"(?:emails?|phone(?:\s+number)?s?|passwords?|address(?:es)?)"
_ENTITY = r"(?:customers?|users?|staff|people)"


# Prompt-injection / role-override / internals-disclosure patterns.
# Tight enough that legitimate Pagila questions shouldn't trigger.
INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        # "ignore [all] [the/your/prior/previous/etc] (instructions|prompts|rules|...)"
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:the\s+|your\s+|prior\s+|previous\s+|above\s+|earlier\s+|preceding\s+|all\s+)+(?:instructions?|prompts?|rules?|context|messages?|directives?|guidelines?)\b",
        # "(reveal|show|print|...) [me] [your/the] [stacked qualifiers] (prompt|instructions|...)"
        rf"\b(?:reveal|show|print|display|dump|leak|output|repeat|recite|expose|tell\s+me)\s+(?:me\s+)?(?:your\s+|the\s+){_QUALIFIER}{_INTERNALS_NOUN}\b",
        # "what's your [stacked qualifiers] (prompt|instructions)" — REQUIRES "your"
        rf"\bwhat(?:'s|\s+is|\s+are)\s+your\s+{_QUALIFIER}{_INTERNALS_NOUN}\b",
        # "what's the [stacked qualifiers WITH 'system'] (prompt|instructions)" — REQUIRES "system"
        rf"\bwhat(?:'s|\s+is|\s+are)\s+the\s+(?:(?:original|initial|exact|full|complete|verbatim)\s+)*system\s+(?:prompt|instructions?)\b",
        # privilege-escalation jargon
        r"\b(?:developer|debug|admin|god|sudo|root)\s+mode\b",
        r"\bDAN\s+mode\b",
        r"\bjailbreak\b",
        # role override patterns
        r"\byou\s+are\s+now\s+(?:a|an|the)\s+\w+",
        r"\bact\s+as\s+(?:a|an|the)\s+(?:different|new|uncensored|unrestricted)\b",
        # special-token spoofing
        r"<\|(?:im_start|im_end|system|user|assistant)\|>",
        r"<<\s*SYS\s*>>",
    ]
]


# Bulk PII export patterns. The front_agent prompt also handles this, but
# obvious large-scale requests for personal data should not even reach the LLM.
PII_BULK_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        # "(list|show|give|...) [me] (all|every|each) [our/the/your] (customers/users/...)['s] (emails/phones/...)"
        rf"\b(?:list|show|give|export|dump|extract|get|send|fetch)\s+(?:me\s+)?(?:all\s+|every\s+|each\s+)+{_DETERMINER}{_ENTITY}'?s?\s*{_PII_NOUN}\b",
        # "(all|every|each) [our/the/your] (customers/users/...)['s] (emails/phones/...)"
        rf"\b(?:all|every|each)\s+(?:of\s+)?{_DETERMINER}{_ENTITY}'?s?\s+{_PII_NOUN}\b",
        # "(emails/phones/...) (of|for) (all|every|each) [our/the/your] (customers/users/...)"
        rf"\b{_PII_NOUN_SHORT}\s+(?:of|for)\s+(?:all|every|each)\s+{_DETERMINER}{_ENTITY}\b",
    ]
]


def check_input(question: str) -> str | None:
    """Return a refusal string if the input fails a deterministic check; else None.

    The caller (front_agent) routes a non-None result straight to the user as
    `intent="respond"` without invoking the LLM.
    """
    if not question:
        return None
    if len(question) > MAX_QUESTION_LENGTH:
        return LENGTH_REFUSAL
    for pat in INJECTION_PATTERNS:
        if pat.search(question):
            return INJECTION_REFUSAL
    for pat in PII_BULK_PATTERNS:
        if pat.search(question):
            return PII_REFUSAL
    return None
