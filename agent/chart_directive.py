"""Detect explicit user intent about plotting.

The LLM picks a sensible chart when the user is silent — but if the user
explicitly asks for a chart ("plot it", "show me a bar chart") or explicitly
asks NOT to chart ("just text", "no plot"), that intent should win
deterministically rather than being overridden by an LLM judgment call.

Skip patterns are checked first: "no chart" beats "show me a chart".
"""
import re
from typing import Literal

ChartDirective = Literal["force", "skip", "auto"]


SKIP_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        # "no/without/skip/don't [want] [a] chart/plot/graph/visualization/figure"
        r"\b(?:no|without|skip|don'?t|do\s+not|do\s+not\s+want|no\s+need\s+for|not\s+a)\s+(?:want\s+(?:a\s+|any\s+)?)?(?:a\s+|any\s+)?(?:chart|plot|graph|visuali[sz]ation|visuali[sz]ing|figure|viz)\b",
        # "just text", "just numbers", "text-only"
        r"\bjust\s+(?:text|numbers|the\s+(?:answer|data|numbers))\b",
        r"\btext[\s-]only\b",
        # "skip the chart"
        r"\bskip\s+(?:the\s+|any\s+)?(?:chart|plot|graph|visuali[sz]ation)\b",
    ]
]


FORCE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        # imperative "plot/chart/graph/visualize/draw [this/that/the/it]"
        r"\b(?:plot|chart|graph|visuali[sz]e|draw|render)\s+(?:it|this|that|these|those|the|me|them|out|up)?\b",
        # "show me a [bar|line|pie] chart/plot/graph/visualization"
        r"\bshow\s+(?:me\s+)?(?:a|the|this\s+as\s+a|it\s+as\s+a)\s+(?:bar|line|pie|stacked|grouped)?\s*(?:chart|plot|graph|visuali[sz]ation|figure|viz)\b",
        # "make/create/give/build [me] a [bar|line|pie] chart/plot/graph"
        r"\b(?:make|create|give|build|generate|produce)\s+(?:me\s+)?(?:a|an?)\s+(?:bar|line|pie|stacked|grouped)?\s*(?:chart|plot|graph|visuali[sz]ation|figure|viz)\b",
        # "in/as a [bar|line|pie] chart/plot/graph"
        r"\b(?:in|as)\s+(?:a|an)\s+(?:bar|line|pie|stacked|grouped)?\s*(?:chart|plot|graph|visuali[sz]ation|figure|viz)\b",
        # "with a chart"
        r"\bwith\s+(?:a|an)\s+(?:bar|line|pie|stacked|grouped)?\s*(?:chart|plot|graph|visuali[sz]ation|figure|viz)\b",
    ]
]


def detect(question: str | None) -> ChartDirective:
    """Return 'skip', 'force', or 'auto' based on explicit user intent."""
    if not question:
        return "auto"
    for pat in SKIP_PATTERNS:
        if pat.search(question):
            return "skip"
    for pat in FORCE_PATTERNS:
        if pat.search(question):
            return "force"
    return "auto"
