import re
import difflib

# Strong regex (covers most spellings)
BROCHURE_REGEX = re.compile(
    r"\b(brochure|brochure|brocher|brochur|brosher|brochere|broshure|brochure)\b",
    re.IGNORECASE
)

# list of correct forms for fuzzy match
BROCHURE_WORDS = ["brochure", "brochures", "broucher", "brocher", "brochur"]

def is_brochure_request(text: str):
    text_lower = text.lower()

    # Check regex first
    if BROCHURE_REGEX.search(text_lower):
        return True

    # Fuzzy fallback for heavy misspellings
    words = text_lower.split()
    for w in words:
        match = difflib.get_close_matches(w, BROCHURE_WORDS, n=1, cutoff=0.65)
        if match:
            return True

    return False
