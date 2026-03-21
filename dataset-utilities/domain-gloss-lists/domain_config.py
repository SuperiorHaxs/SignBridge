"""
Domain Gloss List Configuration

Parses SignBridge app domains and provides semantic candidate pools
for each domain from the WLASL 2000 gloss set.

Candidate pools are intentionally larger than target counts so the
diversity algorithm can select the most pose-distinct subset.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SIGNBRIDGE_TEMPLATE = PROJECT_ROOT / "applications" / "signbridge-app" / "templates" / "index.html"
WLASL_CLASS_INDEX = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "splits" / "2000_classes" / "class_index_mapping.json"
PICKLE_DIR = PROJECT_ROOT / "datasets" / "wlasl_poses_complete" / "pickle_files"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "domain-specific"

# =============================================================================
# DEFAULT COUNTS (overridable via CLI)
# =============================================================================

DEFAULT_COMMON_COUNT = 25
DEFAULT_DOMAIN_COUNT = 35


# =============================================================================
# SIGNBRIDGE DOMAIN PARSER
# =============================================================================

def parse_signbridge_domains(template_path: Path = SIGNBRIDGE_TEMPLATE) -> List[Dict]:
    """
    Parse SignBridge app's index.html to extract scenario domains.

    Returns list of dicts: [{"domain": "healthcare", "scenario": "Doctor Visit"}, ...]
    """
    if not template_path.exists():
        raise FileNotFoundError(f"SignBridge template not found: {template_path}")

    html = template_path.read_text(encoding="utf-8")

    # Match: selectScenario('healthcare', 'Doctor Visit')
    pattern = r"selectScenario\(\s*['\"](\w+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)"
    matches = re.findall(pattern, html)

    domains = []
    seen = set()
    for domain_key, scenario_name in matches:
        # Use scenario name as the unique key (multiple scenarios can share a domain_key)
        if scenario_name not in seen:
            seen.add(scenario_name)
            domains.append({
                "domain_key": domain_key,        # e.g. "healthcare", "generic"
                "scenario": scenario_name,        # e.g. "Doctor Visit", "Restaurant"
                "list_name": scenario_name.lower().replace(" ", "_"),  # filename-safe
            })

    return domains


def load_wlasl_glosses(index_path: Path = WLASL_CLASS_INDEX) -> set:
    """Load the full WLASL 2000 gloss set."""
    with open(index_path, "r") as f:
        mapping = json.load(f)
    return set(mapping.keys())


# =============================================================================
# SEMANTIC CANDIDATE POOLS
#
# Each domain has a curated list of WLASL 2000 glosses that are semantically
# relevant. The diversity algorithm picks the best N from each pool.
# Pools should be ~2x the target count to give the algorithm room to optimize.
# All glosses MUST exist in WLASL 2000 (validated at runtime).
# =============================================================================

# Glosses that MUST appear in the common list and MUST NOT appear in any domain list.
# These are reserved for common_words exclusively. The diversity algorithm fills
# the remaining common slots from COMMON_WORDS_CANDIDATES minus the forced set.
FORCED_COMMON = [
    # Question words (all)
    "WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW", "WHICH",
]

COMMON_WORDS_CANDIDATES = [
    # Pronouns
    "I", "YOU", "HE", "SHE", "WE", "THEY", "ME", "MY", "YOUR", "HER", "HIS", "OUR", "THEIR",
    "MYSELF", "YOURSELF", "THEMSELVES",
    # Question words (also in FORCED_COMMON — kept here so they appear in candidate pool too)
    "WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW", "WHICH",
    # Days of the week
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
    # Time
    "TODAY", "TOMORROW", "YESTERDAY", "NOW", "MORNING", "AFTERNOON", "EVENING", "NIGHT",
    "TIME", "WEEK", "MONTH", "YEAR", "DAY", "HOUR", "MINUTE",
    # Basic responses / politeness
    "YES", "NO", "PLEASE", "THANK YOU", "SORRY", "HELLO", "GOODBYE", "OK",
    # Common verbs
    "WANT", "NEED", "HAVE", "KNOW", "LIKE", "HELP", "GO", "COME", "SEE", "UNDERSTAND",
    # Numbers
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN",
    # Connectors
    "AND", "BUT", "OR", "NOT", "ALSO", "BECAUSE", "IF", "THEN",
    # Common adjectives
    "GOOD", "BAD", "BIG", "SMALL", "MORE", "SAME", "DIFFERENT",
]

DOMAIN_CANDIDATES = {
    "Doctor Visit": [
        "DOCTOR", "NURSE", "HOSPITAL", "MEDICINE", "SICK", "PAIN", "HEADACHE",
        "STOMACH", "BLOOD", "BREATHE", "COUGH", "ALLERGY", "SURGERY", "TEMPERATURE",
        "HEART", "INFECTION", "APPOINTMENT", "FEEL", "BETTER", "WORSE", "HURT",
        "HEALTH", "BODY", "ARM", "HEAD", "NECK", "SHOULDER", "BONE", "MUSCLE",
        "FEVER", "SNEEZE", "VOMIT", "DIZZY", "WEAK", "STRONG", "REST",
        "EAT", "DRINK", "SLEEP", "STRESS", "PREGNANT", "DIABETES", "PNEUMONIA",
        "SORE THROAT", "DIARRHEA", "HEART ATTACK", "THROAT", "SKIN", "EAR",
        "EYE", "TEETH", "MOUTH", "NOSE", "WHEELCHAIR", "HEARING AID",
        "THERMOMETER", "AMBULANCE", "EMERGENCY", "PATIENT", "SURGEON",
        "THERAPY", "RECOVER", "SWALLOW", "POISON", "OPERATE",
        "STAND", "SIT", "WALK", "STOP", "WAIT", "HERE", "WHERE", "HOW",
        "CAN", "WANT", "NEED", "KNOW", "RIGHT", "NAME", "ADDRESS",
        "ANSWER", "YES", "MORE", "MORNING", "NIGHT", "TODAY", "THANK YOU",
    ],
    "Restaurant": [
        "FOOD", "EAT", "DRINK", "WATER", "ORDER", "COOK", "DELICIOUS",
        "CHICKEN", "SALAD", "SOUP", "COFFEE", "TEA", "MILK", "JUICE",
        "BREAD", "CHEESE", "MEAT", "FISH", "RICE", "POTATO", "TOMATO",
        "FRUIT", "APPLE", "ORANGE", "BANANA", "STRAWBERRY", "GRAPES",
        "SALT", "PEPPER", "SUGAR", "BUTTER", "SAUCE", "CHOCOLATE",
        "BREAKFAST", "LUNCH", "DINNER", "DESSERT", "CAKE", "PIE", "COOKIE",
        "PIZZA", "HAMBURGER", "SANDWICH", "HOT DOG", "FRENCH FRIES",
        "RESTAURANT", "WAITER", "MENU", "TABLE", "CHAIR", "PLATE", "FORK",
        "KNIFE", "SPOON", "CUP", "BOWL", "GLASS", "NAPKIN",
        "HUNGRY", "THIRSTY", "FULL", "TASTE", "SWEET", "SOUR", "BITTER",
        "HOT", "COLD", "WANT", "PLEASE", "THANK YOU", "MORE", "ENOUGH",
        "BILL", "PAY", "TIP", "CHECK", "DOLLAR",
    ],
    "Shopping": [
        "BUY", "SELL", "MONEY", "COST", "PRICE", "EXPENSIVE", "CHEAP",
        "STORE", "SHOP", "SHOPPING", "CLOTHES", "SHOES", "SHIRT", "PANTS",
        "JACKET", "DRESS", "SKIRT", "HAT", "BELT", "BOOTS", "SOCKS",
        "SIZE", "COLOR", "RED", "BLUE", "GREEN", "BLACK", "WHITE", "YELLOW",
        "PINK", "PURPLE", "BROWN", "ORANGE", "GRAY",
        "BIG", "SMALL", "LONG", "SHORT", "NEW", "OLD", "BEAUTIFUL",
        "CHANGE", "EXCHANGE", "RETURN", "DISCOUNT", "CARD", "DOLLAR",
        "PAY", "CASH", "RECEIPT", "BAG", "BOX", "GIFT", "WRAP",
        "FIND", "LOOK FOR", "CHOOSE", "TRY", "LIKE", "WANT", "NEED",
        "HELP", "PLEASE", "THANK YOU", "HOW MUCH", "ENOUGH",
        "CUSTOMER", "WALLET", "JEWELRY", "RING", "NECKLACE", "BRACELET",
        "WATCH", "PERFUME",
    ],
    "Job Interview": [
        "WORK", "EXPERIENCE", "SKILL", "EDUCATION", "SCHOOL", "COLLEGE",
        "UNIVERSITY", "DEGREE", "CERTIFICATE", "GRADUATE", "STUDY", "LEARN",
        "TEACH", "SALARY", "MANAGER", "BOSS", "OFFICE", "COMPANY",
        "BUSINESS", "PROFESSIONAL", "INTERVIEW", "QUESTION", "ANSWER",
        "NAME", "ADDRESS", "TELL", "EXPLAIN", "DESCRIBE", "IMPROVE",
        "GOAL", "PLAN", "TEAM", "LEAD", "MANAGE", "RESPONSIBLE",
        "RESPONSIBILITY", "COMMUNICATE", "SCHEDULE", "MEET", "MEETING",
        "PROJECT", "PROBLEM", "SOLVE", "CREATE", "DEVELOP", "DECIDE",
        "AGREE", "COOPERATE", "COMPUTER", "TECHNOLOGY", "LANGUAGE",
        "STRONG", "HONEST", "HARD", "EASY", "IMPORTANT", "READY",
        "CONFIDENT", "START", "FINISH", "FULL", "PART", "HOUR", "WEEK",
        "YEAR", "THANK YOU", "PLEASE", "INTRODUCE",
    ],
    "Banking": [
        "MONEY", "BANK", "SAVE", "PAY", "DEPOSIT", "LOAN", "DOLLAR",
        "CHECK", "COST", "FINANCE", "INTEREST", "INVEST", "DEBT",
        "CENT", "NICKEL", "DIME", "QUARTER", "POUND", "ACCOUNT",
        "BALANCE", "TRANSFER", "EXCHANGE", "CREDIT", "CASH",
        "CARD", "RECEIPT", "PROFIT", "SALARY",
        "NUMBER", "TOTAL", "PERCENT", "INCREASE", "DECREASE", "DEDUCT",
        "ADD", "SUBTRACT", "CALCULATE", "AMOUNT",
        "INSURANCE", "BUDGET", "PURCHASE", "OWE", "BORROW", "LEND",
        "RICH", "POOR", "EXPENSIVE", "CHEAP", "FREE",
        "OPEN", "CLOSE", "SIGN", "FORM", "DOCUMENT", "IDENTIFY",
        "ADDRESS", "NAME", "HELP", "PLEASE", "THANK YOU", "NEED",
        "WANT", "HOW", "WHERE", "WHEN",
    ],
    "Emergency": [
        "HELP", "EMERGENCY", "POLICE", "FIRE", "HOSPITAL", "ACCIDENT",
        "HURT", "DANGER", "DANGEROUS", "CALL", "STOP", "GO", "RUN",
        "FAST", "QUICK", "NOW", "HERE", "WHERE", "AMBULANCE",
        "FIREFIGHTER", "COP", "DOCTOR", "NURSE",
        "BLOOD", "PAIN", "BREATHE", "ALIVE", "DEAD", "FALL IN LOVE",
        "BREAK", "CRASH", "FLOOD", "EARTHQUAKE", "HURRICANE", "TORNADO",
        "FIRE", "SMOKE", "ALARM", "SIREN",
        "SAFE", "SCARED", "AFRAID", "CAREFUL", "WARN", "ESCAPE",
        "RESCUE", "PROTECT", "HIDE", "WAIT", "COME", "COME HERE",
        "ADDRESS", "PHONE", "NAME", "NEED", "WANT", "CAN",
        "CHILD", "BABY", "MAN", "WOMAN", "FAMILY",
        "DOOR", "WINDOW", "STAIRS", "OUTSIDE", "INSIDE",
        "LEFT", "RIGHT", "UP", "DOWN", "NORTH", "SOUTH", "EAST", "WEST",
    ],
    "Education": [
        "SCHOOL", "CLASS", "CLASSROOM", "TEACH", "TEACHER", "STUDENT",
        "LEARN", "STUDY", "READ", "WRITE", "BOOK", "HOMEWORK", "TEST",
        "EXAM", "GRADE", "PASS", "FAIL", "GRADUATE", "GRADUATION",
        "COLLEGE", "UNIVERSITY", "DEGREE", "DIPLOMA", "CERTIFICATE",
        "MATH", "SCIENCE", "HISTORY", "ENGLISH", "ART", "MUSIC",
        "BIOLOGY", "CHEMISTRY", "PHYSICS", "GEOGRAPHY", "PSYCHOLOGY",
        "ALGEBRA", "GEOMETRY", "CALCULUS", "STATISTICS",
        "LIBRARY", "LIBRARIAN", "PENCIL", "PAPER", "ERASER", "DESK",
        "COMPUTER", "KEYBOARD", "PRINTER", "INTERNET",
        "LECTURE", "LESSON", "CHAPTER", "PARAGRAPH", "SENTENCE", "WORD",
        "VOCABULARY", "GRAMMAR", "SPELL", "DICTIONARY",
        "QUESTION", "ANSWER", "EXPLAIN", "EXAMPLE", "PRACTICE",
        "PROFESSOR", "PRINCIPAL", "TUTOR", "COACH",
        "ELEMENTARY", "HIGH SCHOOL", "KINDERGARTEN", "PRESCHOOL",
    ],
    "Travel": [
        "TRAVEL", "TRIP", "VACATION", "AIRPLANE", "TRAIN", "BUS", "CAR",
        "TAXI", "SUBWAY", "BICYCLE", "MOTORCYCLE", "BOAT", "SHIP",
        "TICKET", "PASSPORT", "LICENSE", "LUGGAGE", "BACKPACK",
        "AIRPORT", "ROAD", "HIGHWAY", "FREEWAY", "STREET",
        "CITY", "TOWN", "COUNTRY", "ISLAND", "MOUNTAIN", "BEACH",
        "OCEAN", "RIVER", "LAKE", "FOREST",
        "NORTH", "SOUTH", "EAST", "WEST", "LEFT", "RIGHT",
        "NEAR", "FAR", "ARRIVE", "LEAVE", "GO", "COME", "DRIVE", "WALK",
        "FLY", "RIDE", "STOP", "WAIT",
        "AMERICA", "EUROPE", "ASIA", "AFRICA", "AUSTRALIA",
        "FRANCE", "GERMANY", "JAPAN", "CHINA", "INDIA", "MEXICO",
        "ENGLAND", "ITALY", "SPAIN", "CANADA",
        "WEATHER", "HOT", "COLD", "RAIN", "SNOW", "SUN", "WIND",
        "CAMP", "CAMPING", "EXPLORE", "ADVENTURE",
    ],
    "Government Services": [
        "GOVERNMENT", "LAW", "VOTE", "PRESIDENT", "CONGRESS", "SENATE",
        "POLICE", "COURT", "JUDGE", "LAWYER", "ATTORNEY", "LEGAL",
        "LICENSE", "PERMIT", "FORM", "DOCUMENT", "SIGN", "CERTIFICATE",
        "RULE", "POLICY", "AUTHORITY", "OFFICIAL", "FEDERAL",
        "CITIZEN", "NATION", "CONSTITUTION", "DEMOCRAT", "REPUBLICAN",
        "ELECTION", "CANDIDATE", "REPRESENT", "COMMITTEE",
        "COMMUNITY", "PUBLIC", "SERVICE", "TAX", "REPORT",
        "ARREST", "JAIL", "PRISON", "GUILTY", "INNOCENT",
        "TESTIFY", "WITNESS", "EVIDENCE", "PENALTY",
        "NAME", "ADDRESS", "BIRTH", "AGE", "IDENTIFY",
        "APPLY", "REQUEST", "REQUIRE", "APPROVE", "DENY", "REFUSE",
        "WAIT", "HELP", "NEED", "PAY", "FREE",
        "MILITARY", "ARMY", "SOLDIER", "WAR", "PEACE",
        "INSURANCE", "WELFARE", "BENEFIT",
    ],
}


def get_validated_candidates(
    domain_name: str,
    wlasl_glosses: set,
) -> List[str]:
    """
    Return candidate glosses for a domain, filtered to only those
    that actually exist in WLASL 2000.
    """
    candidates = DOMAIN_CANDIDATES.get(domain_name, [])
    valid = [g for g in candidates if g in wlasl_glosses]
    invalid = [g for g in candidates if g not in wlasl_glosses]
    if invalid:
        print(f"  [{domain_name}] {len(invalid)} candidates not in WLASL 2000: {invalid[:10]}{'...' if len(invalid) > 10 else ''}")
    return valid


def get_validated_common_candidates(wlasl_glosses: set) -> List[str]:
    """Return common word candidates filtered to WLASL 2000."""
    valid = [g for g in COMMON_WORDS_CANDIDATES if g in wlasl_glosses]
    invalid = [g for g in COMMON_WORDS_CANDIDATES if g not in wlasl_glosses]
    if invalid:
        print(f"  [common] {len(invalid)} candidates not in WLASL 2000: {invalid[:10]}{'...' if len(invalid) > 10 else ''}")
    return valid
