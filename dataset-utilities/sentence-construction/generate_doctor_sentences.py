"""
Generate 1000 semantically valid ASL-grammar sentences from the 45-class doctor visit vocabulary.

ASL grammar rules applied:
- Topic-comment structure (topic first, comment after)
- Time signs at beginning (YESTERDAY, NOW, TOMORROW, etc.)
- WH-questions at the end (WHO, WHAT, WHEN, WHERE, WHY, HOW, WHICH)
- Adjectives after nouns
- No articles, copulas, or prepositions (ASL drops these)
"""

import random
import json
import os

# Full 45-class vocabulary
VOCAB = [
    "arm", "bad", "but", "day", "evening", "eye", "goodbye", "headache",
    "hearing aid", "heart", "hello", "his", "hospital", "hour", "how",
    "my", "neck", "need", "night", "now", "pain", "please", "pregnant",
    "recover", "right", "shoulder", "sick", "six", "sneeze", "stomach",
    "stress", "sunday", "their", "thermometer", "they", "tomorrow",
    "what", "wheelchair", "when", "where", "which", "who", "why",
    "yesterday", "your"
]

# Semantic groupings
TIME_SIGNS = ["now", "yesterday", "tomorrow", "day", "night", "evening", "sunday", "hour", "six"]
BODY_PARTS = ["arm", "neck", "shoulder", "stomach", "eye", "heart"]
SYMPTOMS = ["sick", "headache", "pain", "stress", "sneeze"]
MEDICAL = ["hospital", "thermometer", "wheelchair", "hearing aid", "recover", "pregnant"]
PEOPLE = ["they", "who"]
POSSESSIVES = ["my", "your", "his", "their"]
WH_QUESTIONS = ["what", "when", "where", "which", "who", "why", "how"]
DESCRIPTORS = ["bad", "right", "six"]
CONNECTORS = ["but", "please", "need"]
GREETINGS = ["hello", "goodbye"]

def generate_sentences():
    sentences = set()

    # ---- TEMPLATE CATEGORY 1: TIME + SYMPTOM statements ----
    # "YESTERDAY SICK" = "I was sick yesterday"
    for time in TIME_SIGNS:
        for symptom in SYMPTOMS:
            sentences.add(f"{time}, {symptom}")
        for poss in POSSESSIVES:
            for symptom in SYMPTOMS:
                sentences.add(f"{time}, {poss}, {symptom}")

    # ---- TEMPLATE 2: BODY PART + SYMPTOM ----
    # "ARM PAIN" = "My arm hurts"
    for body in BODY_PARTS:
        for symptom in SYMPTOMS:
            sentences.add(f"{body}, {symptom}")
        sentences.add(f"{body}, bad")
        for poss in POSSESSIVES:
            sentences.add(f"{poss}, {body}, {symptom}")
            sentences.add(f"{poss}, {body}, bad")

    # ---- TEMPLATE 3: TIME + BODY + SYMPTOM ----
    # "YESTERDAY ARM PAIN" = "Yesterday my arm hurt"
    for time in TIME_SIGNS:
        for body in BODY_PARTS:
            for symptom in SYMPTOMS:
                sentences.add(f"{time}, {body}, {symptom}")

    # ---- TEMPLATE 4: POSSESSIVE + BODY + SYMPTOM ----
    # "MY NECK PAIN" = "My neck hurts"
    for poss in POSSESSIVES:
        for body in BODY_PARTS:
            for symptom in SYMPTOMS:
                sentences.add(f"{poss}, {body}, {symptom}")
            sentences.add(f"{poss}, {body}, bad")

    # ---- TEMPLATE 5: TIME + POSSESSIVE + BODY + SYMPTOM ----
    # "YESTERDAY MY ARM PAIN BAD" = "Yesterday my arm pain was bad"
    for time in TIME_SIGNS:
        for poss in POSSESSIVES:
            for body in BODY_PARTS:
                for symptom in SYMPTOMS:
                    sentences.add(f"{time}, {poss}, {body}, {symptom}")
                    sentences.add(f"{time}, {poss}, {body}, {symptom}, bad")

    # ---- TEMPLATE 6: WH-questions about symptoms ----
    # "ARM PAIN WHEN" = "When does my arm hurt?"
    for body in BODY_PARTS:
        for wh in WH_QUESTIONS:
            sentences.add(f"{body}, {symptom}, {wh}")
            sentences.add(f"{body}, bad, {wh}")
        for symptom in SYMPTOMS:
            for wh in WH_QUESTIONS:
                sentences.add(f"{body}, {symptom}, {wh}")

    # ---- TEMPLATE 7: WH-questions about medical ----
    # "HOSPITAL WHERE" = "Where is the hospital?"
    for med in MEDICAL:
        for wh in WH_QUESTIONS:
            sentences.add(f"{med}, {wh}")

    # ---- TEMPLATE 8: POSSESSIVE + BODY + WH ----
    # "YOUR ARM PAIN WHY" = "Why does your arm hurt?"
    for poss in POSSESSIVES:
        for body in BODY_PARTS:
            for wh in WH_QUESTIONS:
                sentences.add(f"{poss}, {body}, {wh}")
            for symptom in SYMPTOMS:
                for wh in WH_QUESTIONS:
                    sentences.add(f"{poss}, {body}, {symptom}, {wh}")

    # ---- TEMPLATE 9: NEED + MEDICAL ----
    # "NEED HOSPITAL" = "I need to go to the hospital"
    for med in MEDICAL:
        sentences.add(f"need, {med}")
        for poss in POSSESSIVES:
            sentences.add(f"{poss}, need, {med}")
        sentences.add(f"need, {med}, now")
        sentences.add(f"need, {med}, please")
        sentences.add(f"need, {med}, when")
        sentences.add(f"need, {med}, where")

    # ---- TEMPLATE 10: TIME + NEED + MEDICAL ----
    for time in TIME_SIGNS:
        for med in MEDICAL:
            sentences.add(f"{time}, need, {med}")

    # ---- TEMPLATE 11: RECOVER questions/statements ----
    # "RECOVER WHEN" = "When will I recover?"
    for wh in WH_QUESTIONS:
        sentences.add(f"recover, {wh}")
    for time in TIME_SIGNS:
        sentences.add(f"recover, {time}")
        sentences.add(f"{time}, recover")
    for poss in POSSESSIVES:
        sentences.add(f"{poss}, recover, when")
        sentences.add(f"{poss}, recover, how")

    # ---- TEMPLATE 12: SICK + NEED + MEDICAL ----
    # "SICK NEED HOSPITAL" = "I'm sick, I need the hospital"
    for symptom in SYMPTOMS:
        for med in MEDICAL:
            sentences.add(f"{symptom}, need, {med}")
            sentences.add(f"{symptom}, need, {med}, please")
            sentences.add(f"{symptom}, need, {med}, now")

    # ---- TEMPLATE 13: Greeting + symptom/question ----
    # "HELLO MY ARM PAIN" = "Hello, my arm hurts"
    for greet in GREETINGS:
        for symptom in SYMPTOMS:
            sentences.add(f"{greet}, {symptom}")
        for poss in POSSESSIVES:
            for body in BODY_PARTS:
                sentences.add(f"{greet}, {poss}, {body}, bad")
                sentences.add(f"{greet}, {poss}, {body}, pain")

    # ---- TEMPLATE 14: THEY/WHO + medical ----
    # "THEY SICK" = "They are sick"
    for person in PEOPLE:
        for symptom in SYMPTOMS:
            sentences.add(f"{person}, {symptom}")
        for body in BODY_PARTS:
            sentences.add(f"{person}, {body}, bad")
            sentences.add(f"{person}, {body}, pain")
        for med in MEDICAL:
            sentences.add(f"{person}, need, {med}")

    # ---- TEMPLATE 15: BUT connector sentences ----
    # "SICK BUT RECOVER TOMORROW" = "I'm sick but I'll recover tomorrow"
    for symptom in SYMPTOMS:
        for time in TIME_SIGNS:
            sentences.add(f"{symptom}, but, recover, {time}")
            sentences.add(f"{symptom}, but, {time}, recover")
    for body in BODY_PARTS:
        sentences.add(f"{body}, bad, but, recover")
        sentences.add(f"{body}, pain, but, recover, tomorrow")
        sentences.add(f"{body}, pain, but, recover, now")

    # ---- TEMPLATE 16: Complex medical scenarios ----
    # "YESTERDAY NIGHT SICK NOW STOMACH BAD"
    for time in TIME_SIGNS:
        for body in BODY_PARTS:
            sentences.add(f"{time}, sick, now, {body}, bad")
            sentences.add(f"{time}, sick, {body}, pain")
            sentences.add(f"{time}, {body}, pain, now, bad")

    # ---- TEMPLATE 17: PLEASE + need/recover ----
    for med in MEDICAL:
        sentences.add(f"please, need, {med}")
        sentences.add(f"please, {med}, where")
    sentences.add(f"please, recover, when")
    sentences.add(f"please, recover, how")

    # ---- TEMPLATE 18: TIME + BODY + BAD + RECOVER + WH ----
    # "YESTERDAY ARM BAD RECOVER WHEN" = "My arm was bad yesterday, when will I recover?"
    for time in TIME_SIGNS:
        for body in BODY_PARTS:
            for wh in WH_QUESTIONS:
                sentences.add(f"{time}, {body}, bad, recover, {wh}")

    # ---- TEMPLATE 19: PREGNANT specific ----
    sentences.add("pregnant, when")
    sentences.add("pregnant, how")
    sentences.add("pregnant, hospital, when")
    sentences.add("pregnant, need, hospital")
    sentences.add("pregnant, need, hospital, please")
    sentences.add("pregnant, stomach, pain")
    sentences.add("pregnant, sick")
    sentences.add("pregnant, stress")
    sentences.add("pregnant, need, thermometer")
    for poss in POSSESSIVES:
        sentences.add(f"{poss}, pregnant")
        sentences.add(f"{poss}, pregnant, when")
        sentences.add(f"{poss}, pregnant, hospital, need")

    # ---- TEMPLATE 20: WHEELCHAIR specific ----
    sentences.add("need, wheelchair")
    sentences.add("need, wheelchair, please")
    sentences.add("need, wheelchair, now")
    sentences.add("wheelchair, where")
    sentences.add("wheelchair, when")
    for poss in POSSESSIVES:
        sentences.add(f"{poss}, need, wheelchair")
        sentences.add(f"{poss}, wheelchair, where")

    # ---- TEMPLATE 21: HEARING AID specific ----
    sentences.add("need, hearing aid")
    sentences.add("need, hearing aid, please")
    sentences.add("hearing aid, where")
    sentences.add("hearing aid, bad")
    for poss in POSSESSIVES:
        sentences.add(f"{poss}, hearing aid, bad")
        sentences.add(f"{poss}, need, hearing aid")

    # ---- TEMPLATE 22: HEART specific ----
    sentences.add("heart, bad")
    sentences.add("heart, pain")
    sentences.add("heart, stress")
    sentences.add("heart, pain, bad")
    sentences.add("heart, pain, hospital, need")
    for poss in POSSESSIVES:
        sentences.add(f"{poss}, heart, bad")
        sentences.add(f"{poss}, heart, pain")
        sentences.add(f"{poss}, heart, stress")

    # ---- TEMPLATE 23: SIX + TIME ----
    # "SIX HOUR" = "six hours"
    sentences.add("six, hour")
    sentences.add("six, day")
    sentences.add("sick, six, hour")
    sentences.add("sick, six, day")
    sentences.add("pain, six, hour")
    sentences.add("recover, six, hour")
    sentences.add("recover, six, day")
    for body in BODY_PARTS:
        sentences.add(f"{body}, pain, six, hour")
        sentences.add(f"{body}, pain, six, day")

    # ---- TEMPLATE 24: Double body part ----
    # "ARM SHOULDER PAIN" = "My arm and shoulder hurt"
    body_pairs = [
        ("arm", "shoulder"), ("arm", "neck"), ("neck", "shoulder"),
        ("stomach", "heart"), ("eye", "headache"), ("neck", "headache"),
    ]
    for b1, b2 in body_pairs:
        sentences.add(f"{b1}, {b2}, pain")
        sentences.add(f"{b1}, {b2}, bad")
        sentences.add(f"{b1}, {b2}, pain, when")
        sentences.add(f"{b1}, {b2}, pain, why")

    # ---- TEMPLATE 25: STRESS + BODY combinations ----
    for body in BODY_PARTS:
        sentences.add(f"stress, {body}, pain")
        sentences.add(f"stress, {body}, bad")
        sentences.add(f"{body}, stress, bad")
        for poss in POSSESSIVES:
            sentences.add(f"{poss}, {body}, stress")

    # ---- TEMPLATE 26: Multi-clause with BUT ----
    # "SICK BUT NEED HOSPITAL" = "I'm sick but I need the hospital"
    for symptom in SYMPTOMS:
        sentences.add(f"{symptom}, but, need, hospital")
        sentences.add(f"{symptom}, but, need, thermometer")
        sentences.add(f"{symptom}, but, recover, please")
    for body in BODY_PARTS:
        sentences.add(f"{body}, bad, but, need, hospital")

    # ---- TEMPLATE 27: SNEEZE specific ----
    sentences.add("sneeze, sick")
    sentences.add("sneeze, bad")
    sentences.add("sneeze, need, thermometer")
    sentences.add("sneeze, six, hour")
    sentences.add("sneeze, yesterday")
    sentences.add("sneeze, now")
    for poss in POSSESSIVES:
        sentences.add(f"{poss}, sneeze, bad")

    # ---- TEMPLATE 28: RIGHT + BODY (right arm, right eye, etc.) ----
    for body in BODY_PARTS:
        sentences.add(f"right, {body}, pain")
        sentences.add(f"right, {body}, bad")
        for poss in POSSESSIVES:
            sentences.add(f"{poss}, right, {body}, pain")
            sentences.add(f"{poss}, right, {body}, bad")
        for wh in WH_QUESTIONS:
            sentences.add(f"right, {body}, pain, {wh}")

    # ---- TEMPLATE 29: TIME + sick + RECOVER + TIME ----
    for t1 in ["yesterday", "now"]:
        for t2 in ["tomorrow", "sunday"]:
            sentences.add(f"{t1}, sick, recover, {t2}")
            sentences.add(f"{t1}, sick, recover, {t2}, when")
            sentences.add(f"{t1}, pain, recover, {t2}")

    # ---- TEMPLATE 30: EVENING/NIGHT/DAY variations ----
    for tod in ["evening", "night", "day"]:
        for symptom in SYMPTOMS:
            sentences.add(f"{tod}, {symptom}, bad")
        for body in BODY_PARTS:
            sentences.add(f"{tod}, {body}, pain")
            sentences.add(f"{tod}, {body}, bad")

    # ---- TEMPLATE 31: WHO + medical questions ----
    sentences.add("who, sick")
    sentences.add("who, pregnant")
    sentences.add("who, need, hospital")
    sentences.add("who, need, wheelchair")
    sentences.add("who, need, hearing aid")
    sentences.add("who, recover")
    for body in BODY_PARTS:
        sentences.add(f"who, {body}, pain")

    # ---- TEMPLATE 32: THERMOMETER specific ----
    sentences.add("need, thermometer")
    sentences.add("need, thermometer, please")
    sentences.add("need, thermometer, now")
    sentences.add("thermometer, where")
    sentences.add("thermometer, bad")
    for poss in POSSESSIVES:
        sentences.add(f"{poss}, need, thermometer")

    # ---- TEMPLATE 33: Longer 5-6 word combos ----
    for time in ["yesterday", "now", "tomorrow"]:
        for poss in POSSESSIVES:
            for body in BODY_PARTS:
                sentences.add(f"{time}, {poss}, {body}, pain, bad")
                sentences.add(f"{time}, {poss}, {body}, stress, bad")
                sentences.add(f"{time}, {poss}, right, {body}, pain")

    # ---- TEMPLATE 34: NEED + TIME + MEDICAL ----
    for time in TIME_SIGNS:
        sentences.add(f"need, hospital, {time}")
        sentences.add(f"need, thermometer, {time}")

    # ---- TEMPLATE 35: THEIR/HIS body pain ----
    for poss in ["his", "their"]:
        for body in BODY_PARTS:
            for symptom in SYMPTOMS:
                sentences.add(f"{poss}, {body}, {symptom}, bad")
            sentences.add(f"{poss}, {body}, bad, recover, when")

    # ---- TEMPLATE 36: WHICH questions ----
    sentences.add("which, hospital")
    sentences.add("which, thermometer")
    sentences.add("which, day, recover")
    sentences.add("which, arm, pain")
    sentences.add("which, eye, bad")
    sentences.add("sick, which, day")
    sentences.add("recover, which, day")
    sentences.add("hospital, which, where")

    return sentences


def validate_sentence(sentence_str):
    """Ensure all words are in vocabulary."""
    words = [w.strip() for w in sentence_str.split(",")]
    return all(w in VOCAB for w in words) and 2 <= len(words) <= 7


def main():
    random.seed(42)

    all_sentences = generate_sentences()

    # Filter valid sentences
    valid = sorted([s for s in all_sentences if validate_sentence(s)])

    print(f"Generated {len(valid)} valid sentences")

    # If we have more than 1000, sample
    if len(valid) > 1000:
        random.shuffle(valid)
        valid = sorted(valid[:1000])

    # Count by length
    lengths = {}
    for s in valid:
        n = len(s.split(","))
        lengths[n] = lengths.get(n, 0) + 1

    print("Sentence length distribution:")
    for k in sorted(lengths):
        print(f"  {k} words: {lengths[k]}")

    # Verify all vocab words used
    used_words = set()
    for s in valid:
        for w in s.split(","):
            used_words.add(w.strip())
    unused = set(VOCAB) - used_words
    if unused:
        print(f"WARNING: Unused vocabulary: {unused}")
    else:
        print("All 45 vocabulary words used")

    # Write output
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "doctor_visit_sentences_1000.csv")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("sentence_id,gloss_sequence,num_signs\n")
        for i, s in enumerate(valid, 1):
            n = len(s.split(","))
            f.write(f"{i},{s},{n}\n")

    print(f"\nSaved {len(valid)} sentences to {out_path}")

    # Also save as JSON
    json_path = os.path.join(out_dir, "doctor_visit_sentences_1000.json")
    records = []
    for i, s in enumerate(valid, 1):
        words = [w.strip() for w in s.split(",")]
        records.append({
            "id": i,
            "glosses": words,
            "num_signs": len(words)
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved JSON to {json_path}")


if __name__ == "__main__":
    main()
