#!/usr/bin/env python3
"""Rewrite placeholder gloss-join sentences to natural English."""

import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "threshold_experiment_dataset.json"

# Hand-written natural English sentences for each gloss combination.
# Rules:
#   - Use ALL glosses in the sentence (in any grammatical order)
#   - Add articles, prepositions, pronouns as needed for natural English
#   - Keep sentences simple and conversational
#   - Relational nouns (son, family, man) get "my" or "the"

SENTENCES = {
    # ═══ BAND A: 0-33% Top-3 ═══
    1000: "What did the cow do at the birthday dance on the table?",
    1001: "The birthday table was fine, but we ate later when it was full.",
    1002: "What birthday party had a full table later?",
    1003: "The birthday dance had a full table and a cow.",
    1004: "What dance are we doing later?",
    1005: "Tell me what dance is at the table later.",
    1006: "The cow walked to the computer for the birthday dance.",
    1007: "What dance is fine?",
    1008: "The cow was later at the full table, but what happened?",
    1009: "What computer did the cow walk to for the birthday?",
    1010: "I need the table later.",
    1011: "The cow danced on the full table.",
    1012: "No, the cow comes later.",
    1013: "The cow walked to the Thanksgiving birthday table.",
    1014: "The full Thanksgiving birthday dance came later.",
    1015: "The cow was full after the dance.",
    1016: "What dance did the cow do later on the table?",
    1017: "The cow danced and I will tell you.",
    1018: "I need to dance later for the full birthday.",
    1019: "What full birthday table will we walk to?",
    1020: "The fine cow was full later after the dance.",
    1021: "What full dance is on the table?",
    1022: "The birthday dance was on the computer.",
    1023: "The fine table had a cow and a birthday cake later.",
    1024: "Tell me what the full cow did at the dance.",
    1025: "The full Thanksgiving birthday had a cow and a computer.",
    1026: "What birthday is at the table later?",
    1027: "What fine cow walked to the birthday?",
    1028: "The fine full dance was amazing.",
    1029: "What computer did we use later for the Thanksgiving birthday?",

    # ═══ BAND B: 33-50% Top-3 ═══
    1030: "No, what is full?",
    1031: "The cow from Africa is full.",
    1032: "I need to study what is full, no?",
    1033: "Tell the tall man about the birthday cow.",
    1034: "The birthday party is hot later.",
    1035: "The birthday was wrong, but it happened later.",
    1036: "My family had a birthday before the cow was on the table.",
    1037: "The orange jacket was full of birthday gifts later.",
    1038: "I need to dance on Thursday at the full table.",
    1039: "I will get the jacket later from the cow.",
    1040: "The birthday doctor needs to see the cow.",
    1041: "What table in Africa did my son see later?",
    1042: "I need paper for the dance later, not the cow.",
    1043: "The full birthday dance was to play basketball.",
    1044: "What deaf Thanksgiving cow is that?",
    1045: "Africa had a full table before, but later it was empty.",
    1046: "Tell me what is fine, then give me the dance later.",
    1047: "My fine son needs to know what the birthday is about.",
    1048: "But what is wrong with the table and the cow?",
    1049: "The cow was full, but I need to dance.",
    1050: "I need the cow, it is full.",
    1051: "The birthday table is fine.",
    1052: "What birthday did the deaf person walk to?",
    1053: "What computer did we use before the later dance?",
    1054: "Give me the birthday present later.",
    1055: "My family asked what the birthday was about later.",
    1056: "The man walked to the full birthday later.",
    1057: "It was wrong to dance later on Thanksgiving, but what happened?",
    1058: "What Thanksgiving doctor was full after the dance?",
    1059: "Tell the cow about the full orange on the table.",

    # ═══ BAND C: 50-67% Top-3 ═══
    1060: "The hot dance was fine, so give me time.",
    1061: "The hot jacket is on the table next to the cow.",
    1062: "I will play the Thanksgiving paper game later.",
    1063: "The fine bowling walk in Africa was a real dance.",
    1064: "The full orange is on the table by the chair.",
    1065: "We give thanks for studying the cow on Thanksgiving.",
    1066: "The cow is fine and full this time after studying.",
    1067: "The birthday time was full of fine help.",
    1068: "It was fine, but later it got full and the wrong tall one came.",
    1069: "The table walk on Thursday was in a jacket for the dance.",
    1070: "The table change was made by the doctor.",
    1071: "The computer change came later with bowling.",
    1072: "No, give the cow a computer drink.",
    1073: "No, what family is that?",
    1074: "What hot place in Africa? No, it came later.",
    1075: "The man read the paper about what time the table was ready.",
    1076: "The birthday was full of family drinks.",
    1077: "Tell the man about the cow from Africa, it is full.",
    1078: "But no, what is going on?",
    1079: "The cow danced to help with drinks.",
    1080: "The full basketball needs water, so what happened?",
    1081: "The Thanksgiving basketball game was on Thursday with the cow.",
    1082: "The computer paper is about what?",
    1083: "Thanksgiving was full before.",
    1084: "I enjoy the Thursday dance on the table.",
    1085: "The orange cow needs to know what happened before.",
    1086: "The basketball and orange are on the table with the cow.",
    1087: "Walk to the table before the time comes later.",
    1088: "Tell the drink computer to play the dance.",
    1089: "The table dance was a year of water.",

    # ═══ BAND D: 67-83% Top-3 ═══
    1090: "The fine tall jacket is for play.",
    1091: "No, the orange water is not tall.",
    1092: "My son needs to change the birthday year with help.",
    1093: "I walk, but the hot drink is ready.",
    1094: "The year of dance was in the study paper.",
    1095: "Give the cow a change to play tall basketball.",
    1096: "The hot time in Africa was helped by Thanksgiving.",
    1097: "My fine son enjoys the doctor and orange.",
    1098: "The deaf doctor helped at the table.",
    1099: "The tall man was fine before the drink.",
    1100: "The orange jacket is on the table on Thursday.",
    1101: "I enjoy the doctor, but the cow does not.",
    1102: "The birthday doctor is my man with an apple.",
    1103: "I walk to the orange in Africa.",
    1104: "No, the apple drink is fine, but I want it.",
    1105: "The hot basketball and orange? No, it is for the birthday.",
    1106: "The man is fine before.",
    1107: "Help the doctor, but no wrong moves later.",
    1108: "I play at the table with a chair on Thursday before.",
    1109: "But the man has a cow.",
    1110: "No, I need help to study.",
    1111: "The birthday change was to study with the doctor.",
    1112: "The change for the deaf on Thanksgiving.",
    1113: "The man changed the computer to play.",
    1114: "The computer time is fine, but it is not ready.",
    1115: "The computer study helps the doctor.",
    1116: "The computer paper has an apple this year.",
    1117: "My son changed the Africa basketball dance.",
    1118: "The hot jacket on Thursday is for the birthday, but it is fine.",
    1119: "I enjoy walking and giving hot basketball.",

    # ═══ BAND E: 83-100% Top-3 ═══
    1120: "I drink hot water and give time to study.",
    1121: "The computer helps with drinking and basketball.",
    1122: "My son in Africa walked in a water jacket.",
    1123: "The jacket on Thursday is to enjoy the paper.",
    1124: "The man is deaf and sits in the chair.",
    1125: "Africa is hot before the Thanksgiving play.",
    1126: "Give the man from Africa a change to play basketball.",
    1127: "The deaf doctor studied the jacket on Thanksgiving.",
    1128: "My family told the bowling paper about the chair.",
    1129: "The deaf person needs water, time, and a jacket.",
    1130: "Tell my hot and wrong son.",
    1131: "My family drinks and helps the man on Thursday.",
    1132: "The wrong change was on Thursday, no.",
    1133: "My son's family has water for Thanksgiving time.",
    1134: "I was wrong to drink before, so tell me.",
    1135: "I give a tall bowling ball.",
    1136: "The hot deaf time was important.",
    1137: "My son was wrong about the drink.",
    1138: "I study and play hot Thursday every year.",
    1139: "The doctor changed my son's play.",
    1140: "I study, but I help.",
    1141: "I drink apple juice this time.",
    1142: "I give the orange jacket and change the bowling.",
    1143: "My son walked with paper for the family.",
    1144: "My deaf son plays and helps.",
    1145: "I enjoy fine basketball and help.",
    1146: "The doctor was wrong, but my family helps with the chair.",
    1147: "I study the orange and need a doctor.",
    1148: "My wrong son went to Africa on Thursday for an orange.",
    1149: "I walk and play to change and enjoy.",
}

def main():
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    updated = 0
    missing = []

    for entry in dataset:
        eid = entry['id']
        if eid in SENTENCES:
            entry['sentence'] = SENTENCES[eid]
            updated += 1
        else:
            missing.append(eid)

    if missing:
        print(f"WARNING: {len(missing)} entries without sentences: {missing}")

    with open(DATASET_PATH, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Updated {updated}/{len(dataset)} sentences in {DATASET_PATH.name}")

    # Verify all glosses appear in their sentence (case-insensitive)
    issues = []
    for entry in dataset:
        sentence_lower = entry['sentence'].lower()
        for gloss in entry['glosses']:
            # Check if gloss word appears in sentence
            g = gloss.lower()
            # Handle multi-word glosses like "thank you"
            if g == "thanksgiving":
                check = "thanksgiv"
            elif g == "birthday":
                check = "birthday"
            else:
                check = g
            if check not in sentence_lower:
                issues.append((entry['id'], gloss, entry['sentence']))

    if issues:
        print(f"\nWARNING: {len(issues)} glosses not found in their sentence:")
        for eid, gloss, sent in issues[:20]:
            print(f"  {eid}: '{gloss}' not in \"{sent}\"")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("All glosses verified present in their sentences.")


if __name__ == "__main__":
    main()
