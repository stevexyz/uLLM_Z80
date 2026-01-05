import random

# ---------- prompt pools ----------

greetings = [
    "hi", "hey", "hello", "yo", "hiya", "hey there", "hi there",
    "sup", "yo there", "hello there", "morning", "evening",
    "hey hi", "oh hi", "hey you"
]

states = [
    "im bored", "im tired", "im sad", "im happy", "im ok", "im fine",
    "im stressed", "im calm", "im excited", "im nervous", "im annoyed",
    "im relaxed", "im confused", "im lost", "im thinking", "im waiting",
    "im here", "im back", "im free", "im busy"
]

questions = [
    "how are you", "how you doing", "whats up", "what are you doing",
    "you ok", "you there", "still here", "are you busy", "got time",
    "can you talk", "can you help", "do you understand", "do you get it",
    "do you agree", "what do you think", "why", "how", "now what",
    "what next", "really"
]

reactions = [
    "ok", "okay", "sure", "yeah", "maybe", "i guess", "lol", "haha",
    "hmm", "oh", "wow", "right", "true", "cool", "nice", "alright",
    "fair enough", "that sucks", "sounds good", "sounds bad"
]

# ---------- answer pools ----------

acknowledgements = [
    "yeah", "ok", "sure", "alright", "i think so", "maybe", "not sure",
    "kinda", "a bit", "yeah maybe", "true", "same", "i guess",
    "probably", "could be", "who knows", "fair enough",
    "sounds right", "maybe later", "we will see"
]

emotional_mirrors = [
    "same here", "me too", "that sucks", "thats nice", "sounds good",
    "sounds rough", "i get it", "i hear you", "yeah i feel that",
    "that happens", "its ok", "no worries", "take it easy",
    "glad to hear that", "sorry to hear that", "makes sense",
    "i understand a bit", "yeah i know", "that feels right", "i get that"
]

presence_replies = [
    "im here", "yeah im here", "still here", "i am", "yep",
    "right here", "listening", "with you", "here now",
    "still talking", "not going anywhere", "im listening",
    "yeah", "present", "here yeah"
]

# ---------- generation rules ----------

pairs = []

# rule A: greeting -> greeting
for p in greetings:
    for a in random.sample(greetings, 5):
        pairs.append((p, a))

# rule B: greeting -> presence
for p in greetings:
    for a in random.sample(presence_replies, 10):
        pairs.append((p, a))

# rule C: state -> emotional mirror
for p in states:
    for a in random.sample(emotional_mirrors, 15):
        pairs.append((p, a))

# rule D: question -> acknowledgement
for p in questions:
    for a in random.sample(acknowledgements, 15):
        pairs.append((p, a))

# rule E: question -> presence
for p in questions:
    for a in random.sample(presence_replies, 10):
        pairs.append((p, a))

# rule F: reaction -> acknowledgement
for p in reactions:
    for a in random.sample(acknowledgements, 10):
        pairs.append((p, a))

# ---------- filtering ----------

clean_pairs = []
for p, a in pairs:
    if len(a) <= 30:
        clean_pairs.append(f"{p} | {a}")

random.shuffle(clean_pairs)

# ---------- output ----------

print(f"generated pairs: {len(clean_pairs)}\n")

for line in clean_pairs:
    print(line)
