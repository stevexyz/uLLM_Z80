import sys

VOWELS = "aeiouy"


def is_vowel(c):
    return c.lower() in VOWELS


def syllabify(word: str):
    word = word.lower()
    if not word.isalpha():
        return [word]

    syllables = []
    start = 0
    i = 0

    while i < len(word):
        # Move to next vowel
        while i < len(word) and not is_vowel(word[i]):
            i += 1
        if i >= len(word):
            break

        # Consume vowel group
        j = i
        while j + 1 < len(word) and is_vowel(word[j + 1]):
            j += 1

        # Count following consonants
        k = j + 1
        consonants = 0
        while k < len(word) and not is_vowel(word[k]):
            consonants += 1
            k += 1

        # Syllable boundary rules
        if consonants <= 1:
            end = j + 1
        else:
            end = j + 2

        syllable = word[start:end]

        # Split syllable if > 3 chars
        while len(syllable) > 3:
            syllables.append(syllable[:3])
            syllable = syllable[3:]
        syllables.append(syllable)

        start = end
        i = end

    # Append leftover characters
    if start < len(word) and syllables:
        leftover = word[start:]
        while len(leftover) > 3:
            syllables.append(leftover[:3])
            leftover = leftover[3:]
        syllables.append(leftover)

    return syllables


def main():
    words = sys.stdin.read().split()

    unique_syllables = set()
    total_syllables = 0

    print("Syllabification (max 3 letters per syllable):")
    print("-" * 50)

    for word in words:
        syllables = syllabify(word)
        total_syllables += len(syllables)
        unique_syllables.update(syllables)

        print(f"{word} -> {'-'.join(syllables)}")

    print("\nStatistics:")
    print("-" * 50)
    print(f"Total words: {len(words)}")
    print(f"Total syllables: {total_syllables}")
    print(f"Unique syllables: {len(unique_syllables)}")

    print("\nUnique syllables:")
    print("-" * 50)
    for s in sorted(unique_syllables):
        print(s)


if __name__ == "__main__":
    main()
