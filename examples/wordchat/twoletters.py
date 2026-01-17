import sys

def chunk_word(word, chunk_size=2):
    """Split a word into chunks of `chunk_size` letters."""
    chunks = [word[i:i + chunk_size] for i in range(0, len(word), chunk_size)]
    return chunks


def main():
    words = sys.stdin.read().split()

    unique_chunks = set()
    total_chunks = 0

    print("Word -> 2-letter chunks")
    print("-" * 40)

    for word in words:
        word_lower = word.lower()
        chunks = chunk_word(word_lower, 2)
        total_chunks += len(chunks)
        unique_chunks.update(chunks)

        print(f"{word} -> {'-'.join(chunks)}")

    print("\nUnique chunks:")
    print("-" * 40)
    for c in sorted(unique_chunks):
        print(c)

    print("\nStatistics:")
    print("-" * 40)
    print(f"Total words: {len(words)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Unique chunks: {len(unique_chunks)}")



if __name__ == "__main__":
    main()
