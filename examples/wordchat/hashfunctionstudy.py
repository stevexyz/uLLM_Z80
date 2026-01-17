import sys
import struct

def hash(data: bytes) -> int:

    crc = 0
    for byte in data:
        crc ^= byte
        crc = crc * 13
    crc &= 0x0FFF  # keep 12 bits
    return crc

"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
    crc &= 0x07FF  # keep 11 bits
    return crc
"""
"""
    length = len(data)
    nblocks = length // 4

    h1 = 0 # seed
    c1 = 0xcc9e2d51
    c2 = 0x1b873593

    # body
    for block_start in range(0, nblocks * 4, 4):
        k1 = struct.unpack_from("<I", data, block_start)[0]

        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF

        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xe6546b64) & 0xFFFFFFFF

    # tail
    tail = data[nblocks * 4:]
    k1 = 0

    if len(tail) == 3:
        k1 ^= tail[2] << 16
    if len(tail) >= 2:
        k1 ^= tail[1] << 8
    if len(tail) >= 1:
        k1 ^= tail[0]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1

    # finalization
    h1 ^= length
    h1 ^= (h1 >> 16)
    h1 = (h1 * 0x85ebca6b) & 0xFFFFFFFF
    h1 ^= (h1 >> 13)
    h1 = (h1 * 0xc2b2ae35) & 0xFFFFFFFF
    h1 ^= (h1 >> 16)

    return h1 & 0x0FFF # keep 12 bits
"""

def main():
    hashes = {}
    total_words = 0
    collisions = 0

    # Read all standard input, and split in single words
    words = sys.stdin.read().split()

    for word in words:
        total_words += 1
        h = hash(word.encode("ascii"))

        if h in hashes:
            collisions += 1
            hashes[h].append(word)
        else:
            hashes[h] = [word]

    print(f"Total words: {total_words}")
    print(f"Unique hashes: {len(hashes)}")
    print(f"Collisions: {collisions}")

    print("\nCollision details:")
    for h, ws in hashes.items():
        if len(ws) > 1:
            print(f"Hash {h:04x}: {ws}")

if __name__ == "__main__":
    main()
