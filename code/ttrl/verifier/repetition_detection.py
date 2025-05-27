import os
import json
from pathlib import Path


# def detect_repetition_rolling_hash(text, min_rep_length=3, max_rep_length=50, min_repeats=2, remove_repeats=False):
#     """
#     Detect repetitions using rolling hash (Rabin-Karp algorithm).
    
#     Args:
#         text (str): The text to analyze for repetitions
#         min_rep_length (int): Minimum length of repeating pattern to detect
#         max_rep_length (int): Maximum length of repeating pattern to check
#         min_repeats (int): Minimum number of times a pattern must repeat to be flagged
        
#     Returns:
#         list: List of tuples containing (pattern, start_index, end_index, repetition_count)
#     """
#     def rolling_hash(s):
#         """Compute rolling hash values for all substrings of length pattern_length."""
#         p = 31  # Prime number for hash calculation
#         m = 10**9 + 9  # Modulo to avoid overflow
#         n = len(s)
        
#         # Precompute powers of p
#         p_pow = [1]
#         for i in range(1, n + 1):
#             p_pow.append((p_pow[-1] * p) % m)
        
#         # Compute prefix hashes
#         h = [0] * (n + 1)
#         for i in range(n):
#             h[i + 1] = (h[i] + (ord(s[i]) - ord('a') + 1) * p_pow[i]) % m
        
#         return h, p_pow, p, m
    
#     def get_substring_hash(h, p_pow, start, length, m):
#         """Get hash value of a substring using precomputed hash values."""
#         return (h[start + length] - h[start] + m) % m
    
#     results = []
#     n = len(text)
    
#     # Compute hash values for entire text
#     hashes, p_powers, p, m = rolling_hash(text)
    
#     # For each potential pattern length
#     for pattern_length in range(min_rep_length, min(max_rep_length + 1, n // 2 + 1)):
#         # Hash table to store positions of each hash value
#         hash_positions = {}
        
#         # Check each possible position
#         for start in range(n - pattern_length + 1):
#             curr_hash = get_substring_hash(hashes, p_powers, start, pattern_length, m)
            
#             # If we've seen this hash before, check if it's a genuine repetition
#             if curr_hash in hash_positions:
#                 for prev_start in hash_positions[curr_hash]:
#                     # Verify the strings match (to handle hash collisions)
#                     if text[prev_start:prev_start + pattern_length] == text[start:start + pattern_length]:
#                         # Check if these form a consecutive repetition
#                         if prev_start + pattern_length == start:
#                             # Count how many times this pattern repeats consecutively
#                             pattern = text[prev_start:prev_start + pattern_length]
#                             pos = start + pattern_length
#                             count = 2  # We already found 2 occurrences
                            
#                             while pos + pattern_length <= n and text[pos:pos + pattern_length] == pattern:
#                                 count += 1
#                                 pos += pattern_length
                            
#                             if count >= min_repeats:
#                                 start_pos = prev_start
#                                 end_pos = start_pos + (pattern_length * count)
#                                 results.append((pattern, start_pos, end_pos, count))
#                                 break
            
#             # Add current position to hash table
#             if curr_hash not in hash_positions:
#                 hash_positions[curr_hash] = []
#             hash_positions[curr_hash].append(start)
    
#     # Remove overlapping results, keeping the longest ones
#     results.sort(key=lambda x: (x[1], -(x[2] - x[1])))
#     filtered_results = []
    
#     if results:
#         filtered_results.append(results[0])
#         for i in range(1, len(results)):
#             current = results[i]
#             prev = filtered_results[-1]
            
#             # Check if current repetition overlaps with previous one
#             if current[1] >= prev[2]:  # Current starts after previous ends
#                 filtered_results.append(current)
    
#     return filtered_results



def detect_repetition(text, min_rep_length=10, max_rep_length=256, min_repeats=10):
    """
    Detect repetitions in text with variable length patterns.
    Returns immediately after finding the first repetition.
    
    Args:
        text (str): The text to analyze for repetitions
        min_rep_length (int): Minimum length of repeating pattern to detect
        max_rep_length (int): Maximum length of repeating pattern to check
        min_repeats (int): Minimum number of times a pattern must repeat to be flagged
        
    Returns:
        list: List containing a single tuple (pattern, start_index, end_index, repetition_count) if repetition found,
              empty list if no repetition found
    """
    text_length = len(text)
    
    # For each potential pattern length
    for pattern_length in range(min_rep_length, min(max_rep_length + 1, text_length // 2 + 1)):
        # Check each possible starting position
        for start in range(text_length - pattern_length * min_repeats + 1):
            pattern = text[start:start + pattern_length]
            count = 1
            pos = start + pattern_length
            
            while pos + pattern_length <= text_length and text[pos:pos + pattern_length] == pattern:
                count += 1
                pos += pattern_length
            
            if count >= min_repeats:
                return [(pattern, start, start + (pattern_length * count), count)]
    
    return []

def no_repetition_reward_fn(generated_text, golden_answer, task="math", is_r1_style=False):
    return 0.0 if detect_repetition(generated_text) else 1.0

# Example usage
def test_detector(test_string):
    # print(f"Text: {test_string}")
    repetitions = detect_repetition(test_string)
    if repetitions:
        for pattern, start, end, count in repetitions:
            print(f"  Found: '{pattern}' repeated {count} times from position {start} to {end}")
        return True
    else:
        print("  No repetitions detected")
        return False


if __name__ == '__main__':
    test_string = "Compute: $1+2-3-4+5+6-7-8+ \\dots +97+98-99-100$.  \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}. \\boxed{}."
# results = detect_repetition_rolling_hash(test_string)
# print(results)


    test_string  = "to start the answer, we can use the fact that the sum of the first n natural numbers is n(n+1)/2. so the sum of the first 100 natural numbers is 100*101/2=5050. so the answer is 5050."
    for i in range(20):
        test_string = test_string + " so the answer is 5050."
    contains_repetition = test_detector(test_string)
    print(contains_repetition)
