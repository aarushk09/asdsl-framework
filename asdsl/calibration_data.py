"""Shared calibration prompts for ASDSL training and calibration scripts.

Used by:
  - scripts/slim_calibrate.py     (SliM bit-width calibration)
  - scripts/train_mtp_head.py     (EAGLE-3 MTP head training)
"""

CALIBRATION_PROMPTS = [
    "The capital of France is",
    "In mathematics, the derivative of x squared is",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "The French Revolution began in the year",
    "To convert Celsius to Fahrenheit, multiply by",
    "The mitochondria is known as",
    "import numpy as np\narr = np.array([1, 2, 3])\nresult =",
    "Once upon a time in a land far away,",
    "The speed of light in a vacuum is approximately",
    "SELECT * FROM users WHERE",
    "The three laws of thermodynamics state that",
    "To train a neural network, we use",
    "The Battle of Waterloo took place in",
    "In Python, a list comprehension looks like",
    "The chemical formula for water is",
    "def quicksort(arr):\n    if len(arr) <= 1:",
    "The largest planet in our solar system is",
    "HTTP status code 404 means",
    "The square root of 144 is",
    "To reverse a string in Python,",
    "The Pythagorean theorem states that",
    "Machine learning models overfit when",
    "The main difference between RAM and ROM is",
    "In quantum mechanics, Heisenberg's uncertainty principle",
    "The Treaty of Versailles was signed in",
    "Binary search has a time complexity of",
    "The boiling point of water at sea level is",
    "Object-oriented programming is based on the concept of",
    "The first element in the periodic table is",
    "To calculate compound interest, the formula is",
    "The Renaissance period began in",
    "In SQL, the JOIN operation combines",
]

# Quick-mode subset (4 prompts for CI/pipeline checks)
QUICK_PROMPTS = CALIBRATION_PROMPTS[:4]
