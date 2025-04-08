import re
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords


# measurement units, common ingredients to remove
UNITS = [
    "cup",
    "cups",
    "tablespoon",
    "tablespoons",
    "teaspoon",
    "teaspoons",
    "ounce",
    "ounces",
    "oz",
    "pound",
    "pounds",
    "gram",
    "grams",
    "can",
    "cans",
    "stick",
    "sticks",
    "clove",
    "cloves",
    "stalk",
    "stalks",
    "slice",
    "slices",
    "dash",
    "pinch",
    "quart",
    "quarts",
    "pint",
    "pints",
    "liter",
    "liters",
    "gallon",
    "gallons",
    "ml",
    "milliliter",
    "milliliters",
    "tsp",
    "tbsp",
    "medium",
    "small",
    "large",
]


# Regex patterns to clean the string
NUMBER_PATTERN = re.compile(r"\b\d+(\.\d+)?\b")  # Matches whole numbers and decimals
PUNCTUATION_PATTERN = re.compile(
    r"[^\w\s]"
)  # Matches punctuation (anything that's not a word or whitespace)


def clean_ingredient(ingredient):
    ingredient = NUMBER_PATTERN.sub("", ingredient).strip()

    ingredient = PUNCTUATION_PATTERN.sub("", ingredient).strip()

    words = word_tokenize(ingredient)

    tagged = pos_tag(words)

    filtered_words = [
        word
        for word, tag in tagged
        if tag in ("NN", "NNS", "NNP", "NNPS")
        and word.lower() not in UNITS  # Remove units
        and word.lower() not in stopwords.words("english")  # Remove stopwords
    ]

    return " ".join(filtered_words)


def clean_ingredients(ingredient_lines):
    ingredients = [clean_ingredient(ing) for ing in ingredient_lines]
    ingredients = [ing for ing in ingredients if ing]

    return ingredients
