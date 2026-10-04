"""Built-in demo corpus: short factual passages with clean answers.

Each passage is written so its best sentence *contains* the answer --
that is what lets the offline extractive generator produce segments the
critic can honestly mark ``fully_supported``.
"""
from __future__ import annotations

from .retriever import Passage

BUILTIN_CORPUS: list[Passage] = [
    Passage(
        id="eiffel",
        title="Eiffel Tower",
        text=(
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
            "The Eiffel Tower stands 330 metres tall and was completed in 1889. "
            "It was designed by Gustave Eiffel for the 1889 World's Fair."
        ),
    ),
    Passage(
        id="curie",
        title="Marie Curie",
        text=(
            "Marie Curie was a Polish-French physicist and chemist who pioneered research on radioactivity. "
            "Marie Curie won Nobel Prizes in both Physics in 1903 and Chemistry in 1911. "
            "She was the first person to win Nobel Prizes in two different sciences."
        ),
    ),
    Passage(
        id="photosynthesis",
        title="Photosynthesis",
        text=(
            "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into oxygen and glucose. "
            "Photosynthesis takes place mainly in the leaves, inside organelles called chloroplasts. "
            "The oxygen released by photosynthesis comes from water molecules."
        ),
    ),
    Passage(
        id="moon",
        title="Moon landing",
        text=(
            "Apollo 11 was the first mission to land humans on the Moon. "
            "Neil Armstrong and Buzz Aldrin landed on the Moon on 20 July 1969. "
            "Michael Collins remained in lunar orbit aboard the command module."
        ),
    ),
    Passage(
        id="python-lang",
        title="Python programming language",
        text=(
            "Python is a high-level programming language created by Guido van Rossum. "
            "Python was first released in 1991 and emphasises code readability. "
            "Python supports multiple programming paradigms, including object-oriented and functional programming."
        ),
    ),
    Passage(
        id="water",
        title="Boiling point of water",
        text=(
            "At standard atmospheric pressure at sea level, water boils at 100 degrees Celsius. "
            "The boiling point of water decreases as altitude increases and pressure falls. "
            "Water freezes at 0 degrees Celsius at standard pressure."
        ),
    ),
    Passage(
        id="everest",
        title="Mount Everest",
        text=(
            "Mount Everest is the highest mountain above sea level on Earth. "
            "Mount Everest rises 8849 metres above sea level in the Himalayas. "
            "Edmund Hillary and Tenzing Norgay first summited Everest in 1953."
        ),
    ),
    Passage(
        id="shakespeare",
        title="William Shakespeare",
        text=(
            "William Shakespeare was an English playwright and poet, widely regarded as the greatest writer in the English language. "
            "Shakespeare wrote the tragedy Hamlet, in which Prince Hamlet of Denmark seeks revenge. "
            "He was born in Stratford-upon-Avon in 1564."
        ),
    ),
]
