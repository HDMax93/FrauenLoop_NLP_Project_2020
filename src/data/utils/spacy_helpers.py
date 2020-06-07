"""
Utility functions to perform different text preprocessing tasks with spacy
"""

from typing import List

import spacy

nlp = spacy.load('en_core_web_sm')


def text2sentences(text: str) -> List[str]:
    """
    Split a text into sentences using the spacy model.

    Args:
        text: text to split into sentences
    """
    doc = nlp(text)

    sentences_text = list()
    for sentence in doc.sents:
        sentences_text.append(sentence.text)
    # or, with list comprehension:
    # sentences_text = [sentence.text for token in doc.sents]

    return sentences_text


def text2sentences_custom(text: str) -> List[str]:
    """
    Split a text into sentences.

    Args:
        text: text to split into sentences
    """
    sentences = text.split('//')
    sentences_ = [sentence.strip() for sentence in sentences]
    sentences2 = [sentence.replace('\n', '') for sentence in sentences_]
    sentences3 = [' '.join(sentence.split())
                  for sentence in sentences2]

    return sentences_


def text2tokens(text: str) -> List[str]:
    """
    Tokenize a text into a list of strings using the spacy model.

    Args:
        text (str): text to tokenize
    """
    doc = nlp(text)

    tokens_text = list()
    for token in doc:
        tokens_text.append(token.text)
    # or, with list comprehension:
    # tokens_text = [token.text for token in doc]

    return tokens_text


def text2tokens_(text: str) -> List[dict]:
    """
    Tokenize a text and return tokens and relevant attributes

    Args:
        text (str): text to tokenize
    """
    doc = nlp(text)

    tokens = [{'text': t.text,
               'lower_': t.lower_,
               'lemma_': t.lemma_,
               'is_alpha': t.is_alpha,
               'is_digit': t.is_digit,
               'is_punct': t.is_punct,
               'pos_': t.pos_}
              for t in doc]

    return tokens