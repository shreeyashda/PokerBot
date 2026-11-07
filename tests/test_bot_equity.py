"""equity.py: my fast evaluator must agree with card_utils.evaluate, since
the engine settles every pot with card_utils; the Monte Carlo equity must be
close to known values and respect its time cap."""

import random
import time

import pytest

from card_utils import Card, evaluate, full_deck
from equity import (CHEN, card_int, cards_to_ints, chen, equity, rank_cards)


def ints(text):
    return cards_to_ints([Card.parse(t) for t in text.split()])


def test_card_int_is_a_bijection_on_the_deck():
    assert sorted(card_int(c) for c in full_deck()) == list(range(52))


@pytest.mark.parametrize("size", [5, 6, 7])
def test_rank_matches_card_utils_on_random_hands(size):
    """Same tuple as card_utils.evaluate, hand after hand."""
    rng = random.Random(size)
    deck = full_deck()
    for _ in range(4000):
        cards = rng.sample(deck, size)
        assert rank_cards(cards_to_ints(cards)) == evaluate(cards), [
            str(c) for c in cards]


def test_rank_matches_card_utils_on_pair_heavy_hands():
    """Random deals rarely make quads, full houses and straight flushes:
    deal from a few ranks and one suit to force them."""
    rng = random.Random(99)
    deck = [c for c in full_deck() if c.rank in "A2345K" or c.suit == "S"]
    for _ in range(6000):
        cards = rng.sample(deck, rng.choice([5, 6, 7]))
        assert rank_cards(cards_to_ints(cards)) == evaluate(cards), [
            str(c) for c in cards]


@pytest.mark.parametrize("text, category", [
    ("AS KS QS JS TS 2C 3D", 8),   # royal flush
    ("5H 4H 3H 2H AH KC KD", 8),   # steel wheel
    ("9C 9D 9H 9S AC AD 2S", 7),   # quads
    ("7C 7D 7H KS KC 2S 2D", 6),   # full house, a spare pair
    ("7C 7D 7H KS KC KD 2D", 6),   # two trips: the better one leads
    ("AH 9H 7H 4H 2H KC KD", 5),   # flush beats the pair
    ("AC 2D 3H 4S 5C KD 9D", 4),   # the wheel
    ("9C TD JH QS KC 2D 2S", 4),   # a straight beats a pair
    ("QC QD QH 2S 9C 4D 7H", 3),   # trips
    ("AC AD KH KS QC QD 2H", 2),   # three pairs: best two, best kicker
    ("AC AD 2H 5S 9C JD 7H", 1),   # pair
    ("AC KD 2H 5S 9C JD 7H", 0),   # high card
])
def test_known_categories(text, category):
    cards = ints(text)
    assert rank_cards(cards)[0] == category
    assert rank_cards(cards) == evaluate([Card.parse(t) for t in text.split()])


def test_wheel_loses_to_six_high_straight():
    assert rank_cards(ints("AC 2D 3H 4S 5C")) < rank_cards(ints("2C 3D 4H 5S 6C"))


def test_chen_known_scores():
    def score(text):
        a, b = ints(text)
        return CHEN[a][b]

    assert score("AS AH") == 20
    assert score("KS KH") == 16
    assert score("AS KS") == 12
    assert score("AS KD") == 10
    assert score("JS TS") == 9
    assert score("2S 2H") == 5          # pairs score at least 5
    assert score("7C 2D") == -1
    assert chen(14, 13, True) == chen(13, 14, True)  # order does not matter


def test_chen_table_is_symmetric_and_ranks_pairs_and_aces_high():
    for a in range(52):
        for b in range(52):
            if a != b:
                assert CHEN[a][b] == CHEN[b][a]
    assert CHEN[ints("AS")[0]][ints("AH")[0]] == max(max(r) for r in CHEN)


# -- Monte Carlo ------------------------------------------------------------


def eq(hole, board="", opponents=1, **kw):
    kw.setdefault("samples", 4000)
    return equity(ints(hole), ints(board) if board else [], opponents,
                  random.Random(7), **kw)


def test_equity_matches_known_heads_up_values():
    assert eq("AS AH") == pytest.approx(0.85, abs=0.03)
    assert eq("AS KS") == pytest.approx(0.67, abs=0.03)
    assert eq("7C 2D") == pytest.approx(0.35, abs=0.03)


def test_equity_falls_with_more_opponents():
    assert eq("AS AH", opponents=3) < eq("AS AH", opponents=1) - 0.15


def test_equity_of_a_board_royal_is_an_even_split():
    # The board is a royal flush: every opponent ties.
    assert eq("2C 3D", "AS KS QS JS TS") == pytest.approx(0.5)
    assert eq("2C 3D", "AS KS QS JS TS", opponents=3) == pytest.approx(0.25)


def test_equity_of_a_near_unbeatable_and_of_a_hopeless_hand():
    assert eq("AS KS", "QS JS 2S", samples=2000) > 0.9      # nut flush
    # Seven-high with no help on a board of aces and kings.
    assert eq("7C 4D", "AS AD KH KC 9S", samples=2000) < 0.3


def test_a_tight_range_lowers_equity():
    random_range = eq("7C 2D")
    tight_range = eq("7C 2D", cutoff=8)
    assert tight_range < random_range - 0.05


def test_cutoff_range_is_actually_dealt():
    """With a cutoff above every hand but AA, the opponent always holds
    aces: our KK is then a big underdog, never the 80% favourite."""
    value = eq("KS KH", cutoff=20, samples=1500)
    assert value < 0.25


def test_max_seconds_caps_the_work():
    start = time.perf_counter()
    equity(ints("AS KS"), [], 3, random.Random(1), samples=10_000_000,
           max_seconds=0.1)
    assert time.perf_counter() - start < 1.0


def test_equity_does_not_change_its_inputs():
    hole, board = ints("AS KS"), ints("QS JS 2D")
    saved = (list(hole), list(board))
    equity(hole, board, 2, random.Random(1), samples=100)
    assert (hole, board) == saved


def test_equity_is_repeatable_with_a_seed():
    assert eq("QS QH", samples=500) == eq("QS QH", samples=500)
