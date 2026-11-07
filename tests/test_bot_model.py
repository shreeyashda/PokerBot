"""model.Model against what the arena checks: it must be a clean, picklable,
fault-free submission that decides sensibly and learns its opponents.

Scenarios use the real GameState and GameStateView, so they fail if the bot
reads a name the engine does not provide."""

import os
import pickle
import random
import time

import pytest

from arena.dummies import CallBot, FoldBot, RandomBot, ShoveBot, TightBot
from arena.match import MatchConfig, play_match
from arena.players import player
from arena import vet
from engine import play_hand
from evals.score import score
from game_state import GameState, GameStateView
from player_action import PlayerAction as A
from poker_test_helpers import cards, std_deck

from model import Model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAMS = ["bot", "villain", "third", "fourth"]


# -- helpers ----------------------------------------------------------------


def make_state(stacks=(500, 500)):
    return GameState(list(stacks), teams=TEAMS[:len(stacks)])


def to_street(state, board):
    """Everyone calls (checks) round to the next street, then the dealer
    reveals the board cards."""
    while not state.is_street_closed:
        state.process_action(A.call(state.to_act))
    for card in cards(board):
        state.process_action(A.reveal_card(card))


def bot(hand, seat, samples=400, seed=1):
    model = Model()
    model.samples = samples
    model.rng = random.Random(seed)
    model.seed(cards(hand), seat)
    return model


def act(model, state, actions=()):
    return model.run(list(actions), GameStateView(state))


# -- it is a valid submission -----------------------------------------------


def test_model_pickles_and_keeps_what_it_learned():
    model = Model()
    model.stats["villain"] = {"hands": 3, "raises": 2, "calls": 1,
                              "folds": 0, "faced": 1, "fold_faced": 0}
    copy = pickle.loads(pickle.dumps(model, protocol=4))
    assert copy.stats == model.stats


def test_the_pickle_names_only_allowed_globals(tmp_path):
    path = tmp_path / "bot.pkl"
    path.write_bytes(pickle.dumps(Model(), protocol=4))
    names, bad = vet.scan_pickle(str(path), own_modules={"model", "equity"})
    assert bad == [], bad


@pytest.mark.parametrize("name", ["model.py", "equity.py"])
def test_source_has_nothing_the_reviewers_flag(name):
    findings = vet.scan_source(os.path.join(REPO, name), name,
                               own_modules={"model", "equity"})
    assert findings == [], [str(f) for f in findings]


def test_submit_script_accepts_the_bot(tmp_path, monkeypatch, capsys):
    """python -m submit --team ...: pickles, vets and plays the dummies."""
    import submit

    monkeypatch.chdir(REPO)
    assert submit.main(["--team", "pokerbottest", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "OK: played legally against every dummy" in out
    assert "model.py" in out and "equity.py" in out


# -- preflop ----------------------------------------------------------------


def test_aces_raise_on_the_button_and_trash_folds():
    for hand, kind in [("AS AH", "raise"), ("7C 2D", "fold")]:
        state = make_state()
        move = act(bot(hand, state.to_act), state)
        assert move.kind == kind, hand
        assert move.player_id == 0


def test_the_raise_is_a_legal_total():
    state = make_state()
    move = act(bot("AS AH", 0), state)
    low, high = GameStateView(state).raise_range
    assert low <= move.amount <= high


def test_big_blind_option_checks_trash_and_raises_aces():
    for hand, kind in [("7C 2D", "call"), ("AS AH", "raise")]:
        state = make_state()
        state.process_action(A.call(0))  # the button limps
        move = act(bot(hand, state.to_act), state)
        assert move.kind == kind, hand


def test_facing_a_raise_trash_folds_and_aces_reraise():
    for hand, kind in [("7C 2D", "fold"), ("AS AH", "raise")]:
        state = make_state()
        state.process_action(A.raise_to(0, 60))
        move = act(bot(hand, state.to_act), state)
        assert move.kind == kind, hand


def test_facing_a_shove_calls_with_aces_and_folds_trash():
    for hand, kind in [("AS AH", "call"), ("7C 2D", "fold")]:
        state = make_state()
        state.process_action(A.raise_to(0, 500))
        move = act(bot(hand, state.to_act, samples=800), state)
        assert move.kind == kind, hand


def test_short_stack_pushes_or_folds():
    for hand, kind in [("AS AH", "raise"), ("7C 2D", "fold")]:
        state = make_state((240, 240))  # 12 big blinds
        view = GameStateView(state)
        move = act(bot(hand, state.to_act), state)
        assert move.kind == kind, hand
        if kind == "raise":
            assert move.amount == view.max_raise_to


# -- postflop ---------------------------------------------------------------


def flop(board, stacks=(500, 500)):
    state = make_state(stacks)
    to_street(state, board)
    return state


def test_a_set_bets_when_checked_to():
    state = flop("7D 2H 9H")
    move = act(bot("7C 7S", state.to_act), state)
    assert move.kind == "raise"
    assert GameStateView(state).min_raise_to <= move.amount


def test_a_set_never_folds_to_a_bet_but_air_folds_to_a_pot_bet():
    for hand, folds in [("7C 7S", False), ("4S 3D", True)]:
        state = flop("7D 2H 9H" if not folds else "KH 9C 8D")
        state.process_action(A.raise_to(state.to_act, 40))  # pot-size bet
        move = act(bot(hand, state.to_act), state)
        assert (move.kind == "fold") == folds, hand


def test_a_committed_stack_shoves_a_strong_hand():
    state = flop("7D 2H 9H", stacks=(60, 60))  # stack equals the pot
    view = GameStateView(state)
    move = act(bot("7C 7S", state.to_act), state)
    assert move.kind == "raise" and move.amount == view.max_raise_to


def test_air_bluffs_more_against_a_player_who_folds():
    def bluffs(stats):
        count = 0
        for seed in range(150):
            state = flop("KH 9C 8D")
            model = bot("4S 3D", state.to_act, samples=100, seed=seed)
            model.stats = stats
            count += act(model, state).kind == "raise"
        return count

    # We act first after the flop (seat 1): the opponent is seat 0, "bot".
    folder = {"bot": {"hands": 50, "raises": 0, "calls": 10,
                          "folds": 90, "faced": 100, "fold_faced": 95}}
    assert bluffs(folder) > bluffs({}) + 15


def test_decisions_fit_the_time_limit_at_default_settings():
    state = make_state((500, 500, 500, 500))
    to_street(state, "7D 2H 9H")
    model = Model()  # default samples and time cap
    model.seed(cards("AS KS"), state.to_act)
    start = time.perf_counter()
    act(model, state)
    assert time.perf_counter() - start < 3.0  # the arena allows 10 s


# -- learning the opponents ---------------------------------------------------


def test_run_counts_opponent_actions_and_ignores_our_own():
    state = make_state()
    model = bot("AS AH", 0)
    act(model, state, [A.call(1), A.raise_to(1, 60), A.call(0)])
    assert model.stats["villain"]["raises"] == 1
    assert model.stats["villain"]["calls"] == 1
    assert "bot" not in model.stats


def test_folding_to_a_raise_is_tracked_separately_from_a_plain_fold():
    state = make_state()
    model = bot("AS AH", 0)
    act(model, state, [A.fold(1)])  # nothing to face
    act(model, state, [A.raise_to(0, 60), A.fold(1)])
    stats = model.stats["villain"]
    assert stats["folds"] == 2 and stats["faced"] == 1
    assert stats["fold_faced"] == 1


def test_end_game_counts_hands_and_stats_survive_the_next_seed():
    state = make_state()
    model = bot("AS AH", 0)
    model.end_game([A.raise_to(1, 60)], GameStateView(state))
    assert model.stats["villain"]["hands"] == 1
    assert model.stats["villain"]["raises"] == 1
    model.seed(cards("2C 3D"), 1)
    assert model.stats["villain"]["hands"] == 1  # seed kept it


def test_the_profile_moves_with_what_was_seen():
    state = make_state()
    view = GameStateView(state)
    model = bot("AS AH", 0)
    raise_rate, fold_to_bet = model._profile(view)
    assert 0.2 < raise_rate < 0.3 and 0.3 < fold_to_bet < 0.4  # priors
    model.stats["villain"] = {"hands": 99, "raises": 90, "calls": 5,
                              "folds": 5, "faced": 50, "fold_faced": 45}
    raise_rate, fold_to_bet = model._profile(view)
    assert raise_rate > 0.7 and fold_to_bet > 0.7


def test_a_move_is_returned_even_if_learning_breaks():
    class Broken:
        def __getattr__(self, name):
            raise RuntimeError(name)

    state = make_state()
    model = bot("AS AH", 0)
    move = model.run([A.call(1)], GameStateView(state))
    assert move.kind in ("fold", "call", "raise")
    # A view that raises on every read still yields a legal fallback.
    assert model.run([], Broken()).kind == "fold"


# -- the real engine ----------------------------------------------------------


def dealt(seed):
    deck = std_deck()
    random.Random(seed).shuffle(deck)
    return deck


OPPONENTS = {"call": CallBot, "fold": FoldBot, "shove": ShoveBot,
             "random": RandomBot, "tight": TightBot}


@pytest.mark.parametrize("name", list(OPPONENTS))
@pytest.mark.parametrize("stacks", [(500, 500), (60, 60), (30, 500)])
def test_no_faults_in_play_hand_in_any_seat(name, stacks):
    """The whole point: every move is legal, in either seat, at any depth."""
    for seat in (0, 1):
        for d in range(6):
            model = Model()
            model.samples = 200
            models = [OPPONENTS[name](seed=d), OPPONENTS[name](seed=d)]
            models[seat] = model
            result = play_hand(models, dealt(d), list(stacks), button=d % 2)
            assert seat not in result.faults, result.faults
            assert sum(result.net) == 0


def test_no_faults_at_a_full_table():
    for d in range(6):
        model = Model()
        model.samples = 200
        models = [model, CallBot(), ShoveBot(), RandomBot(seed=d)]
        result = play_hand(models, dealt(d), [200, 500, 80, 500],
                           button=d % 4)
        assert 0 not in result.faults, result.faults


@pytest.fixture(scope="module")
def you(tmp_path_factory):
    model = Model()
    model.samples = 400  # keeps the tests quick; the pickle is otherwise real
    path = tmp_path_factory.mktemp("you") / "you.pkl"
    path.write_bytes(pickle.dumps(model, protocol=4))
    return player(str(path))


def test_beats_the_fold_bot_and_is_not_behind_the_calling_station(you):
    for opponent, must_be_ahead in (("fold", True), ("call", False)):
        result = score([play_match([you, player(opponent)],
                                   MatchConfig(deals=40))])
        bb, ci = result.bb_per_100[0], result.ci95[0]
        assert result.faults[0] == 0, result.first_faults[0]
        assert result.max_move_seconds[0] < 5
        if must_be_ahead:
            assert bb - ci > 0
        else:
            assert bb + ci > 0  # at worst unclear, never clearly behind
