from core.disambiguation import refine_sign
from core.session import Mode, SessionController


def test_content_switch_resets_target():
    controller = SessionController()
    controller.set_content_set("numbers")
    assert controller.state.target == "0"
    controller.set_target(4)
    assert controller.state.target == "4"


def test_quiz_timer_completion_returns_to_learn():
    controller = SessionController()
    controller.start_timed_mode(Mode.QUIZ, 2)
    assert controller.state.mode == Mode.QUIZ
    assert controller.tick_timer() is False
    assert controller.tick_timer() is True
    assert controller.state.mode == Mode.LEARN


def test_disambiguation_uvr():
    # Wide gap should classify as V
    sign = refine_sign(
        "U",
        {
            "openCount": 2,
            "indexUp": True,
            "middleUp": True,
            "ringUp": False,
            "pinkyUp": False,
            "indexMiddleGap": 0.4,
            "indexMiddleCrossed": False,
        },
    )
    assert sign == "V"


def test_disambiguation_6789():
    sign = refine_sign(
        "8",
        {
            "openCount": 3,
            "thumbOut": False,
            "thumbToIndexTip": 0.2,
            "thumbToMiddleTip": 0.6,
            "thumbToRingTip": 0.7,
            "thumbToPinkyTip": 0.9,
        },
    )
    assert sign == "9"
