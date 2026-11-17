"""Прогон всего интерфейса без браузера (streamlit.testing): все вкладки отрисовываются без ошибок."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_app_runs_without_errors(app):
    assert not app.exception
    assert "Рекомендация:" in " ".join(s.value for s in app.success)


def test_tight_budget_shows_message(app):
    app.slider(key="budget").set_value((6490, 7000)).run()            # бюджет, где нет моделей с NFC
    assert not app.exception
    assert any("не подходит ни одна модель" in e.value for e in app.error)


def test_pairwise_mode(app):
    app.slider(key="budget").set_value((20000, 50000)).run()
    app.radio(key="weight_mode").set_value("Попарные сравнения по шкале Саати").run()
    assert not app.exception
    assert any(m.label == "CR" for m in app.metric)
