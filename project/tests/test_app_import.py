"""The Chainlit app imports and handles messages without an index, an API key or a running server."""

import importlib
import os
import sys
from pathlib import Path

import pytest

from sample_docs import EXAMPLE_OUTPUT

APP_DIR = Path(__file__).resolve().parents[1] / "app"

# Chainlit reads/creates `.chainlit/` in its app root at import time: point it at project/app (not the repo root).
if "chainlit" not in sys.modules:
    os.environ.setdefault("CHAINLIT_APP_ROOT", str(APP_DIR))


@pytest.fixture(scope="module")
def app():
    return importlib.import_module("app")


def test_import_registers_callbacks(app):
    from chainlit.config import config

    assert config.code.set_starters is not None
    assert config.code.on_message is not None


async def test_starters(app):
    starters = await app.set_starters(None, None)

    assert [s.message for s in starters] == app.EXAMPLE_QUESTIONS
    assert len(starters) == 5
    assert all(s.label == s.message for s in starters)


class FakeChain:
    async def astream(self, question):
        yield {"question": question}
        yield {"context": EXAMPLE_OUTPUT["context"]}
        for token in ("- Green ", "tea"):
            yield {"answer": token}


@pytest.fixture
def ui(app, monkeypatch):
    """Replace the Chainlit UI primitives used by `main` with recorders."""
    events: list[tuple] = []

    class FakeMessage:
        def __init__(self, content=""):
            self.content = content

        async def stream_token(self, token):
            self.content += token
            events.append(("token", token))

        async def send(self):
            events.append(("message", self.content))

    class FakeErrorMessage(FakeMessage):
        async def send(self):
            events.append(("error", self.content))

    class FakeStep:
        def __init__(self, name, type, **kwargs):
            self.name, self.type, self.output = name, type, ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            events.append(("step", self.name, self.type, self.output))

    monkeypatch.setattr(app.cl, "Message", FakeMessage)
    monkeypatch.setattr(app.cl, "ErrorMessage", FakeErrorMessage)
    monkeypatch.setattr(app.cl, "Step", FakeStep)
    return events


async def test_main_streams_sources_then_answer(app, ui, monkeypatch):
    monkeypatch.setattr(app, "get_chain", FakeChain)

    await app.main(app.cl.Message(content="green tea?"))

    assert ui[0][:3] == ("step", "Retriever", "retrieval")
    assert ui[0][3].startswith("1. [Green tea extract")
    assert ui[1:] == [("token", "- Green "), ("token", "tea"), ("message", "- Green tea")]


async def test_main_reports_missing_index(app, ui, monkeypatch, tmp_path):
    import functions

    functions.get_chain.cache_clear()
    monkeypatch.setattr(functions, "DB_PATH", tmp_path)

    await app.main(app.cl.Message(content="green tea?"))

    assert [kind for kind, _ in ui] == ["error"]
