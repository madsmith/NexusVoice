import logfire
import pytest

service_name = "pytest"

logfire.configure(
    console=False,
    send_to_logfire=True,
    service_name=service_name,
)

def pytest_collection(session: pytest.Session):
    session.stash["session_span"] = logfire.span("test session")


def pytest_itemcollected(item: pytest.Item):
    if "parent_span" not in item.parent.stash:
        item.parent.stash["parent_span"] = logfire.span(item.parent.name)

    item.stash["item_span"] = logfire.span(item.name)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item):
    with item.session.stash["session_span"]:
        with item.parent.stash["parent_span"]:
            with item.stash["item_span"]:
                yield


def pytest_exception_interact(node: pytest.Item, call: pytest.CallInfo):
    logfire.exception(str(call.excinfo.value), _exc_info=call.excinfo.value)