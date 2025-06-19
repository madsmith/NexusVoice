import asyncio
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Union

type EventT = str
type CallbackT = Callable[[Any], Union[Any, Awaitable[Any]]]

class SubscriptionToken:
    def __init__(self, event: EventT, nonce: int):
        self._event: EventT = event
        self._nonce: int = nonce

    def __eq__(self, other):
        if not isinstance(other, SubscriptionToken):
            return False
        return self._event == other._event and self._nonce == other._nonce

    def __hash__(self):
        return hash((self._event, self._nonce))

    def __repr__(self):
        return f"<SubscriptionToken {self._event}>"
    
    @property
    def event(self) -> EventT:
        return self._event

    @property
    def nonce(self) -> int:
        return self._nonce


class SubscriberEntry(NamedTuple):
    callback: CallbackT
    token: SubscriptionToken

class EventBus:
    def __init__(self):
        # For each event label, store a list of SubscriberEntry
        self._subscribers: Dict[EventT, List[SubscriberEntry]] = {}
        # Single global monotonically increasing nonce
        self._nonce: int = 0
        self._loop = asyncio.get_event_loop()

    async def emit(self, event: EventT, data: Any = None):
        """
        Emit an event by name. Triggers all subscribers to that event name and any matching regex pattern.
        """
        # Call string subscribers
        for entry in list(self._subscribers.get(event, [])):
            self._loop.create_task(
                self._emit_callback(entry.callback, data),
                name=f"EventBus-callback"
            )

    async def _emit_callback(self, callback: CallbackT, data: Any):
        result = callback(data)
        if asyncio.iscoroutine(result):
            await result

    def once(self, event: EventT, callback: CallbackT):
        """
        Subscribe to an event by name, but only handle the event once. Automatically unsubscribes after first call.
        Returns the subscription token.
        """
        eventbus = self
        token_holder = {}

        async def once_callback(data):
            # Unsubscribe first to ensure it only runs once, even if callback throws
            eventbus.unsubscribe(token_holder['token'])
            await callback(data)

        token = self.subscribe(event, once_callback)
        token_holder['token'] = token
        return token

    def subscribe(self, event: EventT, callback: CallbackT):
        """
        Subscribe to an event by name. Returns a token for unsubscribing.
        """
        token = SubscriptionToken(event, self._nonce)
        self._nonce += 1

        entry = SubscriberEntry(callback=callback, token=token)
        self._subscribers.setdefault(event, []).append(entry)

        return token

    def unsubscribe(self, token: SubscriptionToken):
        """
        Unsubscribe using the token returned by subscribe.
        """
        if not isinstance(token, SubscriptionToken):
            return False  # Invalid token
        
        event = token.event
        if event not in self._subscribers:
            return False
        
        subscribers = self._subscribers[event]
        for i, entry in enumerate(subscribers):
            if entry.token == token:
                del subscribers[i]
                if not subscribers:
                    del self._subscribers[event]
                return True
        
        return False

