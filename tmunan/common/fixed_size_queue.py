import queue
import asyncio


class AsyncFixedSizeQueue(asyncio.Queue):
    """An asynchronous queue with a fixed size of 1 that overwrites on put."""

    async def put(self, item):
        """Overrides the default put method. If full, empty the queue asynchronously."""

        # empty the queue
        await self.pop_all()

        # Add the new item
        await super().put(item)

    async def pop_all(self):

        # iterate while queue isn't empty
        while not self.empty():
            try:
                # Discard the existing item if possible
                await self.get()
            except asyncio.QueueEmpty:
                break


class FixedSizeQueue(queue.Queue):
    """An asynchronous queue with a fixed size of 1 that overwrites on put."""

    def __init__(self, max_size: int):
        super().__init__()

        self._max_size = max_size

    def put(self, item, block=True, timeout=None):
        """Overrides the default put method. If full, empty the queue asynchronously."""

        # empty the queue
        self.drop_to_size()

        # Add the new item
        super().put(item, block, timeout)

    def drop_to_size(self):

        # iterate while queue isn't empty
        while self.qsize() > self._max_size:
            try:
                # Discard the existing item if possible
                self.get_nowait()
            except asyncio.QueueEmpty:
                break
