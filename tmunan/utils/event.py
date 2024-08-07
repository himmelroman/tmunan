class Event:

    def __init__(self):
        self.handlers = list()

    def clear(self):
        self.handlers = list()

    def handle(self, handler):
        if handler not in self.handlers:
            self.handlers.append(handler)
        return self

    def unhandle(self, handler):
        try:
            self.handlers.remove(handler)
        except:
            raise ValueError("Handler is not handling this event, so cannot unhandle it.")
        return self

    def fire(self, *args, **kargs):
        results = []
        for handler in self.handlers:
            result = handler(*args, **kargs)
            results.append(result)

        return results

    def notify(self, *args, **kwargs):
        results = []
        for handler in self.handlers:

            try:
                result = handler(*args, **kwargs)
                results.append(result)

            except Exception as ex:
                print(str(ex))  # TODO : replace with a more standard approach for event handler failure (see TRAX-692)
                results.append(ex)

        return results

    def getHandlerCount(self):
        return len(self.handlers)

    __iadd__ = handle
    __isub__ = unhandle
    __call__ = fire
    __len__ = getHandlerCount
