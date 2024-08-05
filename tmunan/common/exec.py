import enum
import atexit
import queue
import threading
import multiprocessing

from abc import ABC
from queue import Empty
from typing import Callable, Type

from tmunan.common.event import Event
from tmunan.common.log import get_logger


class _BaseMonitoredProcess(multiprocessing.process.BaseProcess):
    """
        Implemented by running a Thread in parallel to the Process,
        the extra Thread only waits for the process to exit by using `join()` - so no overhead is introduced.
    """

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, daemon=None):
        """
        Creates a standard multiprocessing.Process, see official documentation for reference.
        """

        if kwargs is None:
            kwargs = dict()

        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

        # private
        self._process_watcher_thread = None
        self._process_on_exit_callback = None
        self._memory_profiler_thread = None
        self._memory_profiler_interval = 5

    def _process_watcher_func(self, on_exit: Callable):
        self.join()
        if on_exit:
            on_exit()

    def start(self, on_exit: Callable = None):

        # start process
        super().start()

        # process watcher
        self._process_on_exit_callback = on_exit
        self._process_watcher_thread = threading.Thread(target=self._process_watcher_func,
                                                        args=(self._process_on_exit_callback,),
                                                        daemon=True)
        self._process_watcher_thread.start()

    def run(self):
        super().run()


class ForkMonitoredProcess(_BaseMonitoredProcess, multiprocessing.context.ForkProcess):
    pass


class SpawnMonitoredProcess(_BaseMonitoredProcess, multiprocessing.context.SpawnProcess):
    pass


class MonitoredThread(threading.Thread):
    """
        Subclass of threading.Thread which supports invocation of a callback function when the thread exits.
        Implemented by direct invocation of the callback function right after the thread's main function returns.
    """

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, daemon=None):
        """
        Creates a standard threading.Thread, see official documentation for reference.
        """

        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

        # private
        self._process_on_exit_callback = None

    def start(self, on_exit: Callable = None):

        # save exit callback func
        self._process_on_exit_callback = on_exit

        # start thread
        super().start()

    def run(self):
        super().run()

        # fire callback
        if self._process_on_exit_callback:
            self._process_on_exit_callback()


class BackgroundTask(ABC):

    def __init__(self, *args, **kwargs):
        pass

    def setup(self):
        pass

    def exec(self, item):
        raise NotImplemented()

    def cleanup(self):
        pass


class BackgroundExecutor:

    class ProcessCreationMethod(enum.Enum):
        Fork = 0
        Spawn = 1

    def __init__(self, task_class: Type[BackgroundTask], proc_method: ProcessCreationMethod, *args, **kwargs):

        # process method
        self.worker_process_type = ForkMonitoredProcess if proc_method == self.ProcessCreationMethod.Fork else SpawnMonitoredProcess

        # task
        self._task_class = task_class
        self._task_args = args or []
        self._task_kwargs = kwargs or {}

        # multiprocessing
        self._input_queue = multiprocessing.Queue()
        self._output_queue = multiprocessing.Queue()
        self._stop_event = multiprocessing.Event()
        self._proc = self.worker_process_type(
            target=self.run,
            args=(self._input_queue, self._output_queue, self._stop_event,
                  self._task_class, self._task_args, self._task_kwargs))
        self._output_thread = None

        # events
        self.on_exit = Event()
        self.on_error = Event()
        self.on_worker_ready = Event()
        self.on_output_ready = Event()

    @property
    def process(self):
        return self._proc

    @property
    def input_queue(self):
        return self._input_queue

    def push_input(self, item, max_size=None):

        # put new item
        self._input_queue.put(item)

    def stop(self, force=False):

        # if force - stop immediately
        if force:
            self._stop_event.set()

        # otherwise - put death pill on queue
        else:
            self._input_queue.put(None)

        # wait for process to finish
        self._proc.join()

    def _output_thread_func(self):
        while not (self._output_queue.empty() and self._stop_event.is_set()):
            try:
                success, data = self._output_queue.get(timeout=0.1)
                if success is None and data is None:
                    # fire ready event
                    self.on_worker_ready.fire()
                elif success:
                    # fire output event
                    self.on_output_ready.fire(data)
                else:
                    # fire error event
                    self.on_error.fire(data)
            except:
                pass

    def start(self):

        def terminate_child():
            if self._proc.is_alive():
                self._proc.terminate()

        # start output thread
        self._output_thread = threading.Thread(target=self._output_thread_func, daemon=True)
        self._output_thread.start()

        # start background process
        self._proc.start(on_exit=self.on_exit)
        atexit.register(terminate_child)

    @staticmethod
    def run(in_q, out_q, stop_event, task_class, task_args, task_kwargs):

        # init logger
        logger = get_logger(f'{task_class.__name__}Executor')
        try:

            # initialize task
            logger.info(f'Initializing Task of type: {task_class.__name__}')
            task = task_class(*task_args, **task_kwargs)
            task.setup()

            # signal ready
            logger.info(f'Task is ready.')
            out_q.put_nowait((None, None))

            # run task loop
            logger.info(f'about to start while loop {stop_event.is_set()}')
            while not stop_event.is_set():
                try:
                    item = in_q.get(timeout=1)
                    logger.info(f'got item from input queue')
                    if item is None:
                        break

                    logger.info(f'running exec')
                    result = task.exec(item)
                    out_q.put((True, result))
                except Empty:
                    logger.info(f'empty queue')
                    continue
                except Exception as e:
                    logger.exception('Error processing item!')
                    out_q.put((False, e))

            # release resources
            logger.info(f'Cleaning up resources...')
            task.cleanup()

        except Exception as e:
            logger.exception('Error in run loop')
