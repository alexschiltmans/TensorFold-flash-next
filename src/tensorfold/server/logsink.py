"""Server output that can never fail or stall a request.

Every request logs a line with ``print()``, on its HTTP thread, and the engine thread prints too. When the server's
stdout or stderr is a pipe whose reader has gone, or a terminal that was closed (a server started from a shell that
then exited), each write raises: the request log line raises inside ``send_response`` before the status line is
sent, so every request gets an empty reply while the server keeps listening, and an engine-thread print ends the
engine thread. A reader that stops reading fills the pipe instead, and every print then blocks.

``install()`` replaces ``sys.stdout`` and ``sys.stderr`` with ``SafeStream``s. Writes to a regular file (or
/dev/null) go straight through, as before, and an error only costs that text. Writes to a pipe, terminal or socket
are queued and return at once; one writer thread copies the queue to the real streams in write order, a backlog over
``max_chars`` drops text (and says how much once output drains again, or at exit), and a stream that fails for good
is no longer written. ``fileno()`` stays the real descriptor, so ``faulthandler`` still finds it; code that dumps
stacks from a running server uses ``dump_stacks()``, which goes through the queue too.

Queued text is not on the terminal when ``print(..., flush=True)`` returns; a hard crash can lose the last lines.
"""

from __future__ import annotations

import errno
import io
import os
import queue
import stat
import sys
import tempfile
import threading
from typing import TextIO

_MAX_CHARS = 8 * 1024 * 1024        # queued text waiting for a pipe or terminal; more is dropped and counted
_MAX_PENDING = 64 * 1024            # an unfinished line this long is sent without waiting for its newline
_LASTING = {errno.EPIPE, errno.EIO, errno.EBADF, errno.ENXIO}     # errors a stream does not recover from


class _Writer:
    """The one thread that writes queued text, for every queued stream, in the order it was written."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = int(max_chars)
        self._queue: queue.Queue[tuple[SafeStream, str]] = queue.Queue()
        self._lock = threading.Lock()
        self._chars = 0
        self._thread = threading.Thread(target=self._run, name="tensorfold-log", daemon=True)
        self._thread.start()

    def put(self, stream: SafeStream, text: str) -> None:
        with self._lock:
            if self._chars and self._chars + len(text) > self.max_chars:
                stream.dropped += 1
                return
            self._chars += len(text)
        self._queue.put((stream, text))

    def drain(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for the queued text to be written. True when it was."""

        done = threading.Event()

        def wait() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=wait, name="tensorfold-log-drain", daemon=True).start()
        return done.wait(timeout)

    def _run(self) -> None:
        while True:
            stream, text = self._queue.get()
            try:
                stream._emit(text)
            finally:
                with self._lock:
                    self._chars -= len(text)
                self._queue.task_done()


class SafeStream(io.TextIOBase):
    """A text stream whose ``write`` never raises, and never blocks unless ``target`` is a regular file."""

    def __init__(self, target: TextIO, writer: _Writer | None = None) -> None:
        super().__init__()
        self.target = target
        self.writer = writer                # None: write straight through (regular files)
        self.dropped = 0                    # writes dropped because the backlog was full
        self.failed: BaseException | None = None     # the target's lasting write error; nothing is written after it
        self._reported = 0
        self._line_open = False             # the target's last text did not end a line
        self._lock = threading.Lock()
        self._pending: dict[threading.Thread, str] = {}  # each thread's unfinished line (ids get reused)

    # -- the TextIO surface print(), logging, traceback and faulthandler use -----------------------------------
    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self.target, "encoding", None) or "utf-8"

    @property
    def errors(self) -> str | None:  # type: ignore[override]
        return getattr(self.target, "errors", None)

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self.target.fileno()

    def isatty(self) -> bool:
        try:
            return self.target.isatty()
        except (OSError, ValueError):
            return False

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError(f"write() argument must be str, not {type(text).__name__}")
        if text and self.failed is None:
            # whole lines only: print() writes its text and its "\n" separately, and lines from two threads (or a
            # dropped text whose newline was kept) would otherwise run together
            me = threading.current_thread()
            pending = self._pending.pop(me, "") + text
            cut = pending.rfind("\n") + 1
            if len(pending) - cut > _MAX_PENDING:
                cut = len(pending)
            if cut < len(pending):
                self._pending[me] = pending[cut:]
            if cut:
                self._send(pending[:cut])
        return len(text)

    def flush(self, every_thread: bool = False) -> None:
        """Sends this thread's unfinished line (every thread's, at exit); returns without waiting for it."""

        for thread in list(self._pending) if every_thread else [threading.current_thread()]:
            pending = self._pending.pop(thread, "")
            if pending and self.failed is None:
                self._send(pending)

    def _send(self, text: str) -> None:
        if self.writer is None:
            self._emit(text)
        else:
            self.writer.put(self, text)

    def report_dropped(self) -> None:
        """Write the count of dropped writes not reported yet (otherwise written before the next text)."""

        if self.writer is None:
            self._emit("")
        else:
            self.writer._queue.put((self, ""))      # past the size limit: an empty text costs nothing

    # -- writing to the target (the writer thread, or the caller for a regular file) --------------------------
    def _emit(self, text: str) -> None:
        with self._lock:
            if self.failed is not None:
                return
            missed, self._reported = self.dropped - self._reported, self.dropped
            try:
                if missed:
                    end_line = "\n" if self._line_open else ""
                    self.target.write(f"{end_line}[tensorfold] {missed} log writes dropped: "
                                      "output was not being read\n")
                    self._line_open = False
                if text:
                    self.target.write(text)
                    self._line_open = not text.endswith("\n")
                self.target.flush()
            except ValueError as exc:   # a closed file (UnicodeError, text the encoding cannot hold, costs only itself)
                if not isinstance(exc, UnicodeError):
                    self.failed = exc
            except OSError as exc:      # a gone reader or terminal stops the stream; a full disk costs only this text
                if exc.errno in _LASTING:
                    self.failed = exc
                else:
                    self._reported -= missed    # the note did not get out either: report it next time
                    self.dropped += 1
            except Exception:  # noqa: BLE001 - logging must never take down the writer or a request
                pass


def _queues(stream: TextIO) -> bool:
    """False for a regular file or a non-terminal character device (/dev/null): writes there cannot stall."""

    try:
        mode = os.fstat(stream.fileno()).st_mode
        tty = stream.isatty()
    except (OSError, ValueError, AttributeError, io.UnsupportedOperation):
        return True
    return not (stat.S_ISREG(mode) or (stat.S_ISCHR(mode) and not tty))


_installed: tuple[SafeStream, SafeStream] | None = None


def install(drain_s: float = 2.0, max_chars: int = _MAX_CHARS) -> tuple[SafeStream, SafeStream]:
    """Route ``sys.stdout`` and ``sys.stderr`` through ``SafeStream``s (once); at exit, report dropped writes and
    give queued text up to ``drain_s`` seconds to reach the terminal or log."""

    global _installed
    if _installed is not None:
        return _installed
    import atexit

    writer: _Writer | None = None
    streams: list[SafeStream] = []
    for name in ("stdout", "stderr"):
        current = getattr(sys, name)
        queued = _queues(current)
        if queued and writer is None:
            writer = _Writer(max_chars)
        stream = SafeStream(current, writer if queued else None)
        setattr(sys, name, stream)
        streams.append(stream)

    def at_exit() -> None:
        for stream in streams:
            stream.flush(every_thread=True)
            stream.report_dropped()
        if writer is not None:
            writer.drain(drain_s)

    atexit.register(at_exit)
    _installed = (streams[0], streams[1])
    return _installed


def dump_stacks(header: str) -> None:
    """Write ``header`` and every thread's stack to ``sys.stderr`` without writing to its descriptor directly: into a
    full pipe that write would block while holding the interpreter lock, stopping every thread."""

    import faulthandler

    try:
        with tempfile.TemporaryFile("w+") as scratch:
            faulthandler.dump_traceback(file=scratch, all_threads=True)
            scratch.seek(0)
            stacks = scratch.read()
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not end the thread that asked for it
        stacks = f"(no stacks: {type(exc).__name__}: {exc})\n"
    sys.stderr.write(f"{header}\n{stacks}")


def register_stack_dump(signum: int) -> None:
    """``kill -<signum> <pid>`` prints every thread's stack. Into a regular file faulthandler writes it from C, which
    works even while Python is stuck; into a pipe or terminal that C write could block the whole process, so the
    stacks are taken on the main thread and queued like any other output."""

    import faulthandler
    import signal

    stream = sys.stderr
    if isinstance(stream, SafeStream) and stream.writer is not None:
        signal.signal(signum, lambda *_: dump_stacks(f"[tensorfold] signal {signum}: every thread's stack follows"))
    else:
        faulthandler.register(signum, all_threads=True)


__all__ = ["SafeStream", "dump_stacks", "install", "register_stack_dump"]
