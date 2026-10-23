"""Server output that cannot fail or stall requests: a closed pipe or terminal, or a reader that stopped reading,
must cost log lines, never replies (server.logsink)."""

from __future__ import annotations

import errno
import http.client
import io
import os
import pty
import signal
import re
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from tensorfold.server.logsink import SafeStream, _queues, _Writer

SRC = str(Path(__file__).resolve().parents[1] / "src")
LINE = re.compile(r"line \d+|t\d+ \d+|after|LAST|\[tensorfold\] \d+ log writes dropped: output was not being read")


class Broken(io.StringIO):
    def write(self, text: str) -> int:
        raise BrokenPipeError(32, "Broken pipe")


class Stuck(io.StringIO):
    """A target whose writes block until ``release`` is set (a pipe nobody reads)."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def write(self, text: str) -> int:
        self.release.wait()
        return super().write(text)


def queued(target, max_chars: int = 8 * 1024 * 1024) -> SafeStream:
    return SafeStream(target, _Writer(max_chars))


def whole_lines(text: str) -> bool:
    return all(LINE.fullmatch(line) for line in text.split("\n")[:-1]) and text.endswith("\n")


def test_text_reaches_a_pipe_in_order() -> None:
    target = io.StringIO()
    stream = queued(target)
    for i in range(20_000):
        print(f"line {i}", file=stream)
    assert stream.writer.drain(10.0)
    assert stream.dropped == 0
    assert target.getvalue() == "".join(f"line {i}\n" for i in range(20_000))


def test_a_burst_into_a_regular_file_loses_nothing(tmp_path: Path) -> None:
    with open(tmp_path / "log", "w", encoding="utf-8") as target:
        stream = SafeStream(target)         # what install() picks for a regular file
        threads = [threading.Thread(target=lambda k=k: [print(f"t{k} {i}", file=stream) for i in range(5000)])
                   for k in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print("LAST", file=stream)
    text = (tmp_path / "log").read_text()
    assert stream.dropped == 0 and whole_lines(text)
    assert sorted(text.splitlines()) == sorted([f"t{k} {i}" for k in range(4) for i in range(5000)] + ["LAST"])


def test_stdout_and_stderr_keep_their_order_through_one_writer() -> None:
    target = io.StringIO()
    writer = _Writer(8 * 1024 * 1024)
    out, err = SafeStream(target, writer), SafeStream(target, writer)
    for i in range(2000):
        print(f"line {2 * i}", file=out)
        print(f"line {2 * i + 1}", file=err)
    assert writer.drain(10.0)
    assert target.getvalue() == "".join(f"line {i}\n" for i in range(4000))


def test_a_broken_target_never_raises() -> None:
    stream = queued(Broken())
    for i in range(200):
        print(f"line {i}", file=stream, flush=True)
    assert stream.writer.drain(5.0)
    assert isinstance(stream.failed, BrokenPipeError)
    print("after the failure", file=stream, flush=True)
    direct = SafeStream(Broken())
    print("straight through", file=direct, flush=True)
    assert isinstance(direct.failed, BrokenPipeError)


def test_a_stuck_target_never_blocks_and_reports_the_drop() -> None:
    target = Stuck()
    stream = queued(target, max_chars=4096)
    started = time.perf_counter()
    for i in range(10_000):
        print(f"line {i}", file=stream, flush=True)
    assert time.perf_counter() - started < 2.0
    assert stream.dropped > 0
    target.release.set()
    assert stream.writer.drain(5.0)
    print("after", file=stream)
    assert stream.writer.drain(5.0)
    text = target.getvalue()
    assert "log writes dropped: output was not being read" in text
    assert text.endswith("after\n") and whole_lines(text)


def test_every_drop_is_reported() -> None:
    target = Stuck()
    stream = queued(target, max_chars=64)
    for i in range(100):
        print(f"line {i}", file=stream)
    assert stream.dropped > 0
    target.release.set()
    stream.report_dropped()                 # what install()'s exit hook does
    assert stream.writer.drain(5.0)
    text = target.getvalue()
    reported = sum(int(n) for n in re.findall(r"\[tensorfold\] (\d+) log writes dropped", text))
    assert reported == stream.dropped and whole_lines(text)
    assert len(re.findall(r"^line \d+$", text, re.M)) + stream.dropped == 100


def test_an_unfinished_line_is_sent_by_flush() -> None:
    target = io.StringIO()
    stream = queued(target)
    print("progress 10%", end="", file=stream, flush=True)
    assert stream.writer.drain(5.0)
    assert target.getvalue() == "progress 10%"


def test_unencodable_text_costs_only_itself() -> None:
    raw = io.BytesIO()
    target = io.TextIOWrapper(raw, encoding="ascii")
    stream = queued(target)
    print("caf\u00e9", file=stream)
    print("plain", file=stream)
    assert stream.writer.drain(5.0)
    assert stream.failed is None
    assert b"plain\n" in raw.getvalue()


def test_exit_sends_every_thread_s_unfinished_line(tmp_path: Path) -> None:
    script = tmp_path / "exit.py"
    script.write_text(textwrap.dedent("""
        import sys, threading
        from tensorfold.server import logsink
        logsink.install()
        print("first")
        t = threading.Thread(target=lambda: print("from a thread, no newline", end=""))
        t.start(); t.join()
        print("main, no newline", end="")
    """))
    env = {**os.environ, "PYTHONPATH": SRC}
    for output in ("pipe", "file"):
        if output == "pipe":
            out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True).stdout
        else:
            with open(tmp_path / "out", "w") as f:
                subprocess.run([sys.executable, str(script)], env=env, stdout=f)
            out = (tmp_path / "out").read_text()
        assert out.startswith("first\n")
        assert "from a thread, no newline" in out and "main, no newline" in out


class FullDisk(io.StringIO):
    """A regular file on a disk that is full once, then has room again."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_next = False

    def write(self, text: str) -> int:
        if self.fail_next:
            self.fail_next = False
            raise OSError(errno.ENOSPC, "No space left on device")
        return super().write(text)


def test_a_full_disk_costs_only_the_text_it_refused() -> None:
    target = FullDisk()
    stream = SafeStream(target)
    print("before", file=stream)
    target.fail_next = True
    print("lost", file=stream)
    print("after", file=stream)
    assert stream.failed is None and stream.dropped == 1
    assert target.getvalue() == "before\n[tensorfold] 1 log writes dropped: output was not being read\nafter\n"


def test_only_pipes_and_terminals_are_queued(tmp_path: Path) -> None:
    with open(tmp_path / "f", "w") as regular, open(os.devnull, "w") as null:
        assert not _queues(regular) and not _queues(null)
    read, write = os.pipe()
    with os.fdopen(write, "w") as pipe:
        assert _queues(pipe)
    os.close(read)
    master, slave = pty.openpty()
    with os.fdopen(slave, "w") as tty:
        assert _queues(tty)
    os.close(master)


def test_the_exit_hook_reports_drops(tmp_path: Path) -> None:
    script = tmp_path / "burst.py"
    script.write_text(textwrap.dedent("""
        from tensorfold.server import logsink
        logsink.install(max_chars=4096)
        for i in range(20_000):
            print(f"line {i}")
    """))
    proc = subprocess.Popen([sys.executable, str(script)], env={**os.environ, "PYTHONPATH": SRC},
                            stdout=subprocess.PIPE, text=True)
    time.sleep(1.0)                         # nobody reads while it writes
    out = proc.communicate(timeout=30)[0]
    assert proc.returncode == 0
    reported = sum(int(n) for n in re.findall(r"\[tensorfold\] (\d+) log writes dropped", out))
    assert reported > 0 and reported + len(re.findall(r"^line \d+$", out, re.M)) == 20_000 and whole_lines(out)


def test_fileno_and_encoding_pass_through() -> None:
    with open(os.devnull, "w", encoding="utf-8") as target:
        stream = SafeStream(target)
        assert stream.fileno() == target.fileno()
        assert stream.encoding == "utf-8"


# -- the whole server: its output closed or unread, as when the shell that started it has exited ----------------

SERVER = textwrap.dedent("""
    import sys
    from http.server import ThreadingHTTPServer
    from tensorfold.server import logsink
    from tensorfold.server.http import make_handler
    from tests.test_server_openai_compat import FakeApp
    if sys.argv[1].startswith("sink"):
        logsink.install()
    if sys.argv[1].startswith("sink"):
        import signal
        logsink.register_stack_dump(signal.SIGUSR1)
    if sys.argv[1].endswith("dump"):     # the stall watchdog firing over and over
        import faulthandler, threading, time

        def dump():
            while True:
                if sys.argv[1] == "sink-dump":
                    logsink.dump_stacks("[tensorfold] stalled: every thread's stack follows")
                else:
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                time.sleep(0.01)

        threading.Thread(target=dump, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeApp()))
    with open(sys.argv[2], "w") as port:
        port.write(str(server.server_address[1]))
    print("ready", flush=True)
    server.serve_forever()
""")


def _start(tmp_path: Path, mode: str, output: str) -> tuple[subprocess.Popen | int, int]:
    """Start the fake-app server with ``output`` = "dead-pipe", "unread-pipe" or "closed-tty"; return its process
    (or pid) and port once the output is in that state."""

    script = tmp_path / "server.py"
    script.write_text(SERVER)
    port_file = tmp_path / "port"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([SRC, str(Path(SRC).parent)]), "PYTHONUNBUFFERED": "1"}
    args = [sys.executable, str(script), mode, str(port_file)]
    if output == "closed-tty":
        pid, fd = pty.fork()
        if pid == 0:            # the child ignores the hang-up, as a server started in the background does
            import signal

            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            os.execve(args[0], args, env)
        os.read(fd, 64)         # "ready"
        os.close(fd)            # the terminal goes away
        proc: subprocess.Popen | int = pid
    else:
        proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        proc.stdout.readline()  # "ready"
        if output == "dead-pipe":
            proc.stdout.close()
    deadline = time.time() + 10
    while not port_file.exists() or not port_file.read_text():
        assert time.time() < deadline
        time.sleep(0.05)
    return proc, int(port_file.read_text())


def _stop(proc: subprocess.Popen | int) -> None:
    if isinstance(proc, int):
        os.kill(proc, 9)
        os.waitpid(proc, 0)
    else:
        proc.kill()
        proc.wait()


def _replies(port: int, n: int, stop_at_failure: bool = False) -> int:
    ok = 0
    for _ in range(n):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", "/v1/models")
            ok += conn.getresponse().status == 200
        except (OSError, http.client.HTTPException):
            if stop_at_failure:
                break
        finally:
            conn.close()
    return ok


@pytest.mark.parametrize("output", ["dead-pipe", "closed-tty"])
def test_without_the_sink_a_lost_output_fails_every_request(tmp_path: Path, output: str) -> None:
    """The failure as reported: the server listens, every request gets an empty reply."""

    proc, port = _start(tmp_path, "plain", output)
    try:
        assert _replies(port, 20) == 0
    finally:
        _stop(proc)


@pytest.mark.parametrize("mode", ["plain", "plain-dump"])
def test_without_the_sink_an_unread_output_stalls_requests(tmp_path: Path, mode: str) -> None:
    proc, port = _start(tmp_path, mode, "unread-pipe")
    try:
        assert _replies(port, 2000, stop_at_failure=True) < 2000
    finally:
        _stop(proc)


@pytest.mark.parametrize("mode", ["sink", "sink-dump"])
@pytest.mark.parametrize("output", ["dead-pipe", "closed-tty", "unread-pipe"])
def test_with_the_sink_a_lost_output_fails_no_request(tmp_path: Path, output: str, mode: str) -> None:
    proc, port = _start(tmp_path, mode, output)
    try:
        assert _replies(port, 2000 if output == "unread-pipe" else 200) == (2000 if output == "unread-pipe" else 200)
    finally:
        _stop(proc)


def test_sigusr1_stack_dumps_into_an_unread_pipe_block_nothing(tmp_path: Path) -> None:
    proc, port = _start(tmp_path, "sink", "unread-pipe")
    try:
        assert _replies(port, 1500) == 1500    # fills the pipe
        for _ in range(20):
            os.kill(proc.pid, signal.SIGUSR1)
            time.sleep(0.05)
        assert _replies(port, 200) == 200
    finally:
        _stop(proc)
