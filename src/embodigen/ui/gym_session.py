"""One serialized CLI Gym worker, started lazily and stopped with the console."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path


class GymSession:
    def __init__(self, root: Path, env: dict[str, str]):
        self.root, self.env = root, env
        self.proc = None
        self._lock = threading.Lock()
        self._responses = queue.Queue()

    def _start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        self._responses = queue.Queue()
        self.proc = subprocess.Popen([sys.executable, "-m", "embodigen.cli", "gym", "--worker"],
                                     cwd=self.root, env=self.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        proc, responses = self.proc, self._responses

        def read():
            with proc.stdout:
                for line in proc.stdout:
                    try:
                        responses.put(json.loads(line))
                    except ValueError:
                        responses.put({"error": "Gym worker returned invalid JSON"})
            responses.put({"error": "Gym worker stopped; reset the scene to restart"})

        threading.Thread(target=read, daemon=True).start()

    def request(self, command: dict):
        with self._lock:
            self._start()
            try:
                self.proc.stdin.write(json.dumps(command, allow_nan=False) + "\n")
                self.proc.stdin.flush()
                response = self._responses.get(timeout=30)
            except (OSError, queue.Empty) as exc:
                self._stop()
                raise ValueError("Gym worker did not respond; reset the scene to restart") from exc
            if "error" in response:
                raise ValueError(response["error"])
            return response["result"]

    def _stop(self):
        if self.proc is not None:
            if self.proc.stdin:
                self.proc.stdin.close()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
            self.proc = None

    def close(self):
        with self._lock:
            self._stop()
