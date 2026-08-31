"""
Probana backend for p-kit.

Communicates with the Probana physical p-bit computer over USB serial.

Protocol commands:
  PING
  CLEAR <n>
  H <i> <value>
  J <from> <to> <value>
  COMMIT
  RUN <samples> <burn_in> <thin>

J and h are uploaded once. During RUN, the p-bit update loop,
calibration correction and physical sampling are performed locally
on the Probana board.
"""

import time
import numpy as np
from .numpy_backend import NumpyBackend


class ProbanaBackend(NumpyBackend):
    """Backend for the Probana physical p-bit computer."""

    protocol_name = "PROBANA"
    protocol_version = 1
    supports_native_pbits = True

    def __init__(self, port=None, baudrate=115200, timeout=2.0,
                 startup_timeout=60.0, dtype=None):
        super().__init__(dtype=dtype)
        try:
            import serial
            from serial.tools import list_ports
        except ImportError as e:
            raise ImportError("Install pyserial: pip install pyserial") from e

        self.serial, self.list_ports = serial, list_ports
        self.timeout = timeout
        self.port = port or self._find_port()
        self.ser = serial.Serial(self.port, baudrate, timeout=0.2)
        self.n_pbits = self._wait_ready(startup_timeout)
        self.ser.timeout = timeout

    def _find_port(self):
        ports = list(self.list_ports.comports())
        if not ports:
            raise RuntimeError("No serial device found")
        if len(ports) == 1:
            return ports[0].device

        keys = ("arduino", "rp2", "pico", "esp32", "cp210", "ch340")
        found = [p for p in ports
                 if any(k in (p.description or "").lower() for k in keys)]
        if len(found) == 1:
            return found[0].device

        raise RuntimeError(
            "Multiple serial devices found; specify port: " +
            ", ".join(p.device for p in ports)
        )

    def _send(self, cmd):
        self.ser.write((cmd + "\n").encode("ascii"))
        self.ser.flush()

    def _read(self, timeout=None):
        end = time.monotonic() + (timeout or self.timeout)
        while time.monotonic() < end:
            raw = self.ser.readline()
            if raw:
                return raw.decode("ascii", errors="replace").strip()
        raise TimeoutError("Probana did not respond")

    def _parse_ready(self, line):
        p = line.split()
        if len(p) != 5 or p[0] not in ("READY", "OK") or \
           p[1] != self.protocol_name or p[4] != "WORKING":
            return None

        if int(p[2]) != self.protocol_version:
            raise RuntimeError(f"Unsupported protocol version {p[2]}")
        return int(p[3])

    def _wait_ready(self, timeout):
        end, next_ping = time.monotonic() + timeout, 0
        while time.monotonic() < end:
            now = time.monotonic()
            if now >= next_ping:
                self._send("PING")
                next_ping = now + 0.5

            raw = self.ser.readline()
            if not raw:
                continue

            line = raw.decode("ascii", errors="replace").strip()
            n = self._parse_ready(line)
            if n is not None:
                return n
            if line.startswith("ERR "):
                raise RuntimeError(line)

        raise TimeoutError("Probana did not enter WORKING mode")

    def _ok(self):
        while True:
            line = self._read()
            if line == "OK":
                return
            if line.startswith("ERR "):
                raise RuntimeError(line)

    def ping(self):
        self._send("PING")
        line = self._read()
        n = self._parse_ready(line)
        if n is None:
            raise RuntimeError(f"Invalid PING response: {line}")
        return n

    def load_circuit(self, J, h):
        J = np.asarray(J, dtype=float)
        h = np.asarray(h, dtype=float).reshape(-1)

        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError("J must be square")
        if J.shape[0] != h.size:
            raise ValueError("J and h sizes do not match")
        if h.size > self.n_pbits:
            raise ValueError(
                f"Circuit needs {h.size} p-bits; Probana has {self.n_pbits}"
            )

        self._send(f"CLEAR {h.size}"); self._ok()

        for i, v in enumerate(h):
            if v != 0:
                self._send(f"H {i} {v:.9g}"); self._ok()

        rows, cols = np.nonzero(J)
        for src, dst in zip(rows, cols):
            self._send(f"J {src} {dst} {J[src,dst]:.9g}"); self._ok()

        self._send("COMMIT"); self._ok()

    def run_circuit(self, J, h, samples, burn_in=100, thin=1):
        self.load_circuit(J, h)
        n = len(h)

        self._send(f"RUN {int(samples)} {int(burn_in)} {int(thin)}")
        self._ok()
        states = []

        while True:
            line = self._read(max(self.timeout, self.timeout * samples))
            if line == "DONE":
                break
            if line.startswith("ERR "):
                raise RuntimeError(line)
            if not line.startswith("S "):
                continue

            bits = line[2:].strip()
            if len(bits) != n or any(b not in "01" for b in bits):
                raise RuntimeError(f"Invalid sample: {line}")
            states.append([1 if b == "1" else -1 for b in bits])

        if len(states) != samples:
            raise RuntimeError(f"Expected {samples} samples, got {len(states)}")
        return np.asarray(states, dtype=np.int8)

    def close(self):
        if getattr(self, "ser", None) and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()