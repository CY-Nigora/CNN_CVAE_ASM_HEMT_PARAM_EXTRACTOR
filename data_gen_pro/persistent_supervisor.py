# persistent_supervisor.py
import os
import time
import json
import queue
import psutil
import threading
import subprocess
from typing import Dict, Any, Optional, TextIO
import re

# ------------------ ToolBox functions ------------------

# when restart due to timeout, read the dead loop number in last run
def get_last_loop_number(log_path: str) -> int | None:
    """
    Extract the current loop number from the last [child-log] line in the log file.
    For example:
    "[child-log] >> Process 1 :: Loop 5/5 :: used time: 1.16 s"
    will return 5.
    If no match is found, returns None.
    """
    pattern = re.compile(r"Loop\s+(\d+)/(\d+)")  # catch “Loop x/y”
    last_line = None

    # only read last several KB（avoid too slow when reading big files）
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        block = min(8192, size)
        f.seek(-block, 2)
        lines = f.read().decode(errors="ignore").splitlines()

    # backward search for the last relevant line
    for line in reversed(lines):
        if "[child-log]" in line and "Loop" in line:
            last_line = line
            break

    if not last_line:
        return None

    match = pattern.search(last_line)
    if match:
        current, total = match.groups()
        return int(current)
    return None


def default_log_file() -> str:
    """default log dir (cur working path)"""
    return os.path.join(os.getcwd(), "ads_supervisor.log")


class SafeLogger:
    """Thread-safe logger: output to console and file simultaneously"""
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or default_log_file()
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def write(self, msg: str):
        """write log to file"""
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg)
                f.flush()

    def log(self, *args, end="\n", flush=True, prefix_time=True):
        """Console + File Output"""
        text = " ".join(str(a) for a in args)
        if prefix_time:
            # ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
            ts = ''
            text = f"{ts} {text}"
        print(text, end=end, flush=flush)
        self.write(text + end)


# ------------------ kill rest ADS simulation process ------------------

ADS_PROC_NAMES = ["hpeesofsim.exe", "hpeesofsimengine.exe"]

def kill_ads_sim_processes(verbose: bool = True, logger: Optional[SafeLogger] = None):
    killed = []
    for p in psutil.process_iter(['pid', 'name']):
        try:
            name = (p.info.get('name') or "").lower()
            if any(name == x.lower() for x in ADS_PROC_NAMES):
                msg = f"[cleanup] Terminating {name} pid={p.pid}"
                if verbose:
                    (logger.log if logger else print)(msg)
                p.terminate()
                killed.append(p)
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            msg = f"[cleanup] error: {e}"
            (logger.log if logger else print)(msg)
    gone, alive = psutil.wait_procs(killed, timeout=5)
    for p in alive:
        msg = f"[cleanup] Killing (force) pid={p.pid}"
        (logger.log if logger else print)(msg)
        try:
            p.kill()
        except Exception:
            pass


# ------------------ Core Supervision ------------------

class AdsWorker:
    """
    Persistent ADS child process supervisor.
    - stdout: JSON protocol only.
    - stderr: Logs.
    - timeout: Inactivity timeout.
    - All logs are written to the specified file.
    """

    def __init__(
        self,
        ads_python_exe: str,
        worker_script: str,
        extra_env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        log_file: Optional[str] = None,
    ):
        self.ads_python_exe = ads_python_exe
        self.worker_script = worker_script
        self.extra_env = extra_env or {}
        self.cwd = cwd

        # externally accessible log file path
        self.logger = SafeLogger(log_file)

        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._json_queue: "queue.Queue[str]" = queue.Queue()
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._last_activity = time.time()

    # ---------- life cycle ----------
    def start(self):
        if self.proc and self.proc.poll() is None:
            return

        env = os.environ.copy()
        env.update(self.extra_env)
        cmd = [self.ads_python_exe, self.worker_script]
        # self.logger.log(f"[worker] spawn: {cmd}")

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=self.cwd,
        )

        self._last_activity = time.time()

        # stdout pump thread: handles JSON or non-JSON
        def _pump_stdout(pipe):
            try:
                for line in iter(pipe.readline, ''):
                    if not line:
                        break
                    s = line.strip()
                    self._last_activity = time.time()
                    if s.startswith("{") or s.startswith("["):
                        self._json_queue.put(s)
                    else:
                        self.logger.log("[child-stdout-log]", s)
            except Exception as e:
                self.logger.log("[child-stdout-pump-err]", e)

        # stderr pump thread: log
        def _pump_stderr(pipe):
            try:
                for line in iter(pipe.readline, ''):
                    if not line:
                        break
                    self._last_activity = time.time()
                    self.logger.log("[child-log]", line.rstrip())
            except Exception as e:
                self.logger.log("[child-stderr-pump-err]", e)

        self._stdout_thread = threading.Thread(target=_pump_stdout, args=(self.proc.stdout,), daemon=True)
        self._stderr_thread = threading.Thread(target=_pump_stderr, args=(self.proc.stderr,), daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

        hello = self._read_json_poll(inactivity_timeout=20, hard_timeout=30)
        if not hello:
            raise RuntimeError("ADS worker failed to start (no hello/connection).")
        self.logger.log("[worker connection]", json.dumps(hello, ensure_ascii=False))

    def stop(self, graceful: bool = True):
        if not self.proc:
            return
        if self.proc.poll() is None:
            try:
                if graceful:
                    self._send({"cmd": "shutdown"})
                    self.proc.wait(timeout=5)
                else:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
            except Exception:
                pass

        try:
            if self._stdout_thread and self._stdout_thread.is_alive():
                self._stdout_thread.join(timeout=1)
            if self._stderr_thread and self._stderr_thread.is_alive():
                self._stderr_thread.join(timeout=1)
        except Exception:
            pass

        self.proc = None
        self._stdout_thread = None
        self._stderr_thread = None

        try:
            while not self._json_queue.empty():
                self._json_queue.get_nowait()
        except Exception:
            pass

    def restart(self):
        self.stop(graceful=False)
        kill_ads_sim_processes(logger=self.logger)
        time.sleep(1.0)
        self.start()

    # ---------- I/O ----------
    def _send(self, obj: dict):
        if not self.proc or self.proc.poll() is not None:
            raise RuntimeError("worker not running")
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def _read_json_poll(self, inactivity_timeout: float, hard_timeout: Optional[float] = None) -> Optional[dict]:
        t_start = time.time()
        POLL_STEP = 0.5
        while True:
            if hard_timeout is not None and (time.time() - t_start) > hard_timeout:
                return None
            if (time.time() - self._last_activity) > inactivity_timeout:
                return None
            try:
                s = self._json_queue.get(timeout=POLL_STEP)
            except queue.Empty:
                continue
            try:
                return json.loads(s)
            except Exception:
                self.logger.log("[child-stdout-nonjson?]", s)
        return None

    # ---------- API ----------
    def run_task(self, args: dict, timeout: int = 900, hard_timeout: Optional[int] = None) -> dict:
        with self._lock:
            self._last_activity = time.time()
            self._send({"cmd": "run", "args": args})
            self.logger.log(f"[worker] task start (timeout={timeout}s)")
            rep = self._read_json_poll(inactivity_timeout=timeout, hard_timeout=hard_timeout)
            if rep is None:
                self.logger.log("[worker] task timeout")
                raise TimeoutError("inactivity timeout")
            self.logger.log("[worker] task done:", rep)
            return rep
