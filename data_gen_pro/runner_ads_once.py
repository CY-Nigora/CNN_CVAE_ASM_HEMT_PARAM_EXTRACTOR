# worker_ads.py — Executed in the ADS built-in Python program, always resident
# Purpose: Reads JSON data from stdin, calls your simulation function, and outputs the JSON result to stdout

import sys, json, traceback, time
from ads_driver import singel_process_iteration_data_gen2h5

def run_one(args: dict) -> dict:
    singel_process_iteration_data_gen2h5(**args)
    time.sleep(0.1)
    return {"active": True, "note": "finished one run"}  # can also return paths/stats

def _write(msg: dict):
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def main():
    _write({"Connection": True, "msg": "ADS worker ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            _write({"active": False, "error": "bad_json"})
            continue

        if req.get("cmd") == "ping":
            _write({"active": True, "pong": True, "ts": time.time()})
            continue
        if req.get("cmd") == "shutdown":
            _write({"active": True, "bye": True})
            break
        if req.get("cmd") == "run":
            try:
                res = run_one(req.get("args", {}))
                _write({"active": True, "res": res})
            except Exception:
                _write({"active": False, "trace": traceback.format_exc()})
        else:
            _write({"active": False, "error": "unknown_cmd", "cmd": req.get("cmd")})

if __name__ == "__main__":
    main()
