"""
vehicle_control.py
MacOS keyboard vehicle controller for ETS2 using PWM (pulse-width modulation)
to simulate analog steering/throttle with digital arrow keys.
"""
import time
import threading
from pynput.keyboard import Controller, Key


_KEY_NAMES = {
    Key.up: "UP",
    Key.down: "DOWN",
    Key.left: "LEFT",
    Key.right: "RIGHT",
}


class VehicleController:
    """
    Smooth vehicle control via PWM on digital keys.
    Steering/Throttle are continuous [-1, 1].

    Key insight for ETS2: keyboard keys need to be held for at least ~60ms
    to register. We use a min-hold-time per key instead of toggling every
    single 33ms cycle.
    """

    def __init__(self, hz: int = 30):
        self.hz = hz
        self.period = 1.0 / hz
        self.kb = Controller()

        self.steering = 0.0
        self.throttle = 0.0

        self._lock = threading.Lock()
        self._running = True
        self._last_cmd_time = time.time()
        self._cycle_counter = 0
        self._pressed_keys = set()
        self._key_press_deadline = {}  # key -> release after this timestamp
        self._last_log_keys = set()
        self._log_counter = 0

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_controls(self, steering: float, throttle: float):
        with self._lock:
            self.steering = max(-1.0, min(1.0, float(steering)))
            self.throttle = max(-1.0, min(1.0, float(throttle)))
            self._last_cmd_time = time.time()

    @property
    def active_keys(self):
        return [_KEY_NAMES.get(k, str(k)) for k in self._pressed_keys]

    def _loop(self):
        while self._running:
            t0 = time.time()

            if time.time() - self._last_cmd_time > 0.6:
                if self._pressed_keys:
                    self._release_all()
                time.sleep(self.period)
                continue

            self._cycle_counter += 1
            with self._lock:
                s = self.steering
                t = self.throttle

            self._apply_axis("STEER", s, Key.left, Key.right)
            self._apply_axis("THROT", t, Key.down, Key.up)

            # Release keys whose hold-time expired
            now = time.time()
            for key in list(self._pressed_keys):
                if now >= self._key_press_deadline.get(key, 0):
                    self._release(key)

            # Log only when key set changes or every ~1.5s
            self._log_counter += 1
            current_keys = set(self._pressed_keys)
            if current_keys != self._last_log_keys or self._log_counter >= 45:
                self._log_counter = 0
                self._last_log_keys = current_keys
                names = self.active_keys
                if not names:
                    names = ["none"]
                print(f"[VC] steer={s:+.2f} thr={t:+.2f} keys=[{','.join(names)}]")

            elapsed = time.time() - t0
            sleep = self.period - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _apply_axis(self, name, value, neg_key, pos_key):
        dead = 0.03
        abs_val = abs(value)

        if abs_val < dead:
            # In dead zone: if key not being held, release both
            if neg_key not in self._pressed_keys:
                self._release(neg_key)
            if pos_key not in self._pressed_keys:
                self._release(pos_key)
            return

        active = pos_key if value > 0 else neg_key
        inactive = neg_key if value > 0 else pos_key

        # Release inactive side immediately (unless it has active hold)
        if inactive not in self._pressed_keys:
            self._release(inactive)

        # Decide hold duration based on intensity
        # Stronger = hold longer. Weaker = tap briefly.
        if abs_val > 0.50:
            hold_ms = 0.120  # 120ms strong hold
        elif abs_val > 0.20:
            hold_ms = 0.080  # 80ms moderate
        else:
            hold_ms = 0.050  # 50ms gentle tap

        # If active key already held, extend its deadline
        if active in self._pressed_keys:
            self._key_press_deadline[active] = time.time() + hold_ms
            return

        # Otherwise press it (but respect a small cooldown between presses)
        last_release = getattr(self, '_last_release_time', {}).get(active, 0)
        if time.time() - last_release < 0.020:
            return  # 20ms minimum gap

        self._press(active)
        self._key_press_deadline[active] = time.time() + hold_ms

    def _press(self, key):
        if key not in self._pressed_keys:
            try:
                self.kb.press(key)
                self._pressed_keys.add(key)
            except Exception as e:
                print(f"[VC] ERROR press {key}: {e}")

    def _release(self, key):
        if key in self._pressed_keys:
            try:
                self.kb.release(key)
                self._pressed_keys.discard(key)
                if not hasattr(self, '_last_release_time'):
                    self._last_release_time = {}
                self._last_release_time[key] = time.time()
            except Exception as e:
                print(f"[VC] ERROR release {key}: {e}")

    def _release_all(self):
        for key in list(self._pressed_keys):
            self._release(key)

    def emergency_stop(self):
        print("[VC] EMERGENCY STOP")
        with self._lock:
            self.steering = 0.0
            self.throttle = -1.0
        self._release_all()
        self._press(Key.down)

    def stop(self):
        print("[VC] Stopping...")
        self._running = False
        self._release_all()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        print("[VC] Stopped")
