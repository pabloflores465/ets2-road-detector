"""
vehicle_control.py
MacOS keyboard vehicle controller for ETS2 using PWM (pulse-width modulation)
to simulate analog steering/throttle with digital arrow keys.

Arrow Up    = accelerate
Arrow Down  = brake / reverse
Arrow Left  = steer left
Arrow Right = steer right
"""
import time
import threading
from pynput.keyboard import Controller, Key


class VehicleController:
    """
    Smooth vehicle control via PWM on digital keys.

    Steering/Throttle are continuous [-1, 1]. We translate to key presses:
      - |value| < 0.05     -> dead zone, release both keys
      - 0.05 < |value| < 0.25 -> tap every 4th cycle (gentle micro-adjust)
      - 0.25 < |value| < 0.55 -> tap every 2nd cycle (moderate)
      - |value| > 0.55      -> hold every cycle (strong)

    Cycle rate is configurable (default 30 Hz = ~33 ms per cycle).
    """

    def __init__(self, hz: int = 30):
        self.hz = hz
        self.period = 1.0 / hz
        self.kb = Controller()

        self.steering = 0.0   # -1 = full left, +1 = full right
        self.throttle = 0.0   # -1 = full brake, +1 = full accel

        self._lock = threading.Lock()
        self._running = True
        self._last_cmd_time = time.time()
        self._cycle_counter = 0
        self._pressed_keys = set()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_controls(self, steering: float, throttle: float):
        """Update desired steering [-1,1] and throttle [-1,1]."""
        with self._lock:
            self.steering = max(-1.0, min(1.0, float(steering)))
            self.throttle = max(-1.0, min(1.0, float(throttle)))
            self._last_cmd_time = time.time()

    def _loop(self):
        while self._running:
            t0 = time.time()

            # Watchdog: if no commands for > 0.6 s, emergency release
            if time.time() - self._last_cmd_time > 0.6:
                self._release_all()
                time.sleep(self.period)
                continue

            self._cycle_counter += 1
            with self._lock:
                s = self.steering
                t = self.throttle

            self._apply_axis(s, Key.left, Key.right)
            self._apply_axis(t, Key.down, Key.up)

            elapsed = time.time() - t0
            sleep = self.period - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _apply_axis(self, value, neg_key, pos_key):
        """PWM key control for a single axis."""
        dead = 0.05
        abs_val = abs(value)

        if abs_val < dead:
            self._release(neg_key)
            self._release(pos_key)
            return

        active = pos_key if value > 0 else neg_key
        inactive = neg_key if value > 0 else pos_key
        self._release(inactive)

        # Determine duty based on intensity
        if abs_val > 0.55:
            duty = 1          # every cycle
        elif abs_val > 0.25:
            duty = 2          # every 2nd cycle
        else:
            duty = 4          # every 4th cycle (micro-adjust)

        if self._cycle_counter % duty == 0:
            self._press(active)
        else:
            self._release(active)

    def _press(self, key):
        if key not in self._pressed_keys:
            try:
                self.kb.press(key)
                self._pressed_keys.add(key)
            except Exception:
                pass

    def _release(self, key):
        if key in self._pressed_keys:
            try:
                self.kb.release(key)
                self._pressed_keys.discard(key)
            except Exception:
                pass

    def _release_all(self):
        for key in list(self._pressed_keys):
            self._release(key)

    def emergency_stop(self):
        """Hard brake, release steering."""
        with self._lock:
            self.steering = 0.0
            self.throttle = -1.0
        self._release_all()
        self._press(Key.down)

    def stop(self):
        """Shutdown controller, release all keys."""
        self._running = False
        self._release_all()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
