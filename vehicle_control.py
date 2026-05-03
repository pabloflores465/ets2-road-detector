"""
vehicle_control.py
MacOS keyboard vehicle controller for ETS2.
Synchronous simple approach: press/release keys directly on every set_controls() call.
"""
from pynput.keyboard import Controller, Key


_KEY_NAMES = {
    Key.up: "UP",
    Key.down: "DOWN",
    Key.left: "LEFT",
    Key.right: "RIGHT",
}


class VehicleController:
    """
    Synchronous vehicle control via keyboard arrow keys.
    Caller must invoke set_controls() every frame (~30Hz).
    """

    def __init__(self, hz: int = 30):
        self.kb = Controller()
        self._steering = 0.0
        self._throttle = 0.0
        self._active_keys = set()

    def set_controls(self, steering: float, throttle: float):
        """
        steering: -1.0 (full left) to +1.0 (full right)
        throttle: -1.0 (full brake) to +1.0 (full accel)
        """
        self._steering = max(-1.0, min(1.0, float(steering)))
        self._throttle = max(-1.0, min(1.0, float(throttle)))

        dead = 0.01
        keys_changed = []

        # Steering
        if abs(self._steering) < dead:
            if Key.left in self._active_keys:
                self.kb.release(Key.left)
                self._active_keys.discard(Key.left)
                keys_changed.append("-LEFT")
            if Key.right in self._active_keys:
                self.kb.release(Key.right)
                self._active_keys.discard(Key.right)
                keys_changed.append("-RIGHT")
        elif self._steering < 0:
            if Key.right in self._active_keys:
                self.kb.release(Key.right)
                self._active_keys.discard(Key.right)
                keys_changed.append("-RIGHT")
            if Key.left not in self._active_keys:
                self.kb.press(Key.left)
                self._active_keys.add(Key.left)
                keys_changed.append("+LEFT")
        else:
            if Key.left in self._active_keys:
                self.kb.release(Key.left)
                self._active_keys.discard(Key.left)
                keys_changed.append("-LEFT")
            if Key.right not in self._active_keys:
                self.kb.press(Key.right)
                self._active_keys.add(Key.right)
                keys_changed.append("+RIGHT")

        # Throttle
        if abs(self._throttle) < dead:
            if Key.up in self._active_keys:
                self.kb.release(Key.up)
                self._active_keys.discard(Key.up)
                keys_changed.append("-UP")
            if Key.down in self._active_keys:
                self.kb.release(Key.down)
                self._active_keys.discard(Key.down)
                keys_changed.append("-DOWN")
        elif self._throttle > 0:
            if Key.down in self._active_keys:
                self.kb.release(Key.down)
                self._active_keys.discard(Key.down)
                keys_changed.append("-DOWN")
            if Key.up not in self._active_keys:
                self.kb.press(Key.up)
                self._active_keys.add(Key.up)
                keys_changed.append("+UP")
        else:
            if Key.up in self._active_keys:
                self.kb.release(Key.up)
                self._active_keys.discard(Key.up)
                keys_changed.append("-UP")
            if Key.down not in self._active_keys:
                self.kb.press(Key.down)
                self._active_keys.add(Key.down)
                keys_changed.append("+DOWN")

        if keys_changed:
            names = self.active_keys
            if not names:
                names = ["none"]
            print(f"[VC] S={self._steering:+.2f} T={self._throttle:+.2f} keys=[{','.join(names)}] chg={keys_changed}")

    @property
    def active_keys(self):
        return [_KEY_NAMES.get(k, str(k)) for k in self._active_keys]

    def emergency_stop(self):
        print("[VC] EMERGENCY STOP")
        for k in list(self._active_keys):
            self.kb.release(k)
        self._active_keys.clear()
        self.kb.press(Key.down)
        self._active_keys.add(Key.down)

    def stop(self):
        print("[VC] Stopping...")
        for k in list(self._active_keys):
            self.kb.release(k)
        self._active_keys.clear()
        print("[VC] Stopped")
