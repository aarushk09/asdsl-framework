"""Windows IoRing availability detection.

Checks whether Windows IoRing (introduced in Windows 11 22H2, build 22621+)
is available on the current system by:
  1. Reading the Windows version from the registry (build number >= 22621)
  2. Checking for IoRingCreateIoRing export in KernelBase.dll

Usage:
    python asdsl/io/iouring_detect.py
"""
from __future__ import annotations

import ctypes
import sys


def is_iouring_available() -> tuple[bool, str]:
    """Return (available: bool, reason: str).

    Must not raise — all exceptions are caught and reported in the reason string.
    """
    if sys.platform != "win32":
        return False, "not Windows"

    # Step 1: check Windows build number
    try:
        import winreg  # noqa: PLC0415
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
        )
        build_str, _ = winreg.QueryValueEx(key, "CurrentBuildNumber")
        build = int(build_str)
        winreg.CloseKey(key)
    except Exception as exc:
        return False, f"Could not read Windows version: {exc}"

    if build < 22621:
        return False, f"Windows build {build} < 22621 (Windows 11 22H2 required for IoRing)"

    # Step 2: check KernelBase.dll for IoRingCreateIoRing export
    try:
        kb = ctypes.WinDLL("KernelBase.dll")
        _ = kb.IoRingCreateIoRing  # AttributeError if symbol absent
        return True, f"Windows build {build}, IoRingCreateIoRing available"
    except AttributeError:
        return False, (
            f"Build {build} >= 22621 but IoRingCreateIoRing not found in "
            "KernelBase.dll — may need a newer Windows update"
        )
    except OSError as exc:
        return False, f"KernelBase.dll load failed: {exc}"


if __name__ == "__main__":
    available, reason = is_iouring_available()
    print(f"IoRing available: {available}")
    print(f"Reason:           {reason}")
    if not available:
        print("[IoRing] async streaming will use thread-based fallback")
