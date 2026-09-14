from __future__ import annotations


import re
from pathlib import Path
from typing import Any
import shutil
import subprocess




def read_text(root: Path, rel: str) -> str | None:
    p = Path(root) / rel.lstrip("/")
    try:
        return p.read_text(errors="replace").strip("\x00").strip()
    except (OSError, UnicodeDecodeError):
        return None


def run(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None


    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None


    if out.returncode != 0:
        return None


    return out.stdout.strip()




def unknown(source: str, why: str) -> dict[str, Any]:
    return {
        "value": None,
        "source": source,
        "status": "unknown",
        "detail": why
    }




def probe_module_model(root: Path = Path("/")) -> dict[str, Any]:
    src = "/proc/device-tree/model"
    raw = read_text(root, src)


    if not raw:
        return unknown(
            src,
            "device tree model node absent — not a Jetson, or /proc not mounted"
        )


    return {
        "value": raw,
        "source": src,
        "status": "ok"
    }




def probe_memory_total_kb(root: Path = Path("/")) -> dict[str, Any]:
    src = "/proc/meminfo"


    text = read_text(root, src)


    if not text:
        return unknown(src, "could not read /proc/meminfo")


    m = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.MULTILINE)


    if not m:
        return unknown(src, "MemTotal not found")


    return {
        "value": int(m.group(1)),
        "source": src,
        "status": "ok"
    }




def probe_root_source(root: Path = Path("/")) -> dict[str, Any]:
    src = "/proc/mounts"


    text = read_text(root, src)


    if not text:
        return unknown(src, "could not read /proc/mounts")


    for line in text.splitlines():
        parts = line.split()


        if len(parts) < 2:
            continue


        device = parts[0]
        mountpoint = parts[1]


        if mountpoint == "/":
            if device.startswith("/dev/nvme"):
                kind = "nvme"
            elif device.startswith("/dev/mmcblk") or device.startswith("/dev/sd"):
                kind = "removable_or_sata"
            else:
                kind = "other"


            return {
                "value": device,
                "kind": kind,
                "source": src,
                "status": "ok"
            }


    return unknown(src, "no root mount entry found in mount table")




def probe_nvme_present(root: Path = Path("/")) -> dict[str, Any]:
    src = "/sys/block/nvme0n1"


    nvme_path = Path(root) / "sys/block/nvme0n1"


    present = nvme_path.exists()


    model = None


    if present:
        model = read_text(root, "/sys/block/nvme0n1/device/model")


    return {
        "value": present,
        "model": model,
        "source": src,
        "status": "ok"
    }


_SPEED_RE = re.compile(r"Speed\s+([\d.]+)GT/s")
_WIDTH_RE = re.compile(r"Width\s+x(\d+)")


_GEN_BY_GTS = {
    2.5: 1,
    5.0: 2,
    8.0: 3,
    16.0: 4,
    32.0: 5,
    64.0: 6
}




def _parse_link_line(line: str) -> dict[str, Any]:
    speed = _SPEED_RE.search(line)
    width = _WIDTH_RE.search(line)


    gts = float(speed.group(1)) if speed else None


    return {
        "raw": line.strip(),
        "gts": gts,
        "width": int(width.group(1)) if width else None,
        "gen": _GEN_BY_GTS.get(gts) if gts is not None else None,
    }


def probe_pcie_link(
    root: Path = Path("/"),
    lspci_output: str | None = None
) -> dict[str, Any]:


    # Step 1: execute lspci -vv
    src = "lspci -vv"


    # Step 2: save output to a variable
    text = lspci_output if lspci_output is not None else run(["lspci", "-vv"])


    if not text:
        return unknown(src, "could not read lspci output")


    # Step 3: extract LnkCap line
    lnkcap_line = None
    for line in text.splitlines():
        if "LnkCap:" in line:
            lnkcap_line = line.strip()
            break


    # Step 4: extract LnkSta line
    lnksta_line = None
    for line in text.splitlines():
        if "LnkSta:" in line:
            lnksta_line = line.strip()
            break


    if lnksta_line is None:
        return unknown(src, "LnkSta not found")


    # Step 5: parse capability and negotiated values
    negotiated = _parse_link_line(lnksta_line)


    if lnkcap_line is not None:
        capability = _parse_link_line(lnkcap_line)
    else:
        capability = None


    # Step 6: generate interpretation
    interpretation = None


    if capability is not None:
        if (
            capability["gen"] is not None
            and negotiated["gen"] is not None
        ):
            if capability["gen"] > negotiated["gen"]:
                interpretation = (
                    f"drive capable of Gen{capability['gen']}, "
                    f"link running at Gen{negotiated['gen']} — "
                    "expected on this carrier board, "
                    "whose M.2 Key-M slot is wired Gen3 x4"
                )
            else:
                interpretation = (
                    f"link running at its full capability, "
                    f"Gen{negotiated['gen']} "
                    f"x{negotiated['width']}"
                )


    return {
        "value": negotiated["gts"],
        "negotiated": negotiated,
        "capability": capability,
        "interpretation": interpretation,
        "source": src,
        "status": "ok"
    }


def probe_thermal_zones(root: Path = Path("/")) -> dict[str, Any]:
    src = "/sys/class/thermal/thermal_zone*/temp"
    base = Path(root) / "sys/class/thermal"


    if not base.exists():
        return unknown(src, "thermal directory not found")


    zones = []


    for zone_path in base.glob("thermal_zone*"):
        zone_name = zone_path.name


        try:
            temp_text = read_text(
                root,
                f"/sys/class/thermal/{zone_name}/temp"
            )


            type_text = read_text(
                root,
                f"/sys/class/thermal/{zone_name}/type"
            )
        except TypeError:
            continue


        if temp_text is None:
            continue


        try:
            temp_c = float(temp_text) / 1000.0
        except ValueError:
            continue


        zones.append({
            "zone": zone_name,
            "type": type_text,
            "temp_c": temp_c
        })


    if not zones:
        return unknown(src, "no readable thermal zones found")


    hottest = max(zone["temp_c"] for zone in zones)


    return {
        "value": hottest,
        "zones": zones,
        "source": src,
        "status": "ok"
    }


def probe_power_mode(
    root: Path = Path("/"),
    nvpmodel_output: str | None = None
) -> dict[str, Any]:


    src = "nvpmodel -q"


    text = nvpmodel_output if nvpmodel_output is not None else run(["nvpmodel", "-q"])


    if not text:
        return unknown(src, "could not read nvpmodel output")


    mode_match = re.search(r"NV Power Mode:\s*(.+)", text)


    if not mode_match:
        return unknown(src, "power mode name not found")


    mode_name = mode_match.group(1).strip()


    mode_id_match = re.search(
        r"^\s*(\d+)\s*$",
        text,
        re.MULTILINE
    )


    mode_id = int(mode_id_match.group(1)) if mode_id_match else None


    return {
        "value": mode_name,
        "mode_id": mode_id,
        "source": src,
        "status": "ok"
    }


if __name__ == "__main__":
    print(probe_module_model())
    print(probe_memory_total_kb())
    print(probe_root_source())
    print(probe_nvme_present())
    print(probe_pcie_link())
    print(probe_thermal_zones())
    print(probe_power_mode())
