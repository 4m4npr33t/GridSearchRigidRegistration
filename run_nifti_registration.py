#!/usr/bin/env python3
"""Convert two NIfTI-1 volumes to project raw files and compare CPU/CUDA runs.

This script has no third-party dependencies. It supports single-file .nii and
.nii.gz NIfTI-1 images, applies the header's intensity scaling, writes
little-endian float32 voxels, runs the native executables, and compares results.
"""

from __future__ import annotations

import argparse
import array
import gzip
import json
import math
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


RESULT_PATTERN = re.compile(
    r"^(?P<backend>CPU result|CPU|Naive CUDA|Optimized CUDA):\s*"
    r"\(tx,ty,tz\)=\((?P<tx>[-+0-9.eE]+),(?P<ty>[-+0-9.eE]+),"
    r"(?P<tz>[-+0-9.eE]+)\).*?"
    r"\(rx,ry,rz\)=\((?P<rx>[-+0-9.eE]+),(?P<ry>[-+0-9.eE]+),"
    r"(?P<rz>[-+0-9.eE]+)\).*?NCC=(?P<ncc>[-+0-9.eE]+)"
)
TIME_PATTERN = re.compile(r"(?:median time|wall)=(?P<wall_ms>[-+0-9.eE]+)\s*ms")

# NIfTI datatype number -> Python array type code and bytes per value.
NIFTI_DTYPES = {
    2: ("B", 1, "uint8"),
    4: ("h", 2, "int16"),
    8: ("i", 4, "int32"),
    16: ("f", 4, "float32"),
    64: ("d", 8, "float64"),
    256: ("b", 1, "int8"),
    512: ("H", 2, "uint16"),
    768: ("I", 4, "uint32"),
    1024: ("q", 8, "int64"),
    1280: ("Q", 8, "uint64"),
}


@dataclass
class NiftiVolume:
    path: Path
    shape: Tuple[int, int, int]
    voxel_sizes: Tuple[float, float, float]
    affine: Optional[List[List[float]]]
    source_dtype: str
    slope: float
    intercept: float
    voxels: array.array


@dataclass
class RegistrationResult:
    executable: str
    backend: str
    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float
    ncc: float
    wall_ms: Optional[float]

    @property
    def parameters(self) -> Tuple[float, float, float, float, float, float]:
        return (self.tx, self.ty, self.tz, self.rx, self.ry, self.rz)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert fixed/moving NIfTI files and compare CPU/CUDA registration."
    )
    parser.add_argument("fixed", type=Path, help="Reference NIfTI volume")
    parser.add_argument("moving", type=Path, help="NIfTI volume to align")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("converted"),
        help="Directory for raw volumes, CSV histories, and summary JSON",
    )
    parser.add_argument("--cpu-exe", type=Path, help="Path to cpu_registration executable")
    parser.add_argument("--cuda-exe", type=Path, help="Path to cuda_registration executable")
    parser.add_argument("--max-translation", type=float, default=12.0)
    parser.add_argument("--max-rotation", type=float, default=12.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Keep every Nth voxel per axis before registration (useful for large volumes)",
    )
    parser.add_argument(
        "--no-build", action="store_true",
        help="Do not try to build missing native executables with CMake",
    )
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    return parser.parse_args()


def unpack_from(endian: str, format_code: str, payload: bytes, offset: int) -> Tuple[Any, ...]:
    return struct.unpack_from(endian + format_code, payload, offset)


def read_nifti(path: Path) -> NiftiVolume:
    if not path.is_file():
        raise FileNotFoundError("NIfTI file does not exist: {}".format(path))
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rb") as source:
        payload = source.read()
    if len(payload) < 352:
        raise ValueError("File is too small to be NIfTI-1: {}".format(path))

    if struct.unpack_from("<i", payload, 0)[0] == 348:
        endian = "<"
    elif struct.unpack_from(">i", payload, 0)[0] == 348:
        endian = ">"
    else:
        raise ValueError("Unsupported NIfTI header (sizeof_hdr is not 348): {}".format(path))
    if payload[344:347] != b"n+1":
        raise ValueError("Only single-file NIfTI-1 .nii/.nii.gz images are supported: {}".format(path))

    dimensions = unpack_from(endian, "8h", payload, 40)
    dimension_count = dimensions[0]
    if dimension_count < 3 or dimension_count > 7:
        raise ValueError("Expected a 3-D NIfTI, header reports {} dimensions".format(dimension_count))
    extra_dimensions = dimensions[4:dimension_count + 1]
    if any(value != 1 for value in extra_dimensions):
        raise ValueError("Expected a 3-D NIfTI; higher dimensions are {}".format(extra_dimensions))
    shape = tuple(int(value) for value in dimensions[1:4])
    if any(value <= 0 for value in shape):
        raise ValueError("Invalid NIfTI shape: {}".format(shape))

    datatype = unpack_from(endian, "h", payload, 70)[0]
    bitpix = unpack_from(endian, "h", payload, 72)[0]
    if datatype not in NIFTI_DTYPES:
        raise ValueError("Unsupported NIfTI datatype code {} in {}".format(datatype, path))
    type_code, bytes_per_value, source_dtype = NIFTI_DTYPES[datatype]
    if bitpix != bytes_per_value * 8:
        raise ValueError("NIfTI datatype/bitpix mismatch in {}".format(path))

    voxel_count = shape[0] * shape[1] * shape[2]
    voxel_offset = int(round(unpack_from(endian, "f", payload, 108)[0]))
    byte_count = voxel_count * bytes_per_value
    if voxel_offset < 348 or voxel_offset + byte_count > len(payload):
        raise ValueError("Voxel payload is truncated or has an invalid vox_offset: {}".format(path))

    values = array.array(type_code)
    values.frombytes(payload[voxel_offset:voxel_offset + byte_count])
    source_is_little = endian == "<"
    host_is_little = sys.byteorder == "little"
    if bytes_per_value > 1 and source_is_little != host_is_little:
        values.byteswap()

    slope = float(unpack_from(endian, "f", payload, 112)[0])
    intercept = float(unpack_from(endian, "f", payload, 116)[0])
    if slope == 0.0 or not math.isfinite(slope):
        slope = 1.0
    if not math.isfinite(intercept):
        intercept = 0.0
    voxels = array.array("f", (float(value) * slope + intercept for value in values))
    if not all(math.isfinite(value) for value in voxels):
        raise ValueError("Scaled volume contains NaN or infinity: {}".format(path))

    pixdim = unpack_from(endian, "8f", payload, 76)
    voxel_sizes = tuple(abs(float(pixdim[index])) for index in range(1, 4))
    sform_code = unpack_from(endian, "h", payload, 254)[0]
    affine = None
    if sform_code > 0:
        affine = [
            list(map(float, unpack_from(endian, "4f", payload, 280))),
            list(map(float, unpack_from(endian, "4f", payload, 296))),
            list(map(float, unpack_from(endian, "4f", payload, 312))),
            [0.0, 0.0, 0.0, 1.0],
        ]
    return NiftiVolume(
        path=path.resolve(), shape=shape, voxel_sizes=voxel_sizes, affine=affine,
        source_dtype=source_dtype, slope=slope, intercept=intercept, voxels=voxels,
    )


def write_cpp_raw(volume: NiftiVolume, path: Path) -> None:
    """NIfTI and main.cu both store x as the fastest-changing coordinate."""
    values = array.array("f", volume.voxels)
    if sys.byteorder != "little":
        values.byteswap()
    with path.open("wb") as destination:
        values.tofile(destination)


def downsample(volume: NiftiVolume, stride: int) -> NiftiVolume:
    if stride == 1:
        return volume
    width, height, depth = volume.shape
    sampled = array.array("f")
    for z in range(0, depth, stride):
        for y in range(0, height, stride):
            row = (z * height + y) * width
            sampled.extend(volume.voxels[row:row + width:stride])
    shape = tuple((dimension + stride - 1) // stride for dimension in volume.shape)
    voxel_sizes = tuple(size * stride for size in volume.voxel_sizes)
    affine = None
    if volume.affine is not None:
        affine = [list(row) for row in volume.affine]
        for row in range(3):
            for column in range(3):
                affine[row][column] *= stride
    return NiftiVolume(
        path=volume.path, shape=shape, voxel_sizes=voxel_sizes, affine=affine,
        source_dtype=volume.source_dtype, slope=volume.slope,
        intercept=volume.intercept, voxels=sampled,
    )


def volume_metadata(volume: NiftiVolume) -> Dict[str, Any]:
    return {
        "path": str(volume.path),
        "shape_xyz": list(volume.shape),
        "source_dtype": volume.source_dtype,
        "dtype_written": "little-endian float32",
        "nifti_scaling": {"slope": volume.slope, "intercept": volume.intercept},
        "voxel_sizes": list(volume.voxel_sizes),
        "sform_affine": volume.affine,
        "value_range": [float(min(volume.voxels)), float(max(volume.voxels))],
    }


def close_sequences(a: Sequence[float], b: Sequence[float], tolerance: float = 1e-5) -> bool:
    return len(a) == len(b) and all(
        abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))
        for left, right in zip(a, b)
    )


def flatten_affine(affine: Optional[List[List[float]]]) -> Optional[List[float]]:
    if affine is None:
        return None
    return [value for row in affine for value in row]


def executable_candidates(project: Path, name: str) -> List[Path]:
    suffix = ".exe" if sys.platform == "win32" else ""
    filename = name + suffix
    return [
        project / "build" / "Release" / filename,
        project / "build" / filename,
        project / "cmake-build-release" / filename,
        project / "cmake-build-debug" / filename,
    ]


def resolve_executable(explicit: Optional[Path], project: Path, name: str) -> Optional[Path]:
    candidates = [explicit] if explicit else executable_candidates(project, name)
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    return None


def try_cmake_build(project: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        print("CMake is not on PATH; native executables cannot be built automatically.")
        return
    build_dir = project / "build"
    commands = [
        [cmake, "-S", str(project), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        [cmake, "--build", str(build_dir), "--config", "Release"],
    ]
    print("Native executable missing; attempting a CMake build.")
    for command in commands:
        completed = subprocess.run(command)
        if completed.returncode != 0:
            print("CMake build stopped with status {}.".format(completed.returncode))
            return


def parse_results(output: str, executable: str) -> List[RegistrationResult]:
    results = []
    for line in output.splitlines():
        match = RESULT_PATTERN.search(line)
        if not match:
            continue
        values = match.groupdict()
        timing = TIME_PATTERN.search(line)
        backend = "CPU" if values["backend"] == "CPU result" else values["backend"]
        results.append(RegistrationResult(
            executable=executable, backend=backend,
            tx=float(values["tx"]), ty=float(values["ty"]), tz=float(values["tz"]),
            rx=float(values["rx"]), ry=float(values["ry"]), rz=float(values["rz"]),
            ncc=float(values["ncc"]),
            wall_ms=float(timing.group("wall_ms")) if timing else None,
        ))
    return results


def run_backend(
    executable: Path,
    label: str,
    fixed_raw: Path,
    moving_raw: Path,
    shape: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[List[RegistrationResult], str]:
    width, height, depth = shape
    csv_path = args.output_dir / "{}_history.csv".format(label)
    command = [
        str(executable), "--fixed", str(fixed_raw), "--moving", str(moving_raw),
        "--width", str(width), "--height", str(height), "--depth", str(depth),
        "--max-translation", str(args.max_translation),
        "--max-rotation", str(args.max_rotation), "--repeats", str(args.repeats),
        "--csv", str(csv_path),
    ]
    print("\nRunning {}: {}".format(label, executable), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    combined = completed.stdout + completed.stderr
    print(combined, end="" if combined.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise RuntimeError("{} exited with status {}".format(label, completed.returncode))
    results = parse_results(combined, label)
    if not results:
        raise RuntimeError("Could not parse a registration result from {} output".format(label))
    return results, combined


def print_comparison(results: List[RegistrationResult]) -> None:
    if not results:
        return
    reference = results[0]
    print("\nComparison (first result is the reference):")
    print(
        "{:<29} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>11} {:>11} {:>11}".format(
            "Executable/backend", "tx", "ty", "tz", "rx", "ry", "rz",
            "NCC", "|dparam|", "dNCC",
        )
    )
    for result in results:
        parameter_delta = math.sqrt(sum(
            (value - reference_value) ** 2
            for value, reference_value in zip(result.parameters, reference.parameters)
        ))
        ncc_delta = abs(result.ncc - reference.ncc)
        name = "{}/{}".format(result.executable, result.backend)
        print(
            "{:<29} {:7.2f} {:7.2f} {:7.2f} {:7.2f} {:7.2f} {:7.2f} "
            "{:11.7f} {:11.4g} {:11.4g}".format(
                name, result.tx, result.ty, result.tz, result.rx, result.ry, result.rz,
                result.ncc, parameter_delta, ncc_delta,
            )
        )


def main() -> int:
    args = parse_args()
    if (args.max_translation < 0 or args.max_rotation < 0 or
            args.repeats <= 0 or args.stride <= 0):
        raise ValueError("Search bounds must be nonnegative; repeats and stride must be positive")

    project = Path(__file__).resolve().parent
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = read_nifti(args.fixed)
    moving = read_nifti(args.moving)
    if fixed.shape != moving.shape:
        raise ValueError(
            "The native executables require equal shapes: fixed={}, moving={}. "
            "Resample the images onto a common grid first.".format(fixed.shape, moving.shape)
        )
    original_shape = fixed.shape
    fixed = downsample(fixed, args.stride)
    moving = downsample(moving, args.stride)

    fixed_raw = args.output_dir / "fixed.float32.raw"
    moving_raw = args.output_dir / "moving.float32.raw"
    write_cpp_raw(fixed, fixed_raw)
    write_cpp_raw(moving, moving_raw)
    metadata = {
        "fixed": volume_metadata(fixed), "moving": volume_metadata(moving),
        "original_shape_xyz": list(original_shape),
        "sampling_stride": args.stride,
        "sampling_method": "every Nth voxel without anti-alias filtering",
        "raw_layout": "index = (z * height + y) * width + x",
        "fixed_raw": str(fixed_raw), "moving_raw": str(moving_raw),
    }
    metadata_path = args.output_dir / "conversion_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Converted shape (x,y,z): {}".format(fixed.shape))
    print("Fixed raw:  {}".format(fixed_raw))
    print("Moving raw: {}".format(moving_raw))
    print("Metadata:   {}".format(metadata_path))

    if not close_sequences(fixed.voxel_sizes, moving.voxel_sizes):
        print("WARNING: voxel sizes differ; main.cu ignores physical voxel spacing.")
    if not close_sequences(fixed.voxel_sizes, [fixed.voxel_sizes[0]] * 3):
        print("WARNING: voxels are anisotropic; main.cu treats them as isotropic.")
    fixed_affine = flatten_affine(fixed.affine)
    moving_affine = flatten_affine(moving.affine)
    if fixed_affine is not None and moving_affine is not None:
        if not close_sequences(fixed_affine, moving_affine):
            print("NOTE: NIfTI affines differ; main.cu registers voxel arrays and ignores origins/affines.")

    cpu_exe = resolve_executable(args.cpu_exe, project, "cpu_registration")
    cuda_exe = resolve_executable(args.cuda_exe, project, "cuda_registration")
    needs_cpu = not args.skip_cpu and cpu_exe is None
    needs_cuda = not args.skip_cuda and cuda_exe is None
    if (needs_cpu or needs_cuda) and not args.no_build:
        try_cmake_build(project)
        cpu_exe = resolve_executable(args.cpu_exe, project, "cpu_registration")
        cuda_exe = resolve_executable(args.cuda_exe, project, "cuda_registration")
    all_results = []
    raw_outputs = {}
    if not args.skip_cpu:
        if cpu_exe:
            results, output = run_backend(cpu_exe, "cpp", fixed_raw, moving_raw, fixed.shape, args)
            all_results.extend(results)
            raw_outputs["cpp"] = output
        else:
            print("CPU executable not found; build cpu_registration or pass --cpu-exe.")
    if not args.skip_cuda:
        if cuda_exe:
            results, output = run_backend(cuda_exe, "cu", fixed_raw, moving_raw, fixed.shape, args)
            all_results.extend(results)
            raw_outputs["cu"] = output
        else:
            print("CUDA executable not found; build cuda_registration or pass --cuda-exe.")

    print_comparison(all_results)
    summary = {
        "metadata": metadata,
        "results": [asdict(result) for result in all_results],
        "program_output": raw_outputs,
    }
    summary_path = args.output_dir / "registration_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Comparison JSON: {}".format(summary_path))
    return 0 if all_results else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Error: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
