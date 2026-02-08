"""Parameter extraction from profiles and design vs as-built comparison."""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


@dataclass
class BenchParams:
    """Parameters for a single bench."""
    bench_number: int
    crest_elevation: float
    crest_distance: float
    toe_elevation: float
    toe_distance: float
    bench_height: float
    face_angle: float
    berm_width: float


@dataclass
class ExtractionResult:
    """Result of parameter extraction for a section."""
    section_name: str
    sector: str
    benches: List[BenchParams] = field(default_factory=list)
    inter_ramp_angle: float = 0.0
    overall_angle: float = 0.0


def extract_parameters(distances, elevations, section_name, sector,
                       resolution=0.5, face_threshold=40.0,
                       berm_threshold=20.0):
    """
    Extract geotechnical parameters from a 2D profile.

    1. Smooth and resample profile at given resolution
    2. Compute local angles between consecutive points
    3. Classify segments as face (>face_threshold) or berm (<berm_threshold)
    4. Extract bench parameters (height, face angle, berm width)
    5. Calculate inter-ramp and overall angles
    """
    result = ExtractionResult(section_name=section_name, sector=sector)

    if len(distances) < 3:
        return result

    # Resample at given resolution
    d_min, d_max = distances.min(), distances.max()
    d_resampled = np.arange(d_min, d_max, resolution)

    if len(d_resampled) < 3:
        return result

    # Interpolate
    f_interp = interp1d(distances, elevations, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    e_resampled = f_interp(d_resampled)

    # Smooth (window ~3 meters)
    window = max(3, int(3.0 / resolution))
    if window % 2 == 0:
        window += 1
    e_smooth = uniform_filter1d(e_resampled, size=window)

    # Compute local angles between consecutive points
    dd = np.diff(d_resampled)
    de = np.diff(e_smooth)
    angles = np.abs(np.degrees(np.arctan2(np.abs(de), dd)))

    # Classify segments
    is_face = angles > face_threshold
    is_berm = angles < berm_threshold

    # Detect benches: face segments followed by berm segments
    benches = []
    bench_num = 0
    i = 0
    n = len(angles)

    while i < n:
        if is_face[i]:
            face_start = i
            while i < n and is_face[i]:
                i += 1
            face_end = i

            crest_idx = face_start
            toe_idx = min(face_end, len(d_resampled) - 1)

            crest_d = d_resampled[crest_idx]
            crest_e = e_smooth[crest_idx]
            toe_d = d_resampled[toe_idx]
            toe_e = e_smooth[toe_idx]

            bench_height = abs(crest_e - toe_e)
            horiz_dist = abs(toe_d - crest_d)

            if bench_height < 2.0:
                continue

            if horiz_dist > 0:
                face_angle = np.degrees(np.arctan2(bench_height, horiz_dist))
            else:
                face_angle = 90.0

            # Find berm width after the toe
            berm_width = 0.0
            if i < n:
                berm_start = i
                while i < n and is_berm[i]:
                    i += 1
                berm_end = i
                if berm_end > berm_start:
                    berm_width = abs(
                        d_resampled[min(berm_end, len(d_resampled) - 1)] -
                        d_resampled[berm_start]
                    )

            bench_num += 1
            benches.append(BenchParams(
                bench_number=bench_num,
                crest_elevation=float(crest_e),
                crest_distance=float(crest_d),
                toe_elevation=float(toe_e),
                toe_distance=float(toe_d),
                bench_height=float(bench_height),
                face_angle=float(face_angle),
                berm_width=float(berm_width),
            ))
        else:
            i += 1

    result.benches = benches

    # Calculate inter-ramp and overall angles
    if len(benches) >= 2:
        first = benches[0]
        last = benches[-1]
        total_height = abs(first.crest_elevation - last.toe_elevation)
        total_dist = abs(last.toe_distance - first.crest_distance)
        if total_dist > 0:
            result.overall_angle = float(
                np.degrees(np.arctan2(total_height, total_dist)))
            result.inter_ramp_angle = result.overall_angle
    elif len(benches) == 1:
        result.overall_angle = benches[0].face_angle
        result.inter_ramp_angle = benches[0].face_angle

    return result


def _evaluate_status(deviation, tol_neg, tol_pos):
    """
    Evaluate compliance using tripartite system.
    CUMPLE: within tolerance
    FUERA DE TOLERANCIA: up to 1.5x tolerance
    NO CUMPLE: exceeds 1.5x tolerance
    """
    if deviation < 0:
        limit = tol_neg
    else:
        limit = tol_pos

    abs_dev = abs(deviation)
    if abs_dev <= limit:
        return "CUMPLE"
    elif abs_dev <= limit * 1.5:
        return "FUERA DE TOLERANCIA"
    else:
        return "NO CUMPLE"


def compare_design_vs_asbuilt(params_design, params_topo, tolerances):
    """
    Compare design vs as-built parameters bench by bench.
    Returns list of comparison dicts.
    """
    comparisons = []
    n_compare = min(len(params_design.benches), len(params_topo.benches))

    for i in range(n_compare):
        bd = params_design.benches[i]
        bt = params_topo.benches[i]

        height_dev = bt.bench_height - bd.bench_height
        angle_dev = bt.face_angle - bd.face_angle
        berm_dev = bt.berm_width - bd.berm_width

        tol_h = tolerances['bench_height']
        tol_a = tolerances['face_angle']
        tol_b = tolerances['berm_width']

        comparisons.append({
            'sector': params_design.sector,
            'section': params_design.section_name,
            'bench_num': bd.bench_number,
            'level': f"{bd.crest_elevation:.0f}",
            'height_design': round(bd.bench_height, 2),
            'height_real': round(bt.bench_height, 2),
            'height_dev': round(height_dev, 2),
            'height_status': _evaluate_status(height_dev, tol_h['neg'],
                                              tol_h['pos']),
            'angle_design': round(bd.face_angle, 1),
            'angle_real': round(bt.face_angle, 1),
            'angle_dev': round(angle_dev, 1),
            'angle_status': _evaluate_status(angle_dev, tol_a['neg'],
                                             tol_a['pos']),
            'berm_design': round(bd.berm_width, 2),
            'berm_real': round(bt.berm_width, 2),
            'berm_dev': round(berm_dev, 2),
            'berm_status': _evaluate_status(berm_dev, tol_b['neg'],
                                            tol_b['pos']),
        })

    return comparisons
