import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox import qm_descriptors
from qktox.qm_descriptors import (
    XTBRunSettings,
    build_initial_geometry,
    get_or_calculate_xtb_descriptors,
    parse_xtb_output,
)


SAMPLE_XTB_OUTPUT = """
       -------------------------------------------------
      |                Property Printout               |
       -------------------------------------------------
     #    Occupation            Energy/Eh            Energy/eV
  -------------------------------------------------------------
     1        2.0000           -0.6801050             -18.5066
     2        2.0000           -0.5683264             -15.4649
     3        2.0000           -0.5108650             -13.9013
     4        2.0000           -0.4463919             -12.1469 (HOMO)
     5                          0.0826818               2.2499 (LUMO)
              HL-Gap            0.5290737 Eh           14.3968 eV
 molecular dipole:
                x           y           z       tot (Debye)
 q only:        0.481       0.000       0.340
   full:        0.696       0.000       0.492       2.167
 Mol. alpha(0) /au        :        9.429351
 -------------------------------------------------
| TOTAL ENERGY               -5.070322476938 Eh   |
| HOMO-LUMO GAP              14.3968 eV            |
 -------------------------------------------------
"""


def test_parse_xtb_output_extracts_intensive_and_size_normalized_values():
    values = parse_xtb_output(SAMPLE_XTB_OUTPUT, heavy_atom_count=10)

    assert values["homo_ev"] == pytest.approx(-12.1469)
    assert values["lumo_ev"] == pytest.approx(2.2499)
    assert values["homo_lumo_gap_ev"] == pytest.approx(14.3968)
    assert values["molecular_dipole_debye"] == pytest.approx(2.167)
    assert values["molecular_polarizability_au"] == pytest.approx(9.429351)
    assert values["total_energy_hartree"] == pytest.approx(-5.070322476938)
    assert values["total_energy_hartree_per_heavy_atom"] == pytest.approx(
        -0.5070322476938
    )


def test_parse_xtb_output_fails_when_required_properties_are_missing():
    with pytest.raises(qm_descriptors.QMCalculationError, match="missing a required"):
        parse_xtb_output("| TOTAL ENERGY -5.0 Eh |", heavy_atom_count=2)


def test_etkdg_geometry_is_deterministic_and_preserves_charge_and_stereochemistry():
    smiles = "C[C@H](O)F"
    xyz_first, metadata_first = build_initial_geometry(
        "herg_karim_000123", smiles, seed=19
    )
    xyz_second, metadata_second = build_initial_geometry(
        "herg_karim_000123", smiles, seed=19
    )

    assert xyz_first == xyz_second
    assert metadata_first == metadata_second
    assert metadata_first["formal_charge"] == 0
    assert metadata_first["conformer_count"] == 1
    assert metadata_first["additional_protonation_or_tautomerization"] is False
    assert metadata_first["input_canonical_isomeric_smiles"] == "C[C@H](O)F"

    _, charged_metadata = build_initial_geometry(
        "herg_karim_000124", "C[NH3+]", seed=19
    )
    assert charged_metadata["formal_charge"] == 1


def test_cache_reuses_only_exact_molecule_structure_and_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = XTBRunSettings(
        executable=tmp_path / "xtb.exe",
        seed=7,
        optimization_level="normal",
        timeout_seconds=30,
        xtb_version="6.8.0 (test)",
        executable_sha256="exe-sha",
        gfn2_parameters_sha256="parameters-sha",
    )
    calls = []

    def calculate(molecule_id, smiles, run_settings):
        calls.append((molecule_id, smiles, run_settings.seed))
        geometry = "1\nH\nH 0 0 0\n"
        return {
            "descriptors": {"homo_ev": -2.0},
            "optimized_geometry_xyz": geometry,
            "optimized_geometry_sha256": qm_descriptors.hashlib.sha256(
                geometry.encode()
            ).hexdigest(),
            "calculation_seconds": 0.25,
        }

    monkeypatch.setattr(qm_descriptors, "calculate_xtb_descriptors", calculate)
    cache_dir = tmp_path / "cache"
    first, first_hit = get_or_calculate_xtb_descriptors(
        "herg_karim_000001",
        "C",
        settings=settings,
        cache_dir=cache_dir,
    )
    second, second_hit = get_or_calculate_xtb_descriptors(
        "herg_karim_000001",
        "C",
        settings=settings,
        cache_dir=cache_dir,
    )
    _, changed_structure_hit = get_or_calculate_xtb_descriptors(
        "herg_karim_000001",
        "N",
        settings=settings,
        cache_dir=cache_dir,
    )

    assert first_hit is False
    assert second_hit is True
    assert changed_structure_hit is False
    assert first == second
    assert len(calls) == 2
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 2
    assert json.loads(cache_files[0].read_text(encoding="utf-8"))["cache_identity"][
        "molecule_id"
    ] == "herg_karim_000001"


def test_molecule_id_and_multifragment_structures_are_rejected():
    with pytest.raises(ValueError, match="molecule ID"):
        build_initial_geometry("drug-1", "C", seed=1)
    with pytest.raises(qm_descriptors.QMCalculationError, match="multiple fragments"):
        build_initial_geometry("herg_karim_000001", "C.C", seed=1)
