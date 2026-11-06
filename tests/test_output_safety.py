import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qktox.output_safety import refuse_existing_outputs


def test_refuse_existing_outputs_reports_conflicting_path(tmp_path, capsys):
    existing_output = tmp_path / "existing.csv"
    existing_output.write_text("existing", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        refuse_existing_outputs([tmp_path / "new.csv", existing_output])

    assert error.value.code == 1
    assert str(existing_output) in capsys.readouterr().err


def test_refuse_existing_outputs_allows_new_paths(tmp_path):
    refuse_existing_outputs([tmp_path / "new.csv"])
