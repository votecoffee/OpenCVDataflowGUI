"""An unsupported native Python representation is evidence, not completed reuse."""

import runpy
from pathlib import Path


def test_pyramid_binding_audit_is_bounded_and_records_actual_support():
    source = Path(__file__).resolve().parents[1] / "scripts/audit_optical_flow_pyramids.py"
    result = runpy.run_path(str(source))["audit"]()
    assert result["input_unchanged"]
    assert len(result["construction"]) == 2 and len(result["consumption"]) == 12
    for build in result["construction"]:
        levels = build["actual_max_level"] + 1
        assert 1 <= levels <= 4
        step = 2 if build["with_derivatives"] else 1
        assert len(build["arrays"]) == levels * step
        for index in range(levels):
            shape = [(129 + 2**index - 1) // 2**index, (161 + 2**index - 1) // 2**index]
            assert build["arrays"][index * step] == {"shape": shape, "dtype": "uint8"}
            if step == 2:
                assert build["arrays"][index * step + 1] == {"shape": [*shape, 2], "dtype": "int16"}
    assert result["python_cached_consumer_qualified"] == any(
        item.get("identity_points_match", False) for item in result["consumption"]
    )
