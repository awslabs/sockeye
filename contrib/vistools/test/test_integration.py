import os
import pytest
from filecmp import dircmp

from generate_graphs import generate

CWD = os.path.dirname(os.path.realpath(__file__))

BEAM_COMPARISONS = [(os.path.join(CWD, "resources", "beams.json"), 
                     os.path.join(CWD, "resources", "output"))]

@pytest.mark.parametrize("beams, expected_output", BEAM_COMPARISONS)
def test_encoding_end2end(beams, expected_output, tmpdir):
    generate(beams, str(tmpdir))

    # Same files in each dir
    result = dircmp(expected_output, str(tmpdir))
    assert result.left_list == result.right_list
