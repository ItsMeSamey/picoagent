"""Transport parsing tests; no live provider calls or credentials."""
import importlib.util
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
spec = importlib.util.spec_from_file_location('colab_persistent_stage_test', Path(__file__).resolve().parents[1] / 'scripts/colab_persistent_stage.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_structured_result_can_span_stream_fragments():
    assert module.parse_result([{'output_type':'stream', 'text':'PICOAGENT_'},
                               {'output_type':'stream', 'text':'RESULT={"verified":true}\n'}]) == {'verified': True}


@pytest.mark.parametrize('outputs', [[],
    [{'output_type':'stream', 'text':'PICOAGENT_RESULT=[]\n'}],
    [{'output_type':'stream', 'text':'PICOAGENT_RESULT={}\nPICOAGENT_RESULT={}\n'}],
    [{'output_type':'error', 'evalue':'private diagnostic'}]])
def test_missing_ambiguous_or_error_result_rejected_without_private_output(outputs):
    with pytest.raises(RuntimeError) as error:
        module.parse_result(outputs)
    assert 'private diagnostic' not in str(error.value)
