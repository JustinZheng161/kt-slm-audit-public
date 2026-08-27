import importlib.util
from pathlib import Path

path = Path(__file__).with_name('test-optimized-dkt.py')
spec = importlib.util.spec_from_file_location('test_optimized_dkt', path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

module.test_layernorm_residual_dkt_shape_and_gradients()
module.test_temporal_attention_dkt_shape_with_padding_mask()
print('optimized-dkt-smoke: PASS')
