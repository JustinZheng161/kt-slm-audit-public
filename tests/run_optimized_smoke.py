from test_optimized_dkt import (
    test_layernorm_residual_dkt_shape_and_gradients,
    test_temporal_attention_dkt_shape_with_padding_mask,
)

test_layernorm_residual_dkt_shape_and_gradients()
test_temporal_attention_dkt_shape_with_padding_mask()
print('optimized_dkt_smoke: PASS')
