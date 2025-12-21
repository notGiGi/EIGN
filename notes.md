EIGN
    loop.py:
        AMP-torch.amp provides convenience methods for mixed precision, where some operations use the torch.float32 (float) datatype and other operations use lower precision floating point datatype (lower_precision_fp): torch.float16 (half) or torch.bfloat16. 
                use_amp, amp_dtype, use_scaler = _get_amp_settings(device)

        Gradscaler- If the forward pass for a particular op has float16 inputs, the backward pass for that op will produce float16 gradients. Gradient values with small magnitudes may not be representable in float16. These values will flush to zero (“underflow”), so the update for the corresponding parameters will be lost.

        To prevent underflow, “gradient scaling” multiplies the network’s loss(es) by a scale factor and invokes a backward pass on the scaled loss(es). Gradients flowing backward through the network are then scaled by the same factor. In other words, gradient values have a larger magnitude, so they don’t flush to zero.

        Each parameter’s gradient (.grad attribute) should be unscaled before the optimizer updates the parameters, so the scale factor does not interfere with the learning rate.


        when AMP????

        _build_param_groups -  helper that splits parameters into 2 groups to apply a weight decay only to big weights.





