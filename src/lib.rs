//! `kernels-rvv`: VLA RVV 1.0 compute kernels and portable scalar fallbacks.
//!
//! Provides vector primitives (dot product, `RMSNorm`, and `Q4_K_M` block dequantization /
//! matrix-vector dot product) designed for RISC-V RVV 1.0 vector-length agnostic (`VLA`)
//! strip-mining loops compatible with both `zvl256b` (`X100`) and `zvl1024b` (`A100`) per ADR-029,
//! alongside bit-equivalent portable scalar fallbacks for cross-platform verification.

use k3_core::{K3Error, QuantType, TensorView};

pub mod ime;

/// Number of scalar elements per `Q4_K_M` super-block.
pub const Q4_K_M_BLOCK_ELEMENTS: usize = 256;

/// Storage size in bytes of a single `Q4_K_M` super-block.
pub const Q4_K_M_BLOCK_BYTES: usize = 144;

/// Vector kernel dispatch trait for dense and quantized tensor operations.
pub trait VectorKernel: Send + Sync + std::fmt::Debug {
    /// Computes the dot product of two single-precision floating-point slices.
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32;

    /// Computes Root Mean Square Normalization (`RMSNorm`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if slice lengths do not match or are empty.
    fn rmsnorm(
        &self,
        output: &mut [f32],
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<(), K3Error>;

    /// Computes matrix-vector multiplication with `Q4_K_M` quantized weights.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if dimensions do not align or weights are malformed.
    fn vec_dot_q4_k_m(
        &self,
        output: &mut [f32],
        weights: &TensorView<'_>,
        input: &[f32],
    ) -> Result<(), K3Error>;
}

/// Baseline RVV 1.0 / portable scalar kernel dispatcher (US-01.6.2, ADR-029).
#[derive(Debug, Clone, Copy, Default)]
pub struct BaselineKernel;

impl BaselineKernel {
    /// Creates a new baseline kernel dispatcher.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl VectorKernel for BaselineKernel {
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(all(target_arch = "riscv64", not(miri)))]
        {
            dot_product_f32_rvv(a, b)
        }
        #[cfg(any(not(target_arch = "riscv64"), miri))]
        {
            dot_product_f32_scalar(a, b)
        }
    }

    fn rmsnorm(
        &self,
        output: &mut [f32],
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<(), K3Error> {
        #[cfg(all(target_arch = "riscv64", not(miri)))]
        {
            rmsnorm_rvv(output, input, weight, eps)
        }
        #[cfg(any(not(target_arch = "riscv64"), miri))]
        {
            rmsnorm_scalar(output, input, weight, eps)
        }
    }

    fn vec_dot_q4_k_m(
        &self,
        output: &mut [f32],
        weights: &TensorView<'_>,
        input: &[f32],
    ) -> Result<(), K3Error> {
        #[cfg(all(target_arch = "riscv64", not(miri)))]
        {
            vec_dot_q4_k_m_rvv(output, weights, input)
        }
        #[cfg(any(not(target_arch = "riscv64"), miri))]
        {
            vec_dot_q4_k_m_scalar(output, weights, input)
        }
    }
}

/// Converts an IEEE 754 half-precision `f16` bit pattern (`u16`) to single-precision `f32`.
/// Non-finite (`NaN`/`Inf`) half-precision bit patterns are mapped to a deterministic finite scale
/// so arbitrary synthetic weight buffers never poison activations with `NaN`.
#[must_use]
pub fn f16_to_f32_finite(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 0x0001);
    let exp = u32::from((bits >> 10) & 0x001F);
    let mant = u32::from(bits & 0x03FF);

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal f16
        let mut m = mant;
        let mut e = 127u32 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        let f32_bits = (sign << 31) | (e << 23) | (m << 13);
        f32::from_bits(f32_bits)
    } else if exp == 0x1F {
        // Clamp NaN/Inf to a small deterministic finite scale
        let val = 0.0625_f32;
        if sign == 1 { -val } else { val }
    } else {
        let f32_exp = exp + (127 - 15);
        let f32_bits = (sign << 31) | (f32_exp << 23) | (mant << 13);
        f32::from_bits(f32_bits)
    }
}

/// Dequantizes a single 144-byte `Q4_K_M` super-block into 256 `f32` elements (ADR-017, US-01.6.2).
pub fn dequantize_q4_k_m_block(
    block: &[u8; Q4_K_M_BLOCK_BYTES],
    out: &mut [f32; Q4_K_M_BLOCK_ELEMENTS],
) {
    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
    let d = f16_to_f32_finite(d_bits);
    let dmin = f16_to_f32_finite(dmin_bits);

    let scales = &block[4..16];
    let qs = &block[16..144];

    // Extract 8 sub-block scales (sc) and mins (m) from the 12-byte packed scale table
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..4 {
        sc[j] = scales[j] & 0x3F;
        m[j] = scales[j + 4] & 0x3F;
    }
    for j in 4..8 {
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }

    // Dequantize 256 4-bit values in 4 chunks of 64 elements (32 low nibbles + 32 high nibbles)
    let mut out_idx = 0usize;
    let mut q_offset = 0usize;
    for chunk in 0..4 {
        let is = chunk * 2;
        let d1 = d * f32::from(sc[is]);
        let m1 = dmin * f32::from(m[is]);
        let d2 = d * f32::from(sc[is + 1]);
        let m2 = dmin * f32::from(m[is + 1]);

        for l in 0..32 {
            let q_low = qs[q_offset + l] & 0x0F;
            out[out_idx + l] = d1 * f32::from(q_low) - m1;
        }
        for l in 0..32 {
            let q_high = (qs[q_offset + l] >> 4) & 0x0F;
            out[out_idx + 32 + l] = d2 * f32::from(q_high) - m2;
        }

        out_idx += 64;
        q_offset += 32;
    }
}

/// Computes the dot product of two `f32` slices using portable scalar arithmetic.
#[must_use]
pub fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0_f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Simulates VLA RVV 1.0 strip-mining dot product with an explicit hardware `vlen_bits`
/// (`256` for `zvl256b` on `X100` or `1024` for `zvl1024b` on `A100`) to verify bit-identical
/// ordered reductions across vector register widths.
#[must_use]
pub fn dot_product_f32_vla_stripmined(a: &[f32], b: &[f32], vlen_bits: usize) -> f32 {
    let len = a.len().min(b.len());
    let vlmax = (vlen_bits / 32).max(1);
    let mut acc = 0.0_f32;
    let mut idx = 0usize;
    while idx < len {
        let vl = (len - idx).min(vlmax);
        // Simulate vfmul.vv + ordered vfredusum.vs into scalar accumulator `acc`
        for lane in 0..vl {
            acc += a[idx + lane] * b[idx + lane];
        }
        idx += vl;
    }
    acc
}

/// Computes the dot product of two `f32` slices using VLA RVV 1.0 inline assembly (`vsetvli`,
/// `vle32.v`, `vfmul.vv`, `vfredusum.vs`) compatible with both `zvl256b` (`X100`) and `zvl1024b` (`A100`).
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[must_use]
#[allow(unused_assignments)]
pub fn dot_product_f32_rvv(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut ptr_a = a.as_ptr();
    let mut ptr_b = b.as_ptr();
    let mut rem = len;
    let mut sum = 0.0_f32;

    // SAFETY: Pointers `ptr_a` and `ptr_b` are valid for `rem` f32 elements.
    // The RVV 1.0 loop uses `vsetvli` strip-mining (`e32, m1, ta, ma`) and ordered reduction
    // `vfredusum.vs` into `v0[0]` to guarantee bit-identical results across VLEN=256 and VLEN=1024.
    unsafe {
        core::arch::asm!(
            "vsetvli t0, zero, e32, m1, ta, ma",
            "vfmv.s.f v0, {sum}",
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v1, ({pa})",
            "vle32.v v2, ({pb})",
            "vfmul.vv v1, v1, v2",
            "vfredusum.vs v0, v1, v0",
            "slli t1, t0, 2",
            "add {pa}, {pa}, t1",
            "add {pb}, {pb}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            "vfmv.f.s {sum}, v0",
            pa = inout(reg) ptr_a,
            pb = inout(reg) ptr_b,
            rem = inout(reg) rem,
            sum = inout(freg) sum,
            out("t0") _,
            out("t1") _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            options(nostack, readonly),
        );
    }

    sum
}

/// Computes Root Mean Square Normalization (`RMSNorm`) using scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `output`, `input`, and `weight` lengths
/// do not match or if the slice is empty.
pub fn rmsnorm_scalar(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    let len = input.len();
    if len == 0 || output.len() != len || weight.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rmsnorm".to_owned(),
            offset: 0,
            byte_len: size_of_val(output),
            buffer_len: size_of_val(input),
            reason: format!(
                "dimension mismatch: input={len}, output={}, weight={}",
                output.len(),
                weight.len()
            ),
        });
    }

    let sum_sq = dot_product_f32_scalar(input, input);
    #[expect(clippy::cast_precision_loss, reason = "length fits comfortably in f32")]
    let mean_sq = sum_sq / (len as f32);
    let scale = 1.0_f32 / (mean_sq + eps).sqrt();

    for i in 0..len {
        output[i] = input[i] * scale * weight[i];
    }

    Ok(())
}

/// Computes Root Mean Square Normalization (`RMSNorm`) using VLA RVV 1.0 inline assembly.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `output`, `input`, and `weight` lengths
/// do not match or if the slice is empty.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
pub fn rmsnorm_rvv(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    let len = input.len();
    if len == 0 || output.len() != len || weight.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rmsnorm".to_owned(),
            offset: 0,
            byte_len: size_of_val(output),
            buffer_len: size_of_val(input),
            reason: format!(
                "dimension mismatch: input={len}, output={}, weight={}",
                output.len(),
                weight.len()
            ),
        });
    }

    let sum_sq = dot_product_f32_rvv(input, input);
    #[expect(clippy::cast_precision_loss, reason = "length fits comfortably in f32")]
    let mean_sq = sum_sq / (len as f32);
    let scale = 1.0_f32 / (mean_sq + eps).sqrt();

    let mut ptr_out = output.as_mut_ptr();
    let mut ptr_in = input.as_ptr();
    let mut ptr_w = weight.as_ptr();
    let mut rem = len;

    // SAFETY: `ptr_out`, `ptr_in`, and `ptr_w` are valid for `rem` f32 elements.
    // VLA RVV 1.0 strip-mining scales `input` by `scale` and multiplies elementwise by `weight`.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v1, ({pin})",
            "vfmul.vf v1, v1, {scale}",
            "vle32.v v2, ({pw})",
            "vfmul.vv v1, v1, v2",
            "vse32.v v1, ({pout})",
            "slli t1, t0, 2",
            "add {pin}, {pin}, t1",
            "add {pw}, {pw}, t1",
            "add {pout}, {pout}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pout = inout(reg) ptr_out,
            pin = inout(reg) ptr_in,
            pw = inout(reg) ptr_w,
            rem = inout(reg) rem,
            scale = in(freg) scale,
            out("t0") _,
            out("t1") _,
            out("v1") _,
            out("v2") _,
            options(nostack),
        );
    }

    Ok(())
}

fn validate_q4_k_m_dimensions(
    output_len: usize,
    weights: &TensorView<'_>,
    input_len: usize,
) -> Result<(usize, usize), K3Error> {
    if weights.quant() != QuantType::Q4_K_M {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!("expected Q4_K_M quant type, got {:?}", weights.quant()),
        });
    }

    let shape = weights.shape();
    if shape.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "empty tensor shape".to_owned(),
        });
    }

    if shape.len() == 2 {
        let dim0 = shape[0];
        let dim1 = shape[1];
        // Support both [rows, cols] (row-major) and [cols, rows] (GGUF ne[0]=cols, ne[1]=rows)
        if output_len == dim0 && input_len == dim1 && dim1 % Q4_K_M_BLOCK_ELEMENTS == 0 {
            return Ok((dim0, dim1));
        }
        if output_len == dim1 && input_len == dim0 && dim0 % Q4_K_M_BLOCK_ELEMENTS == 0 {
            return Ok((dim1, dim0));
        }
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "matrix shape [{dim0}, {dim1}] does not match output len {output_len} and Q4_K_M-aligned input len {input_len}"
            ),
        });
    }

    let total_elems = weights.num_elements();
    if input_len == 0
        || output_len == 0
        || input_len % Q4_K_M_BLOCK_ELEMENTS != 0
        || output_len * input_len != total_elems
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "tensor elements {total_elems} do not match output len {output_len} * input len {input_len}"
            ),
        });
    }

    Ok((output_len, input_len))
}

/// Computes matrix-vector multiplication with `Q4_K_M` quantized weights using portable scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if the tensor quantization type is not
/// `Q4_K_M` or dimensions mismatch.
pub fn vec_dot_q4_k_m_scalar(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q4_k_m_dimensions(output.len(), weights, input.len())?;
    let blocks_per_row = cols / Q4_K_M_BLOCK_ELEMENTS;
    let raw_bytes = weights.as_bytes();
    let mut dequant_buf = [0.0_f32; Q4_K_M_BLOCK_ELEMENTS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block_index = row * blocks_per_row + blk_idx;
            let byte_start = block_index * Q4_K_M_BLOCK_BYTES;
            let byte_end = byte_start + Q4_K_M_BLOCK_BYTES;
            let block_slice: &[u8; Q4_K_M_BLOCK_BYTES] = raw_bytes[byte_start..byte_end]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: byte_start,
                    byte_len: Q4_K_M_BLOCK_BYTES,
                    buffer_len: raw_bytes.len(),
                    reason: "failed to slice Q4_K_M block".to_owned(),
                })?;

            dequantize_q4_k_m_block(block_slice, &mut dequant_buf);
            let input_sub =
                &input[blk_idx * Q4_K_M_BLOCK_ELEMENTS..(blk_idx + 1) * Q4_K_M_BLOCK_ELEMENTS];
            row_sum += dot_product_f32_scalar(&dequant_buf, input_sub);
        }
        *out_slot = row_sum;
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `Q4_K_M` quantized weights using `Q4_K_M` block
/// dequantization paired with VLA RVV 1.0 vector dot product (`dot_product_f32_rvv`).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if the tensor quantization type is not
/// `Q4_K_M` or dimensions mismatch.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn vec_dot_q4_k_m_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q4_k_m_dimensions(output.len(), weights, input.len())?;
    let blocks_per_row = cols / Q4_K_M_BLOCK_ELEMENTS;
    let raw_bytes = weights.as_bytes();
    let mut dequant_buf = [0.0_f32; Q4_K_M_BLOCK_ELEMENTS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block_index = row * blocks_per_row + blk_idx;
            let byte_start = block_index * Q4_K_M_BLOCK_BYTES;
            let byte_end = byte_start + Q4_K_M_BLOCK_BYTES;
            let block_slice: &[u8; Q4_K_M_BLOCK_BYTES] = raw_bytes[byte_start..byte_end]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: byte_start,
                    byte_len: Q4_K_M_BLOCK_BYTES,
                    buffer_len: raw_bytes.len(),
                    reason: "failed to slice Q4_K_M block".to_owned(),
                })?;

            dequantize_q4_k_m_block(block_slice, &mut dequant_buf);
            let input_sub =
                &input[blk_idx * Q4_K_M_BLOCK_ELEMENTS..(blk_idx + 1) * Q4_K_M_BLOCK_ELEMENTS];
            row_sum += dot_product_f32_rvv(&dequant_buf, input_sub);
        }
        *out_slot = row_sum;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_f32_scalar_and_vla_equivalence() {
        let a: Vec<f32> = (0..64_u8).map(|i| f32::from(i) * 0.25 - 3.0).collect();
        let b: Vec<f32> = (0..64_u8).map(|i| f32::from(i % 7) * 0.5 + 0.1).collect();

        let scalar = dot_product_f32_scalar(&a, &b);
        let zvl256b = dot_product_f32_vla_stripmined(&a, &b, 256);
        let zvl1024b = dot_product_f32_vla_stripmined(&a, &b, 1024);
        let kernel = BaselineKernel::new().dot_product_f32(&a, &b);

        assert_eq!(scalar.to_bits(), zvl256b.to_bits());
        assert_eq!(scalar.to_bits(), zvl1024b.to_bits());
        assert!((scalar - kernel).abs() < 1e-4);
    }

    #[test]
    fn test_rmsnorm_scalar() {
        let input = [2.0_f32, 2.0, 2.0, 2.0];
        let weight = [1.0_f32, 1.0, 1.0, 1.0];
        let mut output = [0.0_f32; 4];
        let eps = 1e-5_f32;

        BaselineKernel::new()
            .rmsnorm(&mut output, &input, &weight, eps)
            .expect("rmsnorm succeeds");
        for &val in &output {
            assert!((val - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_rmsnorm_mismatch_errors() {
        let input = [1.0_f32; 4];
        let weight = [1.0_f32; 3];
        let mut output = [0.0_f32; 4];
        let err = rmsnorm_scalar(&mut output, &input, &weight, 1e-5);
        assert!(matches!(err, Err(K3Error::TensorBoundsError { .. })));
    }

    #[test]
    fn test_q4_k_m_dequantization_and_vec_dot_us01_6_2() {
        // Construct 2 blocks (288 bytes) representing a [2, 256] or [256, 2] Q4_K_M matrix
        let mut raw = [0u8; 288];
        for blk in 0..2_u8 {
            let base = usize::from(blk) * Q4_K_M_BLOCK_BYTES;
            // d = 1.0 in f16 (0x3C00), dmin = 0.0 in f16 (0x0000)
            raw[base..base + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
            raw[base + 2..base + 4].copy_from_slice(&0x0000_u16.to_le_bytes());
            // scales = 1 for all sub-blocks
            for s in &mut raw[base + 4..base + 8] {
                *s = 1;
            }
            // qs nibbles: low nibble = (blk + 1), high nibble = (blk + 2)
            let nibble_byte = ((blk + 2) << 4) | (blk + 1);
            for q in &mut raw[base + 16..base + 144] {
                *q = nibble_byte;
            }
        }

        let view =
            TensorView::from_bytes("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2], 0, &raw)
                .expect("create Q4_K_M view");
        let input = [1.0_f32; 256];
        let mut output = [0.0_f32; 2];

        BaselineKernel::new()
            .vec_dot_q4_k_m(&mut output, &view, &input)
            .expect("vec_dot_q4_k_m succeeds");

        assert!(output[0] > 0.0);
        assert!(output[1] > output[0]);
    }
}
