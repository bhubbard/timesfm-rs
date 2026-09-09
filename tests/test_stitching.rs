use candle_core::{DType, Device, Tensor};
use timesfm::util::{apply_linear_detrending, get_output_patch_via_roll, stitch_patches};

#[test]
fn test_get_output_patch_via_roll() -> candle_core::Result<()> {
    let device = Device::Cpu;
    // (b=1, v=1, n=3, p=2)
    let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &device)?.reshape((1, 1, 3, 2))?;
    let rolls = 2;
    let (rolled, wrap_mask) = get_output_patch_via_roll(&x, rolls)?;

    // Shape: (b=1, v=1, n=3, rolls*p = 4)
    assert_eq!(rolled.dims(), &[1, 1, 3, 4]);
    assert_eq!(wrap_mask.dims(), &[1, 1, 3, 4]);

    // For patch 0, rolling shift 1 brings patch 1 [3, 4], shift 2 brings patch 2 [5, 6]
    let p0 = rolled.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(p0, vec![3.0, 4.0, 5.0, 6.0]);

    Ok(())
}

#[test]
fn test_stitch_patches_linear() -> candle_core::Result<()> {
    let device = Device::Cpu;
    // 2 patches, patch_len=4, total_len=6 -> overlap=2
    // patch 0: [1, 2, 3, 4, 10, 10]
    // patch 1: [20, 20, 7, 8, 9, 10]
    // stitched overlap should be 1.0 * 10 + 0.0 * 20 = 10, and 0.0 * 10 + 1.0 * 20 = 20
    let p0 = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 10.0, 10.0], &device)?;
    let p1 = Tensor::new(&[20.0f32, 20.0, 7.0, 8.0, 9.0, 10.0], &device)?;

    let patch_preds = Tensor::stack(&[&p0, &p1], 0)?
        .reshape((1, 1, 2, 6, 1))?;

    let stitched = stitch_patches(&patch_preds, 4)?;
    // output length: 2 * 4 + 2 = 10
    assert_eq!(stitched.dims(), &[1, 1, 10, 1]);

    let vals = stitched.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[3], 4.0);
    // overlap:
    assert_eq!(vals[4], 10.0);
    assert_eq!(vals[5], 20.0);
    // second chunk:
    assert_eq!(vals[6], 7.0);
    assert_eq!(vals[9], 10.0);

    Ok(())
}

#[test]
fn test_linear_detrending_slope() -> candle_core::Result<()> {
    let device = Device::Cpu;
    // Perfectly linear sequence: y = 2.0 * t + 5.0
    // 16 context points
    let n = 16;
    let mut y_vec = Vec::with_capacity(n);
    for i in 0..n {
        let y = 2.0 * (i as f32) + 5.0;
        y_vec.push(y);
    }
    let ctx_vals = Tensor::from_vec(y_vec, (1, 1, n), &device)?;
    let ctx_masks = Tensor::zeros((1, 1, n), DType::F32, &device)?;

    let (detrended, _m, _c, cond) = apply_linear_detrending(&ctx_vals, &ctx_masks, 0.5)?;

    let should_detrend = cond.flatten_all()?.to_vec1::<u8>()?[0] > 0;
    assert!(should_detrend, "Perfect line must be detrended");

    // Detrended values should have zero or near-zero variance
    let det_var = detrended.sqr()?.mean_all()?.to_scalar::<f32>()?;
    assert!(det_var < 1e-4, "Detrended variance should be close to 0, got {}", det_var);

    Ok(())
}
