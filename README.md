# QDGrasp CUDA — Kaggle

Repository độc lập chứa notebook GPU và raw CUDA evidence cho QDGrasp. Nó
không chứa source, dataset, checkpoint hoặc credential của dự án chính.

## Phase 0 smoke

Notebook `kaggle/qdgrasp_phase0_cuda.ipynb` fail-closed và chỉ pass khi:

- Python là 3.11;
- PyTorch là `2.11.0+cu128` và `torch.version.cuda == "12.8"`;
- có GPU NVIDIA T4 (`sm_75`) thật và tensor operation chạy trên device CUDA;
- AMP train-step, optimizer step và checkpoint resume chạy thành công;
- Lightning Fabric import và MuJoCo forward pass thành công.
- package `qdgrasp` được cài từ exact commit public và gọi API fail-closed.

Kết quả máy đọc được ghi thành `phase0_cuda_evidence.json` trong Kaggle output.
Kaggle/GitHub tokens không được ghi vào notebook, output hoặc Git history.

## Chạy

```bash
kaggle kernels push -p kaggle
kaggle kernels status niniflo/qdgrasp-phase0-cuda-smoke
kaggle kernels output niniflo/qdgrasp-phase0-cuda-smoke -p evidence/latest
```

Các notebook sau phải tạo evidence riêng và không sửa/xóa kết quả run cũ đã
được dùng trong báo cáo.
