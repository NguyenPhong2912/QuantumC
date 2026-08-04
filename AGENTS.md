# Quy ước làm việc trên nhánh AnKy06

## Nhánh Git

- Mọi thay đổi, commit và push của công việc này phải nằm trên nhánh `AnKy06`.
- Trước khi sửa, commit hoặc push, phải kiểm tra `git branch --show-current` và dừng lại nếu kết quả không phải `AnKy06`.
- Không được push trực tiếp lên `main`.
- Khi cần đẩy code, dùng đích danh nhánh để tránh nhầm: `git push origin HEAD:AnKy06`.
- Nếu cần đưa thay đổi vào `main`, chỉ thực hiện qua pull request và sau khi chủ repo duyệt.

## Môi trường FITLAB-02

- Mọi lệnh chạy, audit, test và huấn luyện trên FIT-LAB mặc định dùng máy `FITLAB-02`.
- Đường dẫn dự án trên server là `/export/users/1165521/iDragonCloud/QuantumC`.
- Môi trường Python trên server là `.venv-fitlab02`.

## Requirements

- Nguồn requirements chuẩn luôn là file của nhánh `main`: `main:environment/requirements-fitlab02-stage1.txt`.
- Không tạo bộ requirements riêng cho `AnKy06` và không tự ý thay đổi phiên bản dependency trên nhánh này.
- Khi cài đặt, xuất đúng bản requirements từ `main`, sau đó mới cài:

  ```bash
  git show main:environment/requirements-fitlab02-stage1.txt > /tmp/qvisionframe-main-requirements.txt
  python -m pip install -r /tmp/qvisionframe-main-requirements.txt
  ```

- Nếu requirements trên `main` thay đổi, đồng bộ thay đổi đó vào `AnKy06` trước khi chạy công việc tiếp theo.

## Bảo mật

- Không commit hoặc push file chứa tài khoản, mật khẩu, token, SSH key hay hướng dẫn truy cập nội bộ.
- Thư mục `cac_buoc_vao_fitlab/` chỉ được lưu cục bộ và không được đưa lên remote.
