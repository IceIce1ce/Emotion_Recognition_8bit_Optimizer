Đồ án sẽ bao gồm 4 folder chính:
- FER2013_VGG19: chứa file đã lưu tại epoch tốt nhất của model
- data: chứa file fer2013.csv dùng cho việc train mà được lấy từ kaggle
- face_detection: chứa một file dat pretrain nhận diện mặt người được lấy từ thư viện dlib
- test_images: chứa các ảnh mặt người dùng cho dự đoán cảm xúc
- Về cách chạy, đầu tiên chạy file build_dataset chúng ta sẽ có 1 file data.h5 trong folder data, đây là file chia train/valid/test từ file csv và được dùng cho việc huấn luyện mô hình. Tiếp theo, chạy file main.py (python main.py --lr 0.01) để train mô hình và tại mỗi epoch tốt nhất kết quả sẽ được lưu vào folder FER2013_VGG19. Cuối cùng chạy file make_pred để nhận diện cảm xúc trên ảnh mặt người, đầu vào có thể là ảnh một người hoặc nhiều người (python make_pred.py --input_img 'test_images/5.jpg' --output_img 'test_images/results/5.jpg')
- Trước khi chạy make_pred.py để dự đoán cảm xúc của một ảnh, cần tạo thêm 2 folder là cropped_images và results để lưu các nén ảnh cho việc dự đoán chính xác hơn và lưu lại kết quả đã dự đoán được
- Link dataset: https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data
- Khi tải dữ liệu từ link trên, ta sẽ có 1 file tên fer2013.tar.gz, giải nén file này sẽ được 1 file tên fer2013.csv và file này sẽ được đặt vào folder data để tạo ra file data.h5 cho việc huấn luyện mô hình
