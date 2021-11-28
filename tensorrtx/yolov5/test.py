# from yolov5_trt import get_img_path_batches
import os
import cv2

def test_yield(image_path_batch):
    for imgpath in image_path_batch:
        yield cv2.imread(imgpath)
def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret
for image_path_batch in get_img_path_batches(1, "test_img/"):
    print(test_yield(image_path_batch))
    # for i, img_path in enumerate(image_path_batch):
    #     print(i)
    #     print(img_path)
    break
        # parent, filename = os.path.split(img_path)
        # save_name = os.path.join('output', filename)
        # # Save image
        # cv2.imwrite(save_name, batch_image_raw[i])


