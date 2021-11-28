##### 1. 修改`tensorrtx/yolov5.cpp`
```
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define CONF_THRESH 0.5     // 标签显示阈值
#define BATCH_SIZE 8    // batch大小
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // 最大图片大小
```

##### 2. 修改`tensorrtx/yololayer.h`
```
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;  // 最大输出box数量
    static constexpr int CLASS_NUM = 4;  // 类型数量
    static constexpr int INPUT_H = 640;  // 输入图片的高  (必须为32的倍数)
    static constexpr int INPUT_W = 640;  // 输入图片的宽  (必须为32的倍数)
```

##### 3. 转换`.pt`到`.wts`
```
// clone code according to above #Different versions of yolov5
// download https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
cp {tensorrtx}/yolov5/gen_wts.py {ultralytics}/yolov5
cd {ultralytics}/yolov5
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
// a file 'yolov5s.wts' will be generated.
```

##### 4. 构建`tensorrtx/yolov5`
```
cd {tensorrtx}/yolov5/
// update CLASS_NUM in yololayer.h if your model is trained on custom dataset
mkdir build
cd build
cp {ultralytics}/yolov5/yolov5s.wts {tensorrtx}/yolov5/build
cmake ..
make
sudo ./yolov5 -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file
sudo ./yolov5 -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.
// For example yolov5s
sudo ./yolov5 -s yolov5s.wts yolov5s.engine s
sudo ./yolov5 -d yolov5s.engine ../samples
// For example Custom model with depth_multiple=0.17, width_multiple=0.25 in yolov5.yaml
sudo ./yolov5 -s yolov5_custom.wts yolov5.engine c 0.17 0.25
sudo ./yolov5 -d yolov5.engine ../samples
```

##### 5. 运行检测并指定参数
```
$ python3 ./yolov5_detection/yolov5_detect_utils_Hackathon5.py --help
usage: yolov5_detect_utils_Hackathon5.py [-h] [--plugin PLUGIN]
                                         [--engine ENGINE] [--imgdir IMGDIR]
                                         [--outdir OUTDIR] [--imwrite IMWRITE]
                                         [--conf CONF] [--iou IOU]

optional arguments:
  -h, --help         show this help message and exit
  --plugin PLUGIN    libmyplugins.so
  --engine ENGINE    yolov5s.engine
  --imgdir IMGDIR    待检测图片地址
  --outdir OUTDIR    检测结果输出路径
  --imwrite IMWRITE  是否绘图, True/False
  --conf CONF        检测阈值
  --iou IOU          IOU_THRESHOLD
```
```
python3 ./yolov5_detection/yolov5_detect_utils_Hackathon5.py \
        --plugin ./yolov5_detection/tensorrtx/yolov5/build/libmyplugins.so  \
        --engine ./yolov5_detection/tensorrtx/yolov5/build/H4animal.engine  \
        --imgdir ./val/images/ \
        --outdir ./val/output_dir/ \
        --imwrite True \
        --conf 0.5 
```