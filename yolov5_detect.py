from hackathon5_yolo_utils import *


def detect_dir(detect_dir,img_resault_dir,label_resault_dir,clss_list,conf_th,yolov5_wrapper,imwrite=True):
    dirs = os.listdir(detect_dir)
    # 按照batch制作输入图片的路径列表
    image_path_batches = get_img_path_batches(yolov5_wrapper.batch_size,detect_dir)
    for batch in image_path_batches:
        # 返回可迭代图片生成器
        raw_image_generator = get_raw_image(batch)
        # 画框后的图片array和检测时间
        batch_image_raw, use_time, result = yolov5_wrapper.infer(raw_image_generator, batch, clss_list)
        for i, img_path in enumerate(batch):
            parent, filename = os.path.split(img_path)
            save_name = os.path.join(img_resault_dir, filename)
            print(save_name)
            if imwrite == True:
                # Save image
                cv2.imwrite(save_name, batch_image_raw[i])
            clss_ = list(result[i][0])
            conf_ = list(result[i][1])
            boxes_ = list(result[i][2])
            with open(label_resault_dir + filename.split('.')[0] + ".txt", 'a') as txtf:
                for a in range(len(boxes_)):
                    class_name = clss_list[int(clss_[a])]
                    conf = conf_[a]
                    bndbox = boxes_[a]
                    xmin = bndbox[0]
                    ymin = bndbox[1]
                    xmax = bndbox[2]
                    ymax = bndbox[3]
                    # with open(label_resault_dir + filename.split('.')[0] + ".txt", 'a') as txtf:
                    txtf.write(str(class_name) + " ")
                    txtf.write(str(conf) + " ")
                    txtf.write(str(int(xmin)) + " " + str(int(ymin)) + " " + str(int(xmax)) + " " + str(int(ymax)))
                    if a != len(boxes_):
                        txtf.write('\n')
                    print(class_name, conf, xmin, ymin, xmax, ymax)
            txtf.close()
        print(str('input->{}, time->{:.2f}ms, saving into '.format(batch, use_time * 1000)) + str(label_resault_dir))

def detect_one(imgpath,filename,img_resault_dir,label_resault_dir,clss_list,conf_th,yolov5_wrapper,imwrite=True):
    batch = [str(os.path.join(imgpath, filename))]
    # 返回可迭代图片生成器
    raw_image_generator = get_raw_image(batch)
    # 画框后的图片array和检测时间
    batch_image_raw, use_time, result = yolov5_wrapper.infer(raw_image_generator, batch, clss_list)
    for i, img_path in enumerate(batch):
        parent, filename = os.path.split(img_path)
        save_name = os.path.join(img_resault_dir, filename)
        print(save_name)
        if imwrite == True:
            cv2.imwrite(save_name, batch_image_raw[i])
        clss_ = list(result[i][0])
        conf_ = list(result[i][1])
        boxes_ = list(result[i][2])
        for a in range(len(boxes_)):
            class_name = clss_list[int(clss_[a])]
            conf = conf_[a]
            bndbox = boxes_[a]
            xmin = bndbox[0]
            ymin = bndbox[1]
            xmax = bndbox[2]
            ymax = bndbox[3]
            with open(label_resault_dir + filename.split('.')[0] + ".txt", 'a') as txtf:
                txtf.write(str(class_name) + " ")
                txtf.write(str(conf) + " ")
                txtf.write(str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax))
                if a != len(boxes_):
                    txtf.write('\n')
                print(class_name, conf, xmin, ymin, xmax, ymax)
            txtf.close()
    print(str('input->{}, time->{:.2f}ms, saving into '.format(batch, use_time * 1000)) + str(label_resault_dir))

def clss_detect(detect_dir,img_resault_dir,label_resault_dir,clss_list,conf_th,yolov5_wrapper,detect_clss="cat",imwrite=True):
    dirs = os.listdir(detect_dir)
    # 按照batch制作输入图片的路径列表
    image_path_batches = get_img_path_batches(yolov5_wrapper.batch_size,detect_dir)
    for batch in image_path_batches:
        # 返回可迭代图片生成器
        raw_image_generator = get_raw_image(batch)
        # 画框后的图片array和检测时间
        batch_image_raw, use_time, result = yolov5_wrapper.infer(raw_image_generator, batch, clss_list)
        for i, img_path in enumerate(batch):
            parent, filename = os.path.split(img_path)
            save_name = os.path.join(img_resault_dir, filename)
            print(save_name)
            clss_ = list(result[i][0])
            conf_ = list(result[i][1])
            boxes_ = list(result[i][2])
            for a in range(len(boxes_)):
                class_name = clss_list[int(clss_[a])]
                conf = conf_[a]
                bndbox = boxes_[a]
                xmin = bndbox[0]
                ymin = bndbox[1]
                xmax = bndbox[2]
                ymax = bndbox[3]
                if class_name == detect_clss:
                    if imwrite == True:
                        # Save image
                        cv2.imwrite(save_name, batch_image_raw[i])
                    with open(label_resault_dir + filename.split('.')[0] + ".txt", 'a') as txtf:
                        txtf.write(str(class_name) + " ")
                        txtf.write(str(conf) + " ")
                        txtf.write(str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax))
                        if a != len(boxes_):
                            txtf.write('\n')
                        print(class_name, conf, xmin, ymin, xmax, ymax)
                    txtf.close()
        print(str('input->{}, time->{:.2f}ms, saving into '.format(batch, use_time * 1000)) + str(label_resault_dir))


if __name__ == '__main__':
    # 定义命令行输入
    parser = argparse.ArgumentParser()
    parser.add_argument('--plugin', default="tensorrtx/yolov5/build/libmyplugins.so",type=str,help='libmyplugins.so')
    parser.add_argument('--engine', default="tensorrtx/yolov5/build/yolov5s.engine",type=str,help='yolov5s.engine')
    parser.add_argument('--imgdir', default="test_img/",type=str,help='img dir')
    parser.add_argument('--imgoutdir', default="output/images/",type=str,help='output dir')
    parser.add_argument('--imgname', default="", type=str, help='detect one imgpath')
    parser.add_argument('--labeloutdir', default="output/labels/",type=str,help='output dir')
    parser.add_argument('--imwrite', default=True,type=bool,help='draw img, True/False')
    parser.add_argument('--conf', default=0.5,type=float,help='CONF_THRESH')
    parser.add_argument('--iou', default=0.4,type=float,help='IOU THRESHOLD')
    parser.add_argument('--detectmode', default="detect_dir",type=str,help='detect_dir/detect_one/clss_detect')
    parser.add_argument('--singl_clss_detect', default="cat",type=str,help='clssname')
    args = parser.parse_args()

    # 定义pulgin文件和engine文件
    PLUGIN_LIBRARY = args.plugin
    engine_file_path = args.engine
    # 定义标签类别
    clss_list = ["cat", "dog", "horse", "person"]
    # clss_list = args.clss_list
    # 输入的图片路径
    img_dir = args.imgdir

    # 加载plugin
    ctypes.CDLL(PLUGIN_LIBRARY)

    # 创建输出文件夹
    if os.path.exists(args.labeloutdir):
        shutil.rmtree(args.labeloutdir)
    label_path = str(args.labeloutdir)
    os.mkdir(label_path)

    if os.path.exists(args.imgoutdir):
        shutil.rmtree(args.imgoutdir)
    img_out_path = str(args.imgoutdir)
    os.mkdir(img_out_path)

    # 定义检测阈值
    CONF_THRESH = args.conf
    IOU_THRESHOLD = args.iou

    # 初始化yolo实例
    yolov5_wrapper = YoLov5TRT(engine_file_path,args)

    # 按照batch制作输入图片的路径列表
    # image_path_batches = get_img_path_batches(yolov5_wrapper.batch_size, img_dir)

    try:
        print('batch size is', yolov5_wrapper.batch_size)
        print("detect mode is", args.detectmode)
        if args.detectmode == "detect_dir":
            detect_dir(img_dir,img_out_path,label_path,clss_list,CONF_THRESH,yolov5_wrapper)
        elif args.detectmode == "detect_one":
            detect_one(img_dir,args.imgname,img_out_path,label_path,clss_list,CONF_THRESH,yolov5_wrapper)
        elif args.detectmode == "clss_detect":
            clss_detect(img_dir,img_out_path,label_path,clss_list,CONF_THRESH,yolov5_wrapper,args.singl_clss_detect)
    finally:
        # destroy the instance
        yolov5_wrapper.destroy()