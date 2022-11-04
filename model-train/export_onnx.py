import argparse
import torch
from datasets.dataset_dota import DOTA
from datasets.dataset_hrsc import HRSC
from models import ctrbox_net
import decoder
import os
import test
import time
import numpy as np
import func_utils
import cv2
from onnxruntime.datasets import get_example
import onnxruntime
import time

def torch2onnx(args, model):
    image_set_index_file = os.path.join(args.data_dir, args.phase + '.txt')
    with open(image_set_index_file, 'r') as f:
        lines = f.readlines()
    image_lists = [line.strip() for line in lines]
    image_path = os.path.join(args.data_dir, 'images')
    imgFile = os.path.join(image_path, image_lists[0] + '.jpg')
    image = cv2.imread(imgFile)
    input_w = args.input_w
    input_h = args.input_h
    image = cv2.resize(image, (input_w, input_h))
    out_image = image.astype(np.float32) / 255.
    out_image = out_image - 0.5
    out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
    dummy_input = torch.from_numpy(out_image).to(ctrbox_obj.device)
    input_names = ['input']  # ����������name
    output_names = ['output']  # ����������name
    print("====", dummy_input.shape)
    torch_out = torch.onnx._export(model, dummy_input, args.onnx_model_path,
                                   verbose=False, input_names=input_names, output_names=output_names, opset_version=11)
    # test onnx model
    session = onnxruntime.InferenceSession(args.onnx_model_path)
    for i in range(100):
        t1=time.time()

        # get the name of the first input of the model
        input_name = session.get_inputs()[0].name
        # print('onnx Input Name:', input_name)
        result = session.run([], {input_name: dummy_input.data.cpu().numpy()})

        t2=time.time()-t1
        print(t2)
    print("the result is {}".format(result))
    print("result[0].shape", result[0].shape)


def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors Implementation')
    parser.add_argument('--num_epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Initial learning rate')
    parser.add_argument('--input_h', type=int, default=512, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=512, help='Resized image width')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.18,
                        help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='model_best.pth', help='Weights resumed in testing and evaluation')
    parser.add_argument('--dataset', type=str, default='dota', help='Name of dataset')
    parser.add_argument('--data_dir', type=str, default='/home/addtion_storage/huangtao/tomato_detect_rotate/split/', help='Data directory')
    parser.add_argument('--phase', type=str, default='test', help='Phase choice= {train, test, eval}')
    parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')
    parser.add_argument('--onnx_model_path', type=str,
                        default='weights_dota/model_best.onnx',
                        help='onnx model path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    dataset = {'dota': DOTA, 'hrsc': HRSC}
    num_classes = {'dota': 3, 'hrsc': 1}
    heads = {'hm': num_classes[args.dataset],
             'wh': 10,
             'reg': 2,
             'cls_theta': 1
             }
    down_ratio = 2#2for resnet 18(but when test, turn it to 4)  4for 101
    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=256,export=True)

    decoder = decoder.DecDecoder(K=args.K,
                                 conf_thresh=args.conf_thresh,
                                 num_classes=num_classes[args.dataset])

    ctrbox_obj = test.TestModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
    save_path = 'weights_' + args.dataset
    ctrbox_obj.model = ctrbox_obj.load_model(ctrbox_obj.model, os.path.join(save_path, args.resume))
    ctrbox_obj.model = ctrbox_obj.model.to(ctrbox_obj.device)
    ctrbox_obj.model.eval()
    torch2onnx(args, ctrbox_obj.model)
