import os
import sys
import argparse

import numpy as np
import tensorrt as trt
from common import *
import torch
from fastflownet import centralize

import pycuda.driver as cuda
import pycuda.autoinit

import ctypes
ctypes.CDLL(open(os.path.join(os.path.dirname(__file__),'tensorrt_plugin_path')).read())

if __name__ == '__main__':
    logger = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(logger, '')
    script_root = os.path.dirname(__file__)
    with open('./engine_fp16', "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        inputs,outputs,bindings,stream = allocate_buffers(engine,True,2)

        import cv2
        from flow_vis import flow_to_color
        img_paths = [os.path.join(script_root,'../data/img_050.jpg'), \
                     os.path.join(script_root,'../data/img_051.jpg')]

        div_flow = 20.0
        div_size = 64
        fig1,fig2 = cv2.imread(img_paths[0]),cv2.imread(img_paths[1])
        fig1,fig2 = cv2.resize(fig1,(512,512)),cv2.resize(fig2,(512,512))

        img1 = torch.from_numpy(fig1).float().permute(2,0,1)[None] / 255.
        img2 = torch.from_numpy(fig2).float().permute(2,0,1)[None] / 255.
        img1,img2,_ = centralize(img1,img2)
        input_t = torch.cat([img1,img2],1)
        for binding in engine:
            print('-------------------')
            print(engine.get_binding_shape(binding))
            print(engine.get_binding_name(engine.get_binding_index(binding)))
        with engine.create_execution_context() as context:
            input_t = input_t.float().numpy()
            inputs[0].host = input_t
            trt_outputs = do_inference_v2(context,bindings=bindings,inputs=inputs,outputs=outputs,stream=stream)
            output = trt_outputs[0].reshape(engine.get_binding_shape(1)).squeeze()

            flow = div_flow * output

            flow = np.transpose(flow,[1,2,0])
            flow_color = flow_to_color(flow,convert_to_bgr=True)

            cv2.namedWindow('tensorrt flow',cv2.WINDOW_NORMAL)
            cv2.imshow('tensorrt flow',flow_color)
            cv2.waitKey(0)

            for i in range(10):
                trt_outputs = do_inference_v2(context,bindings=bindings,inputs=inputs,outputs=outputs,stream=stream)

            import time
            tic = time.time()
            for i in range(1000):
                trt_outputs = do_inference_v2(context,bindings=bindings,inputs=inputs,outputs=outputs,stream=stream)
            toc = time.time()

            print('fps: ',1/((toc-tic)/1000))