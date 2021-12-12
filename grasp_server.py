import sys
import os
import argparse
import Pyro4
import random
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from shutil import rmtree
from pathlib import Path


from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp

#MODEL_PATH = GMDATA_PATH.joinpath('')
#TEST_PATH = GMDATA_PATH.joinpath('datasets/test/test_poses')
#TEST_OUTPUT = GMDATA_PATH.joinpath('ggtest')

MODEL_PATH = Path.home().joinpath('Project/GR-ConvNet/shahao_models')


class Grasp2D(object):
    """
    2D夹爪类型，夹爪投影到深度图像上的坐标.
    这里使用的都是图像坐标和numpy数组的轴顺序相反
    """

    def __init__(self, center, angle, depth, width=0.0, z_depth=0.0, quality=None, coll=None, gh=None):
        """ 一个带斜向因子z的2d抓取, 这里需要假定p1的深度比较大
        center : 夹爪中心坐标，像素坐标表示
        angle : 抓取方向和相机x坐标的夹角, 由深度较小的p0指向深度较大的p1, (-pi, pi)
        depth : 夹爪中心点的深度
        width : 夹爪的宽度像素坐标
        z_depth: 抓取端点到抓取中心点z轴的距离,单位为m而非像素
        quality: 抓取质量
        coll: 抓取是否碰撞
        """
        self.center = center
        self.angle = angle
        self.depth = depth
        self.width_px = width
        self.z_depth = z_depth
        self.quality = quality
        self.coll = coll
        self.gh = gh

    @property
    def norm_angle(self):
        """ 归一化到-pi/2到pi/2的角度 """
        a = self.angle
        while a >= np.pi/2:
            a = a-np.pi
        while a < -np.pi/2:
            a = a+np.pi
        return a

    @property
    def axis(self):
        """ Returns the grasp axis. """
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def endpoints(self):
        """ Returns the grasp endpoints """
        p0 = self.center - (float(self.width_px) / 2) * self.axis
        p1 = self.center + (float(self.width_px) / 2) * self.axis
        p0 = p0.astype(np.int)
        p1 = p1.astype(np.int)
        return p0, p1

    @classmethod
    def from_jaq(cls, jaq_string):
        jaq_string = jaq_string.strip()
        x, y, theta, w, h = [float(v) for v in jaq_string[:-1].split(';')]
        return cls(np.array([x, y]), theta/180.0*np.pi, 0, w, gh=h)


def plot_output(depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img,
                       width_img=grasp_width_img, no_grasps=1)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(depth_img, cmap='gray')
    for g in gs:
        g.plot(ax)
    ax.set_title('Depth')
    ax.axis('off')

    return gs


def get_model(model_path):
    model_path = Path(model_path).resolve()
    max_fn = 0
    max_f = None
    for f in model_path.iterdir():
        fs = f.name.split('_')
        if len(fs) == 4:
            fn = int(fs[1])
            if fn > max_fn:
                max_fn = fn
                max_f = f#这里是想找到最后一次epoch训练的参数结果，也就是使用最新参数
    return max_f





class GraspGenerator:
    def __init__(self, saved_model_path, visualize=False):
        self.saved_model_path = saved_model_path
        self.model = None
        self.device = None
        self.load_model()
        if visualize:
            self.fig = plt.figure(figsize=(10, 10))
        else:
            self.fig = None

    def load_model(self):
        print('Loading model... ')
        self.model = torch.load(self.saved_model_path)
        # Get the compute device
        self.device = get_device(force_cpu=False)

    def generate(self,depth):
        self.cam_data = CameraData(depth.shape[1],depth.shape[0],output_size=300,include_depth=True, include_rgb=False)
        print("1")
        x, depth_img, rgb_img = self.cam_data.get_data(depth=depth)
        # plt.clf()
        # plt.imshow(depth_img)
        # plt.colorbar()
        # plt.savefig("depth_img.png")
        print("2",np.mean(depth_img),np.std(depth_img))
        # Predict the grasp pose using the saved model
        with torch.no_grad():
            xc = x.to(self.device)
            print("3")
            pred = self.model.predict(xc)
        print("4")
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        print("5")
        grasps = detect_grasps(q_img, ang_img, width_img)
        return grasps,q_img, ang_img, width_img


@Pyro4.expose
class Planer(object):
    def __init__(self, model_path):
        self.grcnn = GraspGenerator(model_path)
        # self.model_name=model_path.split("/")[-1]#从路径中剪切出模型名并发送给客户端

    def plan(self, image, width):
        random.seed(0)
        np.random.seed(0)
        im = image.copy()
        try_num = 5
        qs = []
        gs = []
        for _ in range(try_num):
            try:
                ggs,points_out,_,_ = self.grcnn.generate(im)
                if len(ggs) == 0:
                    # plt.clf()
                    # plt.imshow(points_out)
                    # plt.colorbar()
                    # plt.savefig("predict_result.png")
                    # print("detect_grasps——error")
                    continue
                print("@@@中心,角度,宽度",ggs[0].center ,ggs[0].angle,ggs[0].length)
                g = Grasp2D.from_jaq(ggs[0].to_jacquard(scale=1))
                # g.width_px = width
                
                q = points_out[int(g.center[1]), int(g.center[0])]
            except Exception as e:
                print('--------------------出错了----------------------')
                print(e)
            else:
                qs.append(q)
                gs.append(g)
                # if q > 0.9:
                #     break

        # plt.clf()
        # plt.imshow(points_out)
        # plt.colorbar()
        # plt.savefig("predict_result.png")
        
        if len(gs) == 0:
            return [None, None, None,None,None,points_out]
            # return None
        g = gs[np.argmax(qs)]#取得是抓取质量最高的抓取
        q = qs[np.argmax(qs)]
        print("width",width)
        print("ggs[0]",ggs[0].width)
        print("width_px",g.width_px)
        
        p0, p1 = g.endpoints
        print("real_width",np.linalg.norm(p1-p0))
        print('-------------------------')
        print([p0, p1, g.depth, g.depth, q])

        return [p0, p1, g.depth, g.depth, q,points_out]#[0, 0, 0,0, 0,points_out]#


def main(args):
    model_path = MODEL_PATH.joinpath(args.model_name)
    model_path = get_model(model_path)
    pp = Planer(model_path)
    Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
    Pyro4.Daemon.serveSimple({pp: 'grasp'}, ns=False, host='', port=6665)








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dataset to npz')
    parser.add_argument('-m', '--model-name', metavar='narrow2_gmd', type=str, default='narrow2_gmd',
                        help='使用的模型的名字')#sparse_gmd#shahao_jacquard
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()
    if args.test:
        test()
    else:
        main(args)
