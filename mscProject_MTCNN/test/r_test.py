import torch, time
import mscProject_MTCNN.net.checknet as nets
import mscProject_MTCNN.tool.utils as utils
from PIL import Image

if __name__ == '__main__':
    #net
    pnet = nets.P_Net(istraining=False)
    rnet = nets.R_Net(istraining=False)
    if torch.cuda.is_available():
        pnet = pnet.cuda()
        rnet = rnet.cuda()
    pnet.eval()
    rnet.eval()
    pnet.load_state_dict(torch.load(r'D:\AI_Project\MTCNN\test_0808\params\p_params.pkl')) # 导入训练参数
    rnet.load_state_dict(torch.load(r'D:\AI_Project\MTCNN\test_0808\params\r_params.pkl'))  # 导入训练参数

    #输入图片
    img = Image.open("./333333.jpg")

    # P网络
    start_tim = time.time()
    pboxs = utils.PnetDetect(pnet, img, imgshow=False)
    pnet_tim = time.time()
    # R网络
    rboxs = utils.RnetDetect(rnet, img, pboxs,imgshow=True, show_conf=False)
    rnet_tim = time.time()

    print("pnet_time:", pnet_tim-start_tim, "rnet_time:", rnet_tim-pnet_tim)


