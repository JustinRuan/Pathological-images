import caffe
import utils

# 设定文件保存路径

solver_proto = utils.MODEL_PATH + 'alexnet/solver.prototxt'
#pretrained_model = 'H:/PYPROJECT/WorkSpace/Pathological-images/DetectCancer/models/alexnet/bvlc_reference_caffenet.caffemodel'

# 开始训练
def training(solver_proto):
    caffe.set_device(0)
    caffe.set_mode_gpu()

    solver = caffe.SGDSolver(solver_proto)
    # 利用snapshot从断点恢复训练
    # solver.net.copy_from(pretrained_model)
    solver.solve()

if __name__ == '__main__':
    training(solver_proto)
