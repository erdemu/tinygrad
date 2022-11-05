# pip install pyobjc-framework-MetalPerformanceShadersGraph
import MetalPerformanceShadersGraph as mpsgraph
import Metal
import numpy as np
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, GenericExecAST

mps_data_type_float32 = 268435488
device = Metal.MTLCreateSystemDefaultDevice()

mpsgraphdevice = mpsgraph.MPSGraphDevice.alloc().initWithDeviceType_metalDevice_(
    0, device
)

graph = mpsgraph.MPSGraph.alloc().init()


class MPSBuffer(GenericExecAST):
    def __init__(self, data):
        self.shape = list(data.shape)
        self.mps_tensor_data = mpsgraph.MPSGraphTensorData.alloc().initWithDevice_data_shape_dataType_(
            mpsgraphdevice, data.data, list(data.shape), mps_data_type_float32
        )

    def to_cpu(self):
        buf = np.zeros(shape=self.shape, dtype=np.float32)
        self.mps_tensor_data.readBytes_strideBytes_(buf.data, None)
        return buf

    @staticmethod
    def from_cpu(data):
        return MPSBuffer(data=data)

    def relu(self):
        input_tensor = graph.placeholderWithShape_dataType_name_(
            None, mps_data_type_float32, None
        )
        relu_ = graph.reluWithTensor_name_(input_tensor, None)
        run_res = graph.runWithFeeds_targetTensors_targetOperations_(
            {input_tensor: self.mps_tensor_data}, [relu_], None
        )
        return MPSBuffer(run_res[relu_])


if __name__ == "__main__":
    data = np.random.rand(16, 16)
    data = data.astype(dtype=np.float32)

    input_tensor = graph.placeholderWithShape_dataType_name_(
        None, mps_data_type_float32, None
    )
    exp_ = graph.exponentWithTensor_name_(input_tensor, None)

    input_data = (
        mpsgraph.MPSGraphTensorData.alloc().initWithDevice_data_shape_dataType_(
            mpsgraphdevice, data.data, list(data.shape), mps_data_type_float32
        )
    )
    results = graph.runWithFeeds_targetTensors_targetOperations_(
        {input_tensor: input_data}, [exp_], None
    )
    output_data = results[exp_].mpsndarray()
    output_host_data = np.zeros_like(data)
    output_data.readBytes_strideBytes_(output_host_data.data, None)

    print("CpuCalc")
    print(np.exp(data))
    print("MPSCalc")
    print(output_host_data)
    print('OK' if np.allclose(np.exp(data), output_host_data) else 'Not OK')
