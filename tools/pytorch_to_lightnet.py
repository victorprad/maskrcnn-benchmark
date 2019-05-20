import struct
import numpy as np
import copy

class PytorchToLightNet:
    @staticmethod
    def writeImageInput(name, size, lntxt):
        lntxt.write("IMAGEINPUT {} size: {} {} {} {}\n".format(
            name, size[0], size[1], size[2], size[3]))
        lntxt.write("\tscope INPUT\n")
        lntxt.write("\tinput from 0\n")

        return name

    @staticmethod
    def writeConv(conv, scope, name, parent_name, lntxt, lnweights):
        weights = conv.weight.cpu().data.numpy().flatten()
        lnweights.write(struct.pack('f' * len(weights), *weights))
        hasBias = 0

        if conv.bias is not None:
            bias = conv.bias.cpu().data.numpy().flatten()
            lnweights.write(struct.pack('f' * len(bias), *bias))
            hasBias = 1

        type = "CONV"

        if conv.groups == conv.out_channels and conv.groups != 1:
            type = "CONVDEPTHWISE"

        lntxt.write("{} {} size: {} {} {} {} hasBias: {} stride: {} {} pad: {} {} {} {}\n".format(
            type, name,
            conv.kernel_size[0], conv.kernel_size[1], conv.in_channels, conv.out_channels,
            hasBias,
            conv.stride[0], conv.stride[1],
            conv.padding[0], conv.padding[0], conv.padding[1], conv.padding[1]))

        lntxt.write("\tscope {}\n".format(scope))

        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeConvRelu(conv, scope, name, parent_name, lntxt, lnweights):
        weights = conv.weight.cpu().data.numpy().flatten()
        lnweights.write(struct.pack('f' * len(weights), *weights))
        hasBias = 0

        if conv.bias is not None:
            bias = conv.bias.cpu().data.numpy().flatten()
            lnweights.write(struct.pack('f' * len(bias), *bias))
            hasBias = 1

        type = "CONVRELU"

        if conv.groups == conv.out_channels and conv.groups != 1:
            type = "CONVRELUDEPTHWISE"

        lntxt.write("{} {} size: {} {} {} {} hasBias: {} stride: {} {} pad: {} {} {} {}\n".format(
            type, name,
            conv.kernel_size[0], conv.kernel_size[1], conv.in_channels, conv.out_channels,
            hasBias,
            conv.stride[0], conv.stride[1],
            conv.padding[0], conv.padding[0], conv.padding[1], conv.padding[1]))

        lntxt.write("\tscope {}\n".format(scope))

        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeConvTransposed(conv, scope, name, parent_name, lntxt, lnweights):
        weights = conv.weight.cpu().data.numpy().flatten()
        lnweights.write(struct.pack('f' * len(weights), *weights))
        hasBias = 0

        if conv.bias is not None:
            bias = conv.bias.cpu().data.numpy().flatten()
            lnweights.write(struct.pack('f' * len(bias), *bias))
            hasBias = 1

        type = "CONVTRANS"

        if conv.groups == conv.out_channels:
            type = "CONVTRANSDEPTHWISE"

        lntxt.write("{} {} size: {} {} {} {} hasBias: {} upsample: {} {} crop: {} {} {} {}\n".format(
            type, name,
            conv.kernel_size[0], conv.kernel_size[1], conv.in_channels, conv.out_channels,
            hasBias,
            conv.stride[0], conv.stride[1],
            conv.padding[0], conv.padding[0], conv.padding[1], conv.padding[1]))

        lntxt.write("\tscope {}\n".format(scope))

        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeConvTransposed_forMetal(conv, scope, name, parent_name, lntxt, lnweights):
        weights = copy.deepcopy(conv.weight.cpu().data.numpy())
        weights = weights.transpose([1, 0, 2, 3])

        for i in range(0, weights.shape[0]):
            for j in range(0, weights.shape[1]):
                kernel = weights[i, j, :, :]
                kernel = np.flipud(kernel)
                kernel = np.fliplr(kernel)
                weights[i, j, :, :] = kernel

        weights = weights.flatten()

        lnweights.write(struct.pack('f' * len(weights), *weights))
        hasBias = 0

        if conv.bias is not None:
            bias = conv.bias.cpu().data.numpy().flatten()
            lnweights.write(struct.pack('f' * len(bias), *bias))
            hasBias = 1

        type = "CONVTRANS"

        if conv.groups == conv.out_channels:
            type = "CONVTRANSDEPTHWISE"

        lntxt.write("{} {} size: {} {} {} {} hasBias: {} upsample: {} {} crop: {} {} {} {}\n".format(
            type, name,
            conv.kernel_size[0], conv.kernel_size[1], conv.in_channels, conv.out_channels,
            hasBias,
            conv.stride[0], conv.stride[1],
            conv.padding[0], conv.padding[0], conv.padding[1], conv.padding[1]))

        lntxt.write("\tscope {}\n".format(scope))

        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeConvRelu6(conv, scope, name, parent_name, lntxt, lnweights):
        weights = conv.weight.cpu().data.numpy().flatten()
        lnweights.write(struct.pack('f' * len(weights), *weights))
        hasBias = 0

        if conv.bias is not None:
            bias = conv.bias.cpu().data.numpy().flatten()
            lnweights.write(struct.pack('f' * len(bias), *bias))
            hasBias = 1

        type = "CONVRELU6"

        if conv.groups == conv.out_channels:
            type = "CONVRELU6DEPTHWISE"

        lntxt.write("{} {} size: {} {} {} {} hasBias: {} stride: {} {} pad: {} {} {} {}\n".format(
            type, name,
            conv.kernel_size[0], conv.kernel_size[1], conv.in_channels, conv.out_channels,
            hasBias,
            conv.stride[0], conv.stride[1],
            conv.padding[0], conv.padding[0], conv.padding[1], conv.padding[1]))

        lntxt.write("\tscope {}\n".format(scope))

        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeBN(bn, scope, name, parent_name, lntxt, lnweights):
        weights = bn.weight.cpu().data.numpy()
        lnweights.write(struct.pack('f' * len(weights), *weights))

        bias = bn.bias.cpu().data.numpy()
        lnweights.write(struct.pack('f' * len(bias), *bias))

        running_mean = bn.running_mean.cpu().data.numpy()
        lnweights.write(struct.pack('f' * len(running_mean), *running_mean))

        running_var = np.sqrt(bn.running_var.cpu().data.numpy())
        lnweights.write(struct.pack('f' * len(running_var), *running_var))

        lntxt.write("BATCHNORM {}\n".format(name))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeIN(bn, scope, name, parent_name, lntxt, lnweights):
        weights = bn.weight.cpu().data.numpy()
        lnweights.write(struct.pack('f' * len(weights), *weights))

        bias = bn.bias.cpu().data.numpy()
        lnweights.write(struct.pack('f' * len(bias), *bias))

        lntxt.write("INSTANCENORM {}\n".format(name))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeReLU6(relu, scope, name, parent_name, lntxt, lnweights):
        lntxt.write("RELU6 {}\n".format(name))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeReLU(relu, scope, name, parent_name, lntxt, lnweights):
        lntxt.write("RELU {}\n".format(name))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeSigmoid(sigmoid, scope, name, parent_name, lntxt, lnweights):
        lntxt.write("SIGMOID {}\n".format(name))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeFC_afterReshape(fc, parent_dims, scope, name, parent_name, lntxt, lnweights):
        #the horrible flip is needed because the MPS reshape doesn't do the same thing as .view(-1)
        weights = fc.weight.cpu().data.numpy().reshape(parent_dims).transpose([0, 2, 3, 1]).flatten()

        lnweights.write(struct.pack('f' * len(weights), *weights))
        hasBias = 0

        if fc.bias is not None:
            bias = fc.bias.cpu().data.numpy().flatten()
            lnweights.write(struct.pack('f' * len(bias), *bias))
            hasBias = 1

        lntxt.write("INNERPRODUCT {} size: {} {} {} {} hasBias: {} stride: {} {} pad: {} {} {} {} newShape: {} {} {} {} \n".format(
            name, 1, 1, fc.in_features, fc.out_features, hasBias, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeFC(fc, scope, name, parent_name, lntxt, lnweights):
        #the horrible flip is needed because the MPS reshape doesn't do the same thing as .view(-1)
        weights = fc.weight.cpu().data.numpy().flatten()

        lnweights.write(struct.pack('f' * len(weights), *weights))
        hasBias = 0

        if fc.bias is not None:
            bias = fc.bias.cpu().data.numpy().flatten()
            lnweights.write(struct.pack('f' * len(bias), *bias))
            hasBias = 1

        lntxt.write("INNERPRODUCT {} size: {} {} {} {} hasBias: {} stride: {} {} pad: {} {} {} {} newShape: {} {} {} {} \n".format(
            name, 1, 1, fc.in_features, fc.out_features, hasBias, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeRoiAlign(roialign, scope, name, lntxt):
        lntxt.write("ROIALIGN {} resolution: {} {} channels: {} sampling_ratio: {}\n".format(
            name, roialign["resolution"], roialign["resolution"], roialign["channels"], roialign["sampling_ratio"]))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 0\n")

        return name

    @staticmethod
    def writeSum(scope, name, parent_a, parent_b, lntxt):
        lntxt.write("SUM {}\n".format(name))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 2: {} {}\n".format(parent_a, parent_b))

        return name

    @staticmethod
    def writeReshape(size_out, scope, name, parent_name, lntxt):
        lntxt.write("RESHAPE {} out: {} {} {}\n".format(
            name, size_out[0], size_out[1], size_out[2]))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeCopyOutput(name, parent_name, lntxt):
        lntxt.write("COPYOUTPUT {}\n".format(name))
        lntxt.write("\tscope OUTPUT\n")
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeSoftMax(scope, name, parent_name, lntxt):
        lntxt.write("SOFTMAX {}\n".format(name))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeMaxPool(maxpool, scope, name, parent_name, lntxt):
        remove_extra = 1
        if maxpool.ceil_mode:
            remove_extra = 0

        lntxt.write("POOLING {} size: {} {} method: max stride: {} {} pad: {} {} {} {} remove_extra: {}\n".format(
            name, maxpool.kernel_size, maxpool.kernel_size, maxpool.stride, maxpool.stride,
            maxpool.padding, maxpool.padding, maxpool.padding, maxpool.padding, remove_extra))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeMaxPoolFromParams(scope, name, kernel_size, stride, padding, ceil_mode, parent_name, lntxt):
        remove_extra = 1
        if ceil_mode:
            remove_extra = 0

        lntxt.write("POOLING {} size: {} {} method: max stride: {} {} pad: {} {} {} {} remove_extra: {}\n".format(
            name, kernel_size, kernel_size, stride, stride,
            padding, padding, padding, padding, remove_extra))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeUpSampleNearest(upsample, scope, name, ratio, rect, parent_name, lntxt):
        lntxt.write("UPSAMPLE {} upsample: {} {} rect: {} {} method: nearest\n".format(
            name, ratio[0], ratio[1], rect[0], rect[1]))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name

    @staticmethod
    def writeNorm(scope, name, parent_name, lntxt):
        lntxt.write("NORM {}\n".format(name))
        lntxt.write("\tscope {}\n".format(scope))
        lntxt.write("\tinput from 1: {}\n".format(parent_name))

        return name
