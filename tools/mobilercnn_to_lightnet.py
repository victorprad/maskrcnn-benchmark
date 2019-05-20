from tools.pytorch_to_lightnet import PytorchToLightNet as C


def writeConvDW(convdw, scope, name, parent_name, lntxt, lnweights, useConvRelu):
    if useConvRelu:
        C.writeConvRelu6(convdw._modules['0'], scope, name + "_conv0", parent_name, lntxt, lnweights)
        return C.writeConvRelu6(convdw._modules['2'], scope, name + "_conv1", "previous", lntxt, lnweights)
    else:
        C.writeConv(convdw._modules['0'], scope, name + "_conv0", parent_name, lntxt, lnweights)
        C.writeReLU6(convdw._modules['1'], scope, name + "_relu0", "previous", lntxt, lnweights)

        C.writeConv(convdw._modules['2'], scope, name + "_conv1", "previous", lntxt, lnweights)
        return C.writeReLU6(convdw._modules['3'], scope, name + "_relu1", "previous", lntxt, lnweights)


def writeConvFCN(convfcn, scope, name, parent_name, lntxt, lnweights, useConvRelu):
    writeConvDW(convfcn._modules['0'].conv, scope, name + "_dw0", parent_name, lntxt, lnweights, useConvRelu)
    writeConvDW(convfcn._modules['1'].conv, scope, name + "_dw1", parent_name, lntxt, lnweights, useConvRelu)
    writeConvDW(convfcn._modules['2'].conv, scope, name + "_dw2", parent_name, lntxt, lnweights, useConvRelu)
    writeConvDW(convfcn._modules['3'].conv, scope, name + "_dw3", parent_name, lntxt, lnweights, useConvRelu)

    return name + "_fcn3"


def writeConvBN6(convbn, scope, name, parent_name, lntxt, lnweights, useConvRelu):
    if useConvRelu:
        return C.writeConvRelu6(convbn._modules['0'], scope, name + "_conv0", parent_name, lntxt, lnweights)
    else:
        C.writeConv(convbn._modules['0'], scope, name + "_conv0", parent_name, lntxt, lnweights)
        return C.writeReLU6(convbn._modules['1'], scope, name + "_relu0", "previous", lntxt, lnweights)


def writeConvBN(convbn, scope, name, parent_name, lntxt, lnweights, useConvRelu):
    if useConvRelu:
        return C.writeConvRelu(convbn._modules['0'], scope, name + "_conv0", parent_name, lntxt, lnweights)
    else:
        C.writeConv(convbn._modules['0'], scope, name + "_conv0", parent_name, lntxt, lnweights)
        return C.writeReLU(convbn._modules['1'], scope, name + "_relu0", "previous", lntxt, lnweights)


def writeInvertedResidual6(ir, use_res_connect, scope, name, parent_name, lntxt, lnweights, useConvRelu):
    if useConvRelu:
        C.writeConvRelu6(ir._modules['0'], scope, name + "_conv0", parent_name, lntxt, lnweights)
        C.writeConvRelu6(ir._modules['2'], scope, name + "_conv1", "previous", lntxt, lnweights)
    else:
        C.writeConv(ir._modules['0'], scope, name + "_conv0", parent_name, lntxt, lnweights)
        C.writeReLU6(ir._modules['1'], scope, name + "_relu0", "previous", lntxt, lnweights)

        C.writeConv(ir._modules['2'], scope, name + "_conv1", "previous", lntxt, lnweights)
        C.writeReLU6(ir._modules['3'], scope, name + "_relu1", "previous", lntxt, lnweights)

    C.writeConv(ir._modules['4'], scope, name + "_conv2", "previous", lntxt, lnweights)

    if use_res_connect:
        return C.writeSum(scope, name + "_sum", parent_name, name + "_conv2", lntxt)

    return name + "_conv2"


def writeIRFBlock(ir, use_res_connect, scope, name, parent_name, lntxt, lnweights, useConvRelu):
    if useConvRelu:
        C.writeConvRelu(ir.pw[0], scope, name + "_conv0", parent_name, lntxt, lnweights)
        C.writeConvRelu(ir.dw, scope, name + "_conv1", "previous", lntxt, lnweights)
    else:
        C.writeConv(ir.pw[0], scope, name + "_conv0", parent_name, lntxt, lnweights)
        C.writeReLU(ir.pw[1], scope, name + "_relu0", "previous", lntxt, lnweights)

        C.writeConv(ir.dw, scope, name + "_conv1", "previous", lntxt, lnweights)

    C.writeConv(ir.pwl, scope, name + "_conv2", "previous", lntxt, lnweights)

    if use_res_connect:
        return C.writeSum(scope, name + "_sum", parent_name, name + "_conv2", lntxt)

    return name + "_conv2"


def writeRPN_Conv(rpn_conv, scope, name, parent_name, lntxt, lnweights, useConvRelu):
    if useConvRelu:
        return C.writeConvRelu6(rpn_conv + "_conv", scope, name, parent_name, lntxt, lnweights)
    else:
        C.writeConv(rpn_conv, scope, name + "_conv", parent_name, lntxt, lnweights)
        return C.writeReLU("stub", scope, name + "_relu", "previous", lntxt, lnweights)


def writeRPN_ClsScore(cls_score, scope, name, parent_name, lntxt, lnweights):
    return C.writeConv(cls_score, scope, name, parent_name, lntxt, lnweights)


def writeRPN_BboxPred(bbox_pred, scope, name, parent_name, lntxt, lnweights):
    return C.writeConv(bbox_pred, scope, name, parent_name, lntxt, lnweights)


def writeLH_FeatureTransform(lh, scope, name, parent_name, lntxt, lnweights):
    C.writeConv(lh.col_conv._modules['0'], scope, name + "_sc_col_conv0", parent_name, lntxt, lnweights)
    col_conv = C.writeConv(lh.col_conv._modules['1'], scope, name + "_sc_col_conv1", "previous", lntxt, lnweights)

    C.writeConv(lh.row_conv._modules['0'], scope, name + "_sc_row_conv0", parent_name, lntxt, lnweights)
    row_conv = C.writeConv(lh.row_conv._modules['1'], scope, name + "_sc_row_conv1", "previous", lntxt, lnweights)

    sum_conv = C.writeSum(scope, name + "_sc_conv_sum", col_conv, row_conv, lntxt)

    return C.writeReLU("", scope, name + "_sc_relu", sum_conv, lntxt, lnweights)


def writeUpscale(upscale, scope, name, parent_name, lntxt, lnweights):
    # upscale_dw = copy.deepcopy(upscale.upconv)
    # upscale_pw = copy.deepcopy(upscale.upconv_pw)
    #
    # bias_dw = copy.deepcopy(upscale_dw.bias.data.cpu().numpy())
    # w_pw = copy.deepcopy(upscale_pw.weight.data.cpu().numpy())
    # bias_pw = copy.deepcopy(upscale_pw.bias.data.cpu().numpy())
    #
    # bias_dw = np.squeeze(bias_dw)
    # bias_dw = bias_dw[:, np.newaxis]
    # w_pw = np.squeeze(w_pw)
    # bias_pw = np.squeeze(bias_pw)
    # bias_pw = bias_pw[:, np.newaxis]
    #
    # new_bias_cw = np.matmul(w_pw, bias_dw) + bias_pw
    #
    # upscale_dw.bias = None
    # upscale_pw.bias.data = torch.tensor(new_bias_cw)

    C.writeConvTransposed(upscale.upconv, scope, name + "_dw", parent_name, lntxt, lnweights)
    C.writeReLU("", scope, name + "_dw_relu", "previous", lntxt, lnweights)

    C.writeConv(upscale.upconv_pw, scope, name + "_pw", "previous", lntxt, lnweights)
    return C.writeReLU("", scope, name + "_relu", "previous", lntxt, lnweights)


def writeMobileRCNN_base(model, lntxt_path, lnweights_path, useConvRelu):
    scope = "OUTPUT"
    with open(lntxt_path, "w+") as lntxt:
        with open(lnweights_path, "wb+") as lnweights:
            # write the body of the backbone 1st
            features = model.backbone[0].features

            name = "data"
            parent_name = C.writeImageInput(name, [640, 480, 3, 1], lntxt)

            name = "stage1"
            modules = features.stage1
            parent_name = writeConvBN6(modules[0].conv, scope, name + "_convbn", parent_name, lntxt, lnweights,
                                      useConvRelu)
            parent_name = writeInvertedResidual6(modules[1].conv, modules[1].use_res_connect, scope,
                                                name + "_ir0", parent_name, lntxt, lnweights, useConvRelu)

            name = "stage2"
            modules = features.stage2
            parent_name = writeInvertedResidual6(modules[0].conv, modules[0].use_res_connect, scope,
                                                name + "_ir0", parent_name, lntxt, lnweights, useConvRelu)
            stage2_out = writeInvertedResidual6(modules[1].conv, modules[1].use_res_connect, scope,
                                               name + "_ir1", parent_name, lntxt, lnweights, useConvRelu)

            name = "stage3"
            modules = features.stage3
            parent_name = writeInvertedResidual6(modules[0].conv, modules[0].use_res_connect, scope,
                                                name + "_ir0", stage2_out, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual6(modules[1].conv, modules[1].use_res_connect, scope,
                                                name + "_ir1", parent_name, lntxt, lnweights, useConvRelu)
            stage3_out = writeInvertedResidual6(modules[2].conv, modules[2].use_res_connect, scope,
                                               name + "_ir2", parent_name, lntxt, lnweights, useConvRelu)

            name = "stage4"
            modules = features.stage4
            parent_name = writeInvertedResidual6(modules[0].conv, modules[0].use_res_connect, scope,
                                                name + "_ir0", stage3_out, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual6(modules[1].conv, modules[1].use_res_connect, scope,
                                                name + "_ir1", parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual6(modules[2].conv, modules[2].use_res_connect, scope,
                                                name + "_ir2", parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual6(modules[3].conv, modules[3].use_res_connect, scope,
                                                name + "_ir3", parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual6(modules[4].conv, modules[4].use_res_connect, scope,
                                                name + "_ir4", parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual6(modules[5].conv, modules[5].use_res_connect, scope,
                                                name + "_ir5", parent_name, lntxt, lnweights, useConvRelu)
            stage4_out = writeInvertedResidual6(modules[6].conv, modules[6].use_res_connect, scope,
                                               name + "_ir6", parent_name, lntxt, lnweights, useConvRelu)

            name = "stage5"
            modules = features.stage5
            parent_name = writeInvertedResidual6(modules[0].conv, modules[0].use_res_connect, scope,
                                                name + "_ir0", stage4_out, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual6(modules[1].conv, modules[1].use_res_connect, scope,
                                                name + "_ir1", parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual6(modules[2].conv, modules[2].use_res_connect, scope,
                                                name + "_ir2", parent_name, lntxt, lnweights, useConvRelu)
            stage5_out = writeInvertedResidual6(modules[3].conv, modules[3].use_res_connect, scope,
                                               name + "_ir3", parent_name, lntxt, lnweights, useConvRelu)

            # next write the FPN part -- this could be done nicer in a for loop
            fpn = model.backbone[1]

            last_inner = C.writeConv(fpn.fpn_inner4, scope, "fpn_inner4", stage5_out, lntxt, lnweights)
            result_4 = C.writeConv(fpn.fpn_layer4, scope, "fpn_layer4", last_inner, lntxt, lnweights)

            inner_top_down = C.writeUpSampleNearest("stub", scope, "fpn_up3", [2, 2], [0, 0], last_inner, lntxt)
            inner_lateral = C.writeConv(fpn.fpn_inner3, scope, "fpn_inner3", stage4_out, lntxt, lnweights)
            last_inner = C.writeSum(scope, "fpn_sum3", inner_top_down, inner_lateral, lntxt)
            result_3 = C.writeConv(fpn.fpn_layer3, scope, "fpn_layer3", last_inner, lntxt, lnweights)

            inner_top_down = C.writeUpSampleNearest("stub", scope, "fpn_up2", [2, 2], [0, 0], last_inner, lntxt)
            inner_lateral = C.writeConv(fpn.fpn_inner2, scope, "fpn_inner2", stage3_out, lntxt, lnweights)
            last_inner = C.writeSum(scope, "fpn_sum2", inner_top_down, inner_lateral, lntxt)
            result_2 = C.writeConv(fpn.fpn_layer2, scope, "fpn_layer2", last_inner, lntxt, lnweights)

            inner_top_down = C.writeUpSampleNearest("stub", scope, "fpn_up1", [2, 2], [0, 0], last_inner, lntxt)
            inner_lateral = C.writeConv(fpn.fpn_inner1, scope, "fpn_inner1", stage2_out, lntxt, lnweights)
            last_inner = C.writeSum(scope, "fpn_sum1", inner_top_down, inner_lateral, lntxt)
            result_1 = C.writeConv(fpn.fpn_layer1, scope, "fpn_layer1", last_inner, lntxt, lnweights)

            result_5 = C.writeMaxPoolFromParams(scope, "fpn_mp", 1, 2, 0, False, result_4, lntxt)

            # next write the RPN part -- this could be a separate file
            rpn = model.rpn.head
            name = "rpm"

            rpn1_conv = writeRPN_Conv(rpn.conv, scope, name + "1", result_1, lntxt, lnweights, useConvRelu)
            C.writeConv(rpn.cls_logits, scope, name + "1_cls_logits", rpn1_conv, lntxt, lnweights)
            C.writeSigmoid(rpn.cls_logits, scope, name + "1_cls_logits_sm", "previous", lntxt, lnweights)
            C.writeConv(rpn.bbox_pred, scope, name + "1_bbox_pred", rpn1_conv, lntxt, lnweights)

            rpn2_conv = writeRPN_Conv(rpn.conv, scope, name + "2", result_2, lntxt, lnweights, useConvRelu)
            C.writeConv(rpn.cls_logits, scope, name + "2_cls_logits", rpn2_conv, lntxt, lnweights)
            C.writeSigmoid(rpn.cls_logits, scope, name + "2_cls_logits_sm", "previous", lntxt, lnweights)
            C.writeConv(rpn.bbox_pred, scope, name + "2_bbox_pred", rpn2_conv, lntxt, lnweights)

            rpn3_conv = writeRPN_Conv(rpn.conv, scope, name + "3", result_3, lntxt, lnweights, useConvRelu)
            C.writeConv(rpn.cls_logits, scope, name + "3_cls_logits", rpn3_conv, lntxt, lnweights)
            C.writeSigmoid(rpn.cls_logits, scope, name + "3_cls_logits_sm", "previous", lntxt, lnweights)
            C.writeConv(rpn.bbox_pred, scope, name + "3_bbox_pred", rpn3_conv, lntxt, lnweights)

            rpn4_conv = writeRPN_Conv(rpn.conv, scope, name + "4", result_4, lntxt, lnweights, useConvRelu)
            C.writeConv(rpn.cls_logits, scope, name + "4_cls_logits", rpn4_conv, lntxt, lnweights)
            C.writeSigmoid(rpn.cls_logits, scope, name + "4_cls_logits_sm", "previous", lntxt, lnweights)
            C.writeConv(rpn.bbox_pred, scope, name + "4_bbox_pred", rpn4_conv, lntxt, lnweights)

            rpn5_conv = writeRPN_Conv(rpn.conv, scope, name + "5", result_5, lntxt, lnweights, useConvRelu)
            C.writeConv(rpn.cls_logits, scope, name + "5_cls_logits", rpn5_conv, lntxt, lnweights)
            C.writeSigmoid(rpn.cls_logits, scope, name + "5_cls_logits_sm", "previous", lntxt, lnweights)
            C.writeConv(rpn.bbox_pred, scope, name + "5_bbox_pred", rpn5_conv, lntxt, lnweights)

    print("write MobileRCNN base done")


def writeSixDNet_base(model, lntxt_path, lnweights_path, useConvRelu):
    with open(lntxt_path, "w+") as lntxt:
        with open(lnweights_path, "wb+") as lnweights:
            # write the body of the backbone

            name = "data"
            parent_name = C.writeImageInput(name, [640, 480, 3, 1], lntxt)

            backbone = model.backbone

            name = "first"
            parent_name = writeConvBN(backbone.first, "INNER", name + "_convbn", parent_name, lntxt, lnweights,
                                      useConvRelu)

            name = "stages"
            stages = backbone.stages
            for n in range(0, 13):
                parent_name = writeIRFBlock(stages[n], stages[n].use_res_connect, "INNER", name + "_" + str(n),
                                            parent_name, lntxt, lnweights, useConvRelu)

            # write the RPN

            name = "rpn_head"
            rpn_head = model.rpn.head[0].head
            parent_name = writeIRFBlock(rpn_head[0], rpn_head[0].use_res_connect, "INNER", name + "_" + str(0),
                                        parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeIRFBlock(rpn_head[1], rpn_head[1].use_res_connect, "INNER", name + "_" + str(1),
                                        parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeIRFBlock(rpn_head[2], rpn_head[2].use_res_connect, "INNER", name + "_" + str(2),
                                        parent_name, lntxt, lnweights, useConvRelu)

            rpn_feats = parent_name

            rpn_regressor = model.rpn.head[1]
            writeRPN_ClsScore(rpn_regressor.cls_logits, "OUTPUT", "rpn_cls_logits", rpn_feats, lntxt, lnweights)
            writeRPN_BboxPred(rpn_regressor.bbox_pred, "OUTPUT", "rpn_bbox_pred", rpn_feats, lntxt, lnweights)

    print("write SixDNet base done")


def writeMobileRCNN_det(model, lntxt_path, lnweights_path, useConvRelu):
    scope = "OUTPUT"
    with open(lntxt_path, "w+") as lntxt:
        with open(lnweights_path, "wb+") as lnweights:
            name = "det"

            #first write the box transform layer
            roialign = {"resolution" : 7, "channels" : 128, "sampling_ratio" : 2}
            C.writeRoiAlign(roialign, scope, name + "_roialign", lntxt)

            #reshape so the fc works
            C.writeReshape([1, 1, roialign["resolution"] * roialign["resolution"] * roialign["channels"]],
                         scope, name + "_reshape", "previous", lntxt)

            #next write what's left of the Box_Head
            parent_dims = [roialign["channels"], model.feature_extractor.fc6.out_features,
                           roialign["resolution"], roialign["resolution"]]

            C.writeFC(model.feature_extractor.fc6, scope, name + "_fc6", "previous", lntxt, lnweights)
            C.writeReLU("", scope, name + "_relu6", "previous", lntxt, lnweights)
            C.writeFC(model.feature_extractor.fc7, scope, name + "_fc7", "previous", lntxt, lnweights)
            parent_name = C.writeReLU("", scope, name + "_relu7", "previous", lntxt, lnweights)

            # the Box_Output part
            cls_score = C.writeFC(model.predictor.cls_score, scope, name + "_cls_score", parent_name, lntxt, lnweights)
            bbox_pred = C.writeFC(model.predictor.bbox_pred, scope, name + "_bbox_pred", parent_name, lntxt, lnweights)

            # the softmax over class scores (could potentially be remove)
            sm_cls_score = C.writeSoftMax(scope, name + "_sm_cls_score", cls_score, lntxt)

            # the output nodes
            C.writeCopyOutput(name + "_copy_cls_score", sm_cls_score, lntxt)
            C.writeCopyOutput(name + "_copy_bbox_pred", bbox_pred, lntxt)

    print("write MobileRCNN det done")


def writeMobileRCNN_mask(model, lntxt_path, lnweights_path, useConvRelu):
    scope = "OUTPUT"
    with open(lntxt_path, "w+") as lntxt:
        with open(lnweights_path, "wb+") as lnweights:
            name = "mask"

            # first write the box transform layer
            roialign = {"resolution": 14, "channels": 128, "sampling_ratio": 2}
            C.writeRoiAlign(roialign, scope, name + "_roialign", lntxt)

            # write the mask_fcn
            C.writeConv(model.feature_extractor.mask_fcn1, scope, name + "_fcn1", "previous", lntxt, lnweights)
            C.writeReLU("", scope, name + "_relu_fcn1", "previous", lntxt, lnweights)

            C.writeConv(model.feature_extractor.mask_fcn2, scope, name + "_fcn2", "previous", lntxt, lnweights)
            C.writeReLU("", scope, name + "_relu_fcn2", "previous", lntxt, lnweights)

            C.writeConv(model.feature_extractor.mask_fcn3, scope, name + "_fcn3", "previous", lntxt, lnweights)
            C.writeReLU("", scope, name + "_relu_fcn3", "previous", lntxt, lnweights)

            C.writeConv(model.feature_extractor.mask_fcn4, scope, name + "_fcn4", "previous", lntxt, lnweights)
            C.writeReLU("", scope, name + "_relu_fcn4", "previous", lntxt, lnweights)

            # write the predictor
            C.writeConvTransposed(model.predictor.conv5_mask, scope, name + "_conv5", "previous", lntxt, lnweights)
            C.writeReLU("", scope, name + "_relu_conv5", "previous", lntxt, lnweights)

            C.writeConv(model.predictor.mask_fcn_logits, scope, name + "_fcn_logits", "previous", lntxt, lnweights)

            # write post processor
            C.writeSigmoid("", scope, name + "_sigmoid", "previous", lntxt, lnweights)

            # the output nodes
            C.writeCopyOutput(name + "_segmentation", "previous", lntxt)

    print("write MobileRCNN mask done")
