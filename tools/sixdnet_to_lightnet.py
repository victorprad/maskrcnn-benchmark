from tools.pytorch_to_lightnet import PytorchToLightNet as C


def writeConvBN(convbn, scope, name, parent_name, lntxt, lnweights, useConvRelu):
    if useConvRelu:
        return C.writeConvRelu(convbn._modules['0'], scope, name + "_conv0", parent_name, lntxt, lnweights)
    else:
        C.writeConv(convbn._modules['0'], scope, name + "_conv0", parent_name, lntxt, lnweights)
        return C.writeReLU(convbn._modules['1'], scope, name + "_relu0", "previous", lntxt, lnweights)


def writeIRFBlock(ir, use_res_connect, scope, name, parent_name, lntxt, lnweights, useConvRelu):
    if useConvRelu:
        C.writeConvRelu(ir.pw[0], scope, name + "_conv0", parent_name, lntxt, lnweights)

        if ir.upscale is not None:
            C.writeUpSampleNearest(ir.upscale, scope, name + "_up", [ir.upscale.scale, ir.upscale.scale], [0, 0],
                                   "previous", lntxt)

        C.writeConvRelu(ir.dw, scope, name + "_conv1", "previous", lntxt, lnweights)
    else:
        C.writeConv(ir.pw[0], scope, name + "_conv0", parent_name, lntxt, lnweights)
        C.writeReLU(ir.pw[1], scope, name + "_relu0", "previous", lntxt, lnweights)

        if ir.upscale is not None:
            C.writeUpSampleNearest(ir.upscale, scope, name + "_up", [ir.upscale.scale, ir.upscale.scale], [0, 0],
                                   "previous", lntxt)

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


def writeSixDNet_det(model, lntxt_path, lnweights_path, useConvRelu):
    with open(lntxt_path, "w+") as lntxt:
        with open(lnweights_path, "wb+") as lnweights:
            name = "det"

            # first write the roi align
            roialign = {"resolution" : 6, "channels" : 96, "sampling_ratio" : 0}
            parent_name = C.writeRoiAlign(roialign, "INNER", name + "_roialign", lntxt)

            # next write the feature extrator
            box_head = model.roi_heads.box.feature_extractor.head[0]
            parent_name = writeIRFBlock(box_head[0], box_head[0].use_res_connect, "INNER", name + "_" + str(0),
                                        parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeIRFBlock(box_head[1], box_head[1].use_res_connect, "INNER", name + "_" + str(1),
                                        parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeIRFBlock(box_head[2], box_head[2].use_res_connect, "INNER", name + "_" + str(2),
                                        parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeIRFBlock(box_head[3], box_head[3].use_res_connect, "INNER", name + "_" + str(3),
                                        parent_name, lntxt, lnweights, useConvRelu)

            # next write the predictor
            predictor = model.roi_heads.box.predictor
            parent_name = C.writeAvgPoolFromParams("INNER", name + "_avgpool", 3, 1, 0, False, parent_name, lntxt)

            cls_score = C.writeFC(predictor.cls_score, "INNER", name + "_cls_score", parent_name, lntxt, lnweights)
            bbox_pred = C.writeFC(predictor.bbox_pred, "INNER", name + "_bbox_pred", parent_name, lntxt, lnweights)

            # the softmax over class scores (could potentially be removed)
            sm_cls_score = C.writeSoftMax("INNER", name + "_sm_cls_score", cls_score, lntxt)

            # the output nodes
            C.writeCopyOutput(name + "_copy_cls_score", sm_cls_score, lntxt)
            C.writeCopyOutput(name + "_copy_bbox_pred", bbox_pred, lntxt)

    print("write SixDNet det done")


def writeSixDNet_mask(model, lntxt_path, lnweights_path, useConvRelu):
    with open(lntxt_path, "w+") as lntxt:
        with open(lnweights_path, "wb+") as lnweights:
            name = "mask"

            # first write the roi align
            roialign = {"resolution" : 6, "channels" : 96, "sampling_ratio" : 0}
            parent_name = C.writeRoiAlign(roialign, "INNER", name + "_roialign", lntxt)

            # next write the feature extrator
            mask_head = model.roi_heads.mask.feature_extractor.head[0]
            parent_name = writeIRFBlock(mask_head[0], mask_head[0].use_res_connect, "INNER", name + "_" + str(0),
                                        parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeIRFBlock(mask_head[1], mask_head[1].use_res_connect, "INNER", name + "_" + str(1),
                                        parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeIRFBlock(mask_head[2], mask_head[2].use_res_connect, "INNER", name + "_" + str(2),
                                        parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeIRFBlock(mask_head[3], mask_head[3].use_res_connect, "INNER", name + "_" + str(3),
                                        parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeIRFBlock(mask_head[4], mask_head[4].use_res_connect, "INNER", name + "_" + str(4),
                                        parent_name, lntxt, lnweights, useConvRelu)

            # next write the predictor
            predictor = model.roi_heads.mask.predictor
            C.writeConv(predictor.mask_fcn_logits, "INNER", name + "_fcn_logits", "previous", lntxt, lnweights)

            # write post processor
            C.writeSigmoid("", "INNER", name + "_sigmoid", "previous", lntxt, lnweights)

            # # the output node
            C.writeCopyOutput(name + "_segmentation", "previous", lntxt)

    print("write SixDNet mask done")